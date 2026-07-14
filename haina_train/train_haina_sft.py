#!/usr/bin/env python3
"""SFT trainer for HainaOCR NativePixel — ChatML 多轮对话，图文+纯文本混合训练。

用法:
  cd haina_train
  source MX_env.sh
  NPROC_PER_NODE=8 bash run_sft.sh

与 CPT trainer 的核心差异:
  1. ChatML 格式: <|im_start|>system/user/assistant\n...<|im_end|>
  2. Labels: 仅 assistant 消息参与 loss 计算
  3. 图文 + 纯文本混合: 自动处理有无图片的样本
  4. 图片缺失自动跳过
"""

from __future__ import annotations

import argparse, glob, json, logging, math, os, random, shutil, sys, time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch, torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required.") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hainaocr_nativepixel import (
    HainaOCRNativePixelConfig,
    HainaOCRNativePixelForConditionalGeneration,
    HainaOCRNativePixelImageProcessor,
)
from hainaocr_nativepixel.processing import load_tokenizer_with_fixes

# ── Token IDs ──
VISION_START = 151652
VISION_END   = 151653
IMAGE_PAD    = 151655
IM_START     = 151644  # <|im_start|>
IM_END       = 151645  # <|im_end|>
IGNORE_ID    = -100

SYSTEM_PROMPT = "You are a helpful assistant."


# ═══════════════════════════════════════════════════════════════
# Default config (merged with user YAML)
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "model_dir": str(PROJECT_ROOT),
        "llm_path": "/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b",
        "base_checkpoint": "",                   # CPT base weights path (model.safetensors)
        "attn_implementation": "eager",
        "torch_dtype": "bfloat16",
        "use_liger_ce": False,
        "gradient_checkpointing": True,
        "compile": False,
        "train_vision_encoder": True,
        "train_pos_embed": True,
        "train_qwen3": True,
        "train_lm_head": True,
    },
    "data": {
        "image_jsonl_glob": "",                  # VQA 图文数据
        "text_jsonl_glob": "",                   # PEFT 纯文本数据
        "max_length": 4096,
        "shuffle_files": True,
        "skip_bad_samples": True,
        "image_open_retries": 2,
        "cpu_patchify": False,
        "num_workers": 4,
        "prefetch_factor": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "text_ratio": 0.3,                       # 纯文本样本比例 (0=只用图文, 1=只用纯文本)
    },
    "training": {
        "seed": 42,
        "epochs": 2,
        "micro_batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 2.0e-5,
        "min_lr": 0.0,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "warmup_steps": 0,
        "lr_schedule": "cosine",
        "precision": "bf16",
        "log_interval": 10,
        "save_interval": 500,
        "save_each_epoch": True,
        "keep_last_n": 3,
        "resume": "auto",
        "save_final": True,
        "detect_anomaly": False,
        "empty_cache_interval": 0,
        "throughput_window": 50,
    },
    "ddp": {
        "find_unused_parameters": True,
        "static_graph": False,
        "gradient_as_bucket_view": True,
        "broadcast_buffers": False,
    },
    "profile": {
        "enabled": False, "start_step": 10, "num_steps": 20,
        "record_shapes": False, "with_stack": False, "export_chrome_trace": True,
    },
    "output": {
        "output_dir": str(SCRIPT_DIR / "outputs" / "sft"),
        "log_dir": "",
    },
}


# ═══════════════════════════════════════════════════════════════
# Config helpers
# ═══════════════════════════════════════════════════════════════

def deep_update(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return deep_update(DEFAULT_CONFIG, yaml.safe_load(f) or {})

def setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}

def is_dist(): return dist.is_available() and dist.is_initialized()
def barrier():
    if is_dist(): dist.barrier()
def is_main(rank): return rank == 0

def seed_everything(seed, rank):
    seed = int(seed) + int(rank)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def dtype_from_name(name):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}: return torch.bfloat16
    if name in {"fp16", "float16", "half"}: return torch.float16
    if name in {"fp32", "float32"}: return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")

def setup_logging(output_dir, rank):
    log_dir = output_dir / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("haina_sft")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout); stream.setFormatter(fmt)
    logger.addHandler(stream)
    if rank == 0:
        fh = logging.FileHandler(log_dir / "train.log", encoding="utf-8"); fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger

@contextmanager
def timer_bucket(stats, name):
    t0 = time.perf_counter()
    try: yield
    finally: stats[name] += time.perf_counter() - t0


# ═══════════════════════════════════════════════════════════════
# SFT Dataset — ChatML 多轮格式
# ═══════════════════════════════════════════════════════════════

class JsonlSftDataset(IterableDataset):
    """Stream JSONL (VQA-style or PEFT-style), produce SFT ChatML samples."""

    def __init__(self, cfg, tokenizer, image_processor, rank, world_size, epoch=0):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.rank = rank; self.world_size = world_size; self.epoch = epoch
        self.max_length = int(cfg.get("max_length", 4096))
        self.text_ratio = float(cfg.get("text_ratio", 0.3))
        self.is_pure_text = False  # set per-file

        # Collect files
        self.img_files = sorted(glob.glob(str(cfg["image_jsonl_glob"]), recursive=True)) if cfg.get("image_jsonl_glob") else []
        self.txt_files = sorted(glob.glob(str(cfg["text_jsonl_glob"]), recursive=True)) if cfg.get("text_jsonl_glob") else []

        # Encode special tokens once
        self.system_ids = self._encode_ids(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")
        self.im_start_user = self._encode_ids("<|im_start|>user\n")
        self.im_end_n = self._encode_ids("<|im_end|>\n")
        self.im_start_asst = self._encode_ids("<|im_start|>assistant\n")
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.eos_id = tokenizer.eos_token_id

    def _encode_ids(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _shuffled(self, files):
        files = list(files)
        if self.cfg.get("shuffle_files", True):
            rng = random.Random(int(self.cfg.get("seed", 42)) + self.epoch)
            rng.shuffle(files)
        return files

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        shard_id = self.rank * num_workers + worker_id
        num_shards = max(1, self.world_size * num_workers)
        rng = random.Random(int(self.cfg.get("seed", 42)) + self.epoch + shard_id)

        img_files = self._shuffled(self.img_files)
        txt_files = self._shuffled(self.txt_files)
        img_iter = self._iter_files(img_files, shard_id, num_shards, rng) if img_files else None
        txt_iter = self._iter_files(txt_files, shard_id, num_shards, rng) if txt_files else None

        has_img = img_iter is not None
        has_txt = txt_iter is not None

        while True:
            want_text = has_txt and (not has_img or rng.random() < self.text_ratio)
            if want_text:
                try:
                    sample = next(txt_iter)
                    if sample is not None:
                        yield sample
                    continue
                except StopIteration:
                    has_txt = False
                    if not has_img:
                        return
                    continue
            else:
                if has_img:
                    try:
                        sample = next(img_iter)
                        if sample is not None:
                            yield sample
                        continue
                    except StopIteration:
                        has_img = False
                        if not has_txt:
                            return
                        continue
                elif has_txt:
                    try:
                        sample = next(txt_iter)
                        if sample is not None:
                            yield sample
                        continue
                    except StopIteration:
                        return
                else:
                    return

    def _iter_files(self, files, shard_id, num_shards, rng):
        row_index = 0
        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if row_index % num_shards != shard_id:
                        row_index += 1
                        continue
                    row_index += 1
                    if not line.strip():
                        continue
                    sample = self._process_line(line)
                    if sample is not None:
                        yield sample

    # ── Line → sample ──────────────────────────────────────────

    def _process_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            row = json.loads(line)
            messages = row.get("messages") or []
            images = row.get("images") or []
            return self._build_chatml(messages, images)
        except Exception:
            if self.cfg.get("skip_bad_samples", True):
                return None
            raise

    def _build_chatml(self, messages, images) -> Optional[Dict[str, Any]]:
        """Build ChatML input_ids + labels from messages, with optional image."""

        # ── Handle image (if present) ──
        pixel_values = None; image_grid_thw = None; num_vision_tokens = 0
        if images:
            image_path = images[0]
            img = None
            for _ in range(max(1, int(self.cfg.get("image_open_retries", 1)))):
                try:
                    img = Image.open(image_path).convert("RGB")
                    break
                except Exception:
                    pass
            if img is None:
                return None  # skip missing images
            image_inputs = self.image_processor.preprocess(images=img, return_tensors="pt",
                                                           cpu_patchify=bool(self.cfg.get("cpu_patchify", False)))
            pixel_values = image_inputs["pixel_values"]
            if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
                pixel_values = pixel_values[0]
            grid = image_inputs["image_grid_thw"][0]
            grid = [int(x) for x in grid]
            image_grid_thw = torch.tensor(grid, dtype=torch.long)
            num_vision_tokens = int(grid[0]) * int(grid[1]) * int(grid[2])

        # ── Build ChatML token sequence ──
        all_ids = []
        all_labels = []

        # System message
        all_ids.extend(self.system_ids)
        all_labels.extend([IGNORE_ID] * len(self.system_ids))

        for turn_idx, msg in enumerate(messages):
            role = msg.get("role", "")
            content = str(msg.get("content", ""))

            if role == "user":
                # Strip <image> marker, replace with vision tokens in first user message
                content = content.replace("<image>\n", "").replace("<image>", "").strip()
                if not content:
                    continue

                all_ids.extend(self.im_start_user)

                if turn_idx == 0 and num_vision_tokens > 0:
                    # Insert vision tokens in the first user message
                    vision_ids = [VISION_START] + [IMAGE_PAD] * num_vision_tokens + [VISION_END]
                    all_ids.extend(vision_ids)
                    all_labels.extend([IGNORE_ID] * len(self.im_start_user))
                    all_labels.extend([IGNORE_ID] * len(vision_ids))
                else:
                    all_labels.extend([IGNORE_ID] * len(self.im_start_user))

                content_ids = self._encode_ids(content)
                all_ids.extend(content_ids)
                if turn_idx == 0 and num_vision_tokens > 0:
                    all_labels.extend([IGNORE_ID] * len(content_ids))
                else:
                    all_labels.extend([IGNORE_ID] * len(content_ids))

                all_ids.extend(self.im_end_n)
                all_labels.extend([IGNORE_ID] * len(self.im_end_n))

            elif role == "assistant":
                content = content.strip()
                if not content:
                    continue

                all_ids.extend(self.im_start_asst)
                all_labels.extend([IGNORE_ID] * len(self.im_start_asst))  # mask the prefix

                content_ids = self._encode_ids(content)
                all_ids.extend(content_ids)
                all_labels.extend(content_ids)  # ← ASSISTANT CONTENT GETS LOSS

                all_ids.extend(self.im_end_n)
                all_labels.extend(content_ids[:0] + [IGNORE_ID] * len(self.im_end_n))  # mask <|im_end|>

        # ── Truncate to max_length ──
        if len(all_ids) > self.max_length:
            all_ids = all_ids[:self.max_length]
            all_labels = all_labels[:self.max_length]

        if len(all_ids) < 4:
            return None

        input_ids = torch.tensor(all_ids, dtype=torch.long)
        labels = torch.tensor(all_labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "num_tokens": int(input_ids.numel()),
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.contiguous()
            result["image_grid_thw"] = image_grid_thw
        return result


# ═══════════════════════════════════════════════════════════════
# Collate — handle mixed image / pure text batches
# ═══════════════════════════════════════════════════════════════

def collate_sft(samples, pad_token_id):
    max_len = max(s["input_ids"].numel() for s in samples)
    batch = len(samples)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, max_len), IGNORE_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, s in enumerate(samples):
        n = s["input_ids"].numel()
        input_ids[i, :n] = s["input_ids"]
        labels[i, :n] = s["labels"]
        attention_mask[i, :n] = s["attention_mask"]

    pixel_values = None; image_grid_thw = None
    img_samples = [s for s in samples if "pixel_values" in s]
    if img_samples:
        first_pv = img_samples[0]["pixel_values"]
        if first_pv.ndim == 3:
            c = first_pv.shape[0]
            max_h = max(s["pixel_values"].shape[1] for s in img_samples)
            max_w = max(s["pixel_values"].shape[2] for s in img_samples)
            pixel_values = first_pv.new_zeros((len(img_samples), c, max_h, max_w))
            for i, s in enumerate(img_samples):
                pv = s["pixel_values"]
                _, h, w = pv.shape
                pixel_values[i, :, :h, :w] = pv
        else:
            pixel_values = torch.cat([s["pixel_values"] for s in img_samples], dim=0)
        image_grid_thw = torch.stack([s["image_grid_thw"] for s in img_samples], dim=0)

    num_tokens = torch.tensor([s["num_tokens"] for s in samples], dtype=torch.long)
    return {
        "input_ids": input_ids, "labels": labels, "attention_mask": attention_mask,
        "pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "num_tokens": num_tokens,
    }


# ═══════════════════════════════════════════════════════════════
# Model / Optimizer / Scheduler
# ═══════════════════════════════════════════════════════════════

def build_model(cfg, device, logger):
    model_cfg = cfg["model"]
    dtype = dtype_from_name(model_cfg.get("torch_dtype", "bfloat16"))
    config = HainaOCRNativePixelConfig.from_pretrained(model_cfg["model_dir"], trust_remote_code=True)
    config._attn_implementation = model_cfg.get("attn_implementation", "eager")
    config.use_liger_ce = bool(model_cfg.get("use_liger_ce", False))
    model = HainaOCRNativePixelForConditionalGeneration(config)

    # Load base checkpoint (CPT weights)
    base_ckpt = model_cfg.get("base_checkpoint", "")
    if base_ckpt and Path(base_ckpt).exists():
        sd = load_file(base_ckpt) if base_ckpt.endswith(".safetensors") else torch.load(base_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info("Loaded base checkpoint: %s | missing=%d unexpected=%d", base_ckpt, len(missing), len(unexpected))
    elif Path(model_cfg["model_dir"], "model.safetensors").exists():
        sd = load_file(str(Path(model_cfg["model_dir"]) / "model.safetensors"))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info("Loaded model from model_dir | missing=%d unexpected=%d", len(missing), len(unexpected))

    set_trainable(model, model_cfg)
    if bool(model_cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model = model.to(device=device, dtype=dtype)
    if bool(model_cfg.get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model

def set_trainable(model, model_cfg):
    modules = {
        "vision_encoder": bool(model_cfg.get("train_vision_encoder", True)),
        "pos_embed_2d": bool(model_cfg.get("train_pos_embed", True)),
        "qwen3": bool(model_cfg.get("train_qwen3", True)),
        "lm_head": bool(model_cfg.get("train_lm_head", True)),
    }
    for name, param in model.named_parameters():
        train = False
        if ".vision_encoder." in name or name.startswith("model.vision_encoder."):
            train = modules["vision_encoder"]
        elif ".pos_embed_2d." in name or name.startswith("model.pos_embed_2d."):
            train = modules["pos_embed_2d"]
        elif ".qwen3." in name or name.startswith("model.qwen3."):
            train = modules["qwen3"]
        elif name.startswith("lm_head."):
            train = modules["lm_head"]
        param.requires_grad = train

def build_optimizer(model, cfg):
    train_cfg = cfg["training"]
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if name.endswith(".bias") or "norm" in name.lower() or "rms" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": float(train_cfg["weight_decay"])},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(train_cfg["lr"]), betas=tuple(float(x) for x in train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
    )

def build_scheduler(optimizer, cfg, total_steps):
    train_cfg = cfg["training"]
    warmup = int(train_cfg.get("warmup_steps", 0) or 0)
    if warmup <= 0: warmup = int(total_steps * float(train_cfg.get("warmup_ratio", 0.0)))
    min_ratio = float(train_cfg.get("min_lr", 0.0)) / max(float(train_cfg["lr"]), 1e-20)
    schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()

    def lr_lambda(step):
        if step < warmup: return max(1e-8, step / max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        if schedule == "linear": value = 1.0 - progress
        elif schedule == "constant": value = 1.0
        else: value = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_ratio, value)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ═══════════════════════════════════════════════════════════════
# Checkpoint
# ═══════════════════════════════════════════════════════════════

def save_checkpoint(output_dir, step, epoch, model, optimizer, scheduler, cfg, logger):
    output_dir.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    ckpt_dir = output_dir / f"checkpoint-{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in core.state_dict().items()}
    save_file(state, str(ckpt_dir / "model.safetensors"))
    torch.save({"step": step, "epoch": epoch, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "config": cfg}, ckpt_dir / "trainer_state.pt")
    latest = output_dir / "checkpoint-latest"
    tmp = output_dir / "checkpoint-latest.tmp"
    for p in [tmp, latest]:
        if p.exists() or p.is_symlink(): p.unlink()
    try:
        tmp.symlink_to(ckpt_dir.name, target_is_directory=True)
        tmp.rename(latest)
    except OSError:
        shutil.copytree(ckpt_dir, str(latest))
    keep = int(cfg["training"].get("keep_last_n", 3))
    ckpts = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-") and p.name != "checkpoint-latest"], key=lambda p: p.name)
    for old in ckpts[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    logger.info("Saved checkpoint: %s", ckpt_dir)

def load_resume(output_dir, model, optimizer, scheduler, resume, logger):
    if not resume or str(resume).lower() in {"false", "none", "0"}:
        return 0, 0
    ckpt = output_dir / "checkpoint-latest" if str(resume).lower() == "auto" else Path(resume)
    sp = ckpt / "trainer_state.pt"; mp = ckpt / "model.safetensors"
    if not sp.exists() or not mp.exists():
        logger.info("No resume checkpoint at %s", ckpt); return 0, 0
    core = model.module if hasattr(model, "module") else model
    missing, unexpected = core.load_state_dict(load_file(str(mp)), strict=False)
    state = torch.load(sp, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info("Resumed %s | step=%s epoch=%s missing=%d unexpected=%d", ckpt, state["step"], state["epoch"], len(missing), len(unexpected))
    return int(state["step"]), int(state["epoch"])


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════

def make_profiler(cfg, output_dir):
    if not cfg["profile"].get("enabled", False): return None
    trace_dir = output_dir / "profile"; trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=cfg["profile"].get("start_step",10), warmup=1, active=cfg["profile"].get("num_steps",20), repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=cfg["profile"].get("record_shapes", False),
        with_stack=cfg["profile"].get("with_stack", False),
    )

def train(cfg):
    dist_info = setup_dist()
    rank, world_size, local_rank = dist_info["rank"], dist_info["world_size"], dist_info["local_rank"]
    seed_everything(cfg["training"]["seed"], rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["output"]["output_dir"]).expanduser(); output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, rank)
    if is_main(rank):
        with open(output_dir / "resolved_config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    logger.info("rank=%d/%d local_rank=%d device=%s", rank, world_size, local_rank, device)

    tokenizer = load_tokenizer_with_fixes(cfg["model"]["llm_path"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = HainaOCRNativePixelImageProcessor.from_pretrained(cfg["model"]["model_dir"])
    image_processor.cpu_patchify = bool(cfg["data"].get("cpu_patchify", False))

    model = build_model(cfg, device, logger)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("parameters total=%d trainable=%d ratio=%.2f%%", total, trainable, 100.*trainable/max(1,total))
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=bool(cfg["ddp"].get("find_unused_parameters", True)),
                    static_graph=bool(cfg["ddp"].get("static_graph", False)),
                    gradient_as_bucket_view=bool(cfg["ddp"].get("gradient_as_bucket_view", True)),
                    broadcast_buffers=bool(cfg["ddp"].get("broadcast_buffers", False)))

    # Total steps: use samples_per_epoch if set, otherwise estimate
    est_samples = int(cfg["data"].get("samples_per_epoch", 0) or 0)
    if est_samples <= 0:
        est_samples = 700_000  # rough default
    effective = int(cfg["training"]["micro_batch_size"]) * int(cfg["training"]["grad_accum_steps"]) * max(1, world_size)
    steps_per_epoch = max(1, math.ceil(est_samples / max(1, effective)))
    total_steps = steps_per_epoch * int(cfg["training"]["epochs"])

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    start_step, start_epoch = load_resume(output_dir, model, optimizer, scheduler, cfg["training"].get("resume","auto"), logger)

    writer = None
    if is_main(rank):
        writer = SummaryWriter(cfg["output"].get("log_dir") or str(output_dir / "tb"))
        logger.info("epochs=%d steps_per_epoch=%d total_steps=%d output=%s", cfg["training"]["epochs"], steps_per_epoch, total_steps, output_dir)

    precision = str(cfg["training"].get("precision", "bf16")).lower()
    amp_dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
    use_amp = torch.cuda.is_available() and precision in {"bf16", "bfloat16", "fp16", "float16"}
    scaler = torch.amp.GradScaler("cuda", enabled=precision in {"fp16", "float16"})

    global_step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, disable=not is_main(rank), dynamic_ncols=True, desc="sft")
    model.train()
    stats = defaultdict(float)
    tokens_window = 0; samples_window = 0; window_t0 = time.perf_counter()

    profiler = make_profiler(cfg, output_dir)
    if profiler: profiler.__enter__()

    try:
        for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
            data_cfg = dict(cfg["data"])
            data_cfg["seed"] = int(cfg["training"]["seed"])
            dataset = JsonlSftDataset(data_cfg, tokenizer, image_processor, rank, world_size, epoch=epoch)
            loader = DataLoader(
                dataset,
                batch_size=int(cfg["training"]["micro_batch_size"]),
                collate_fn=lambda b: collate_sft(b, tokenizer.pad_token_id),
                num_workers=int(cfg["data"].get("num_workers", 0)),
                prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)) if int(cfg["data"].get("num_workers",0))>0 else None,
                pin_memory=bool(cfg["data"].get("pin_memory", True)),
                persistent_workers=bool(cfg["data"].get("persistent_workers", True)) and int(cfg["data"].get("num_workers",0))>0,
            )
            accum_loss = 0.0; accum_loss_count = 0
            micro_loss_sum = 0.0; micro_loss_count = 0; micro = 0
            optimizer.zero_grad(set_to_none=True)

            for batch in loader:
                if global_step >= total_steps: break

                with timer_bucket(stats, "to_device"):
                    batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
                    if batch.get("pixel_values") is not None:
                        batch["pixel_values"] = batch["pixel_values"].to(dtype=dtype_from_name(cfg["model"]["torch_dtype"]))

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with timer_bucket(stats, "forward_backward"):
                    with amp_ctx:
                        out = model(
                            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                            pixel_values=batch.get("pixel_values"),
                            image_grid_thw=batch.get("image_grid_thw"),
                            labels=batch["labels"], use_cache=False,
                        )
                        loss = out.loss / int(cfg["training"]["grad_accum_steps"])
                    if scaler.is_enabled(): scaler.scale(loss).backward()
                    else: loss.backward()

                micro_loss_sum += float(loss.detach().cpu()) * int(cfg["training"]["grad_accum_steps"])
                micro_loss_count += 1
                tokens_window += int(batch["num_tokens"].sum().item())
                samples_window += int(batch["input_ids"].shape[0])
                micro += 1

                if micro >= int(cfg["training"]["grad_accum_steps"]):
                    step_loss = micro_loss_sum / max(1, micro_loss_count)
                    with timer_bucket(stats, "optimizer"):
                        if scaler.is_enabled(): scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], float(cfg["training"]["max_grad_norm"]))
                        if scaler.is_enabled(): scaler.step(optimizer); scaler.update()
                        else: optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    global_step += 1; micro = 0
                    micro_loss_sum = 0.0; micro_loss_count = 0
                    accum_loss += step_loss; accum_loss_count += 1

                    if profiler: profiler.step()
                    pbar.update(1)

                    if global_step % int(cfg["training"]["log_interval"]) == 0:
                        elapsed = max(1e-9, time.perf_counter() - window_t0)
                        tok_s = tokens_window / elapsed; samp_s = samples_window / elapsed
                        mem = torch.cuda.max_memory_allocated(device)/1024**3 if torch.cuda.is_available() else 0
                        if is_main(rank):
                            avg = accum_loss/max(1,accum_loss_count)
                            logger.info("epoch=%d step=%d/%d loss=%.5f lr=%.3e grad_norm=%.3f samples/s=%.2f tokens/s=%.0f mem=%.2fGB time=%s",
                                epoch+1, global_step, total_steps, avg, scheduler.get_last_lr()[0], float(grad_norm),
                                samp_s, tok_s, mem, dict((k,round(v,3)) for k,v in stats.items()))
                            if writer:
                                writer.add_scalar("train/loss", avg, global_step)
                                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                        accum_loss = 0.0; accum_loss_count = 0
                        tokens_window = 0; samples_window = 0; stats.clear(); window_t0 = time.perf_counter()

                    if int(cfg["training"].get("save_interval",0))>0 and global_step%int(cfg["training"]["save_interval"])==0:
                        barrier()
                        if is_main(rank): save_checkpoint(output_dir, global_step, epoch, model, optimizer, scheduler, cfg, logger)
                        barrier()

                    if int(cfg["training"].get("empty_cache_interval",0))>0 and global_step%int(cfg["training"]["empty_cache_interval"])==0:
                        torch.cuda.empty_cache()

            if bool(cfg["training"].get("save_each_epoch", True)):
                barrier()
                if is_main(rank): save_checkpoint(output_dir, global_step, epoch+1, model, optimizer, scheduler, cfg, logger)
                barrier()
    finally:
        if profiler: profiler.__exit__(None, None, None)
        pbar.close()
        if writer: writer.close()
        if is_dist(): dist.destroy_process_group()

    if is_main(rank) and bool(cfg["training"].get("save_final", True)):
        save_checkpoint(output_dir, global_step, int(cfg["training"]["epochs"]), model, optimizer, scheduler, cfg, logger)
        logger.info("done step=%d output=%s", global_step, output_dir)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(SCRIPT_DIR/"config_sft.yaml"), help="YAML config")
    p.add_argument("--set", action="append", default=[], help="key=value overrides")
    return p.parse_args()

def apply_overrides(cfg, overrides):
    for item in overrides:
        if "=" not in item: raise ValueError(f"Invalid override: {item}")
        key, raw = item.split("=", 1); value = yaml.safe_load(raw)
        cursor = cfg
        for part in key.split(".")[:-1]: cursor = cursor.setdefault(part, {})
        cursor[key.split(".")[-1]] = value
    return cfg

if __name__ == "__main__":
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if config["training"].get("detect_anomaly", False): torch.autograd.set_detect_anomaly(True)
    train(config)
