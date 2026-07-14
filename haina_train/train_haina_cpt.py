#!/usr/bin/env python3
"""Stable CPT trainer for HainaOCR NativePixel.

The script is intentionally self-contained so it can run outside Swift while
keeping the same verified CPT sample format used by swift_arrow_train.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

try:
    import yaml
except Exception as exc:  # pragma: no cover - fail early with a helpful message
    raise RuntimeError("PyYAML is required. Install pyyaml or use the project env.") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hainaocr_nativepixel import (  # noqa: E402
    HainaOCRNativePixelConfig,
    HainaOCRNativePixelForConditionalGeneration,
    HainaOCRNativePixelImageProcessor,
)
from hainaocr_nativepixel.processing import load_tokenizer_with_fixes  # noqa: E402

# ── HTML online render pipeline (text2vision_pt) ──
_T2V_DIR = SCRIPT_DIR.parent / "data" / "text2vision_pt"
if str(_T2V_DIR) not in sys.path:
    sys.path.insert(0, str(_T2V_DIR))
_has_html_render = True
try:
    from dataset.render_dataset import RenderDataset  # noqa: E402
    from dataset.render_collator import RenderCollator  # noqa: E402
    from rendering.html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig  # noqa: E402
except Exception:
    _has_html_render = False
    RenderDataset = None
    RenderCollator = None
    HtmlOCRRenderer = None


VISION_START = 151652
VISION_END = 151653
IMAGE_PAD = 151655


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "model_dir": str(PROJECT_ROOT),
        "llm_path": "/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b",
        "cnn_weights": str(PROJECT_ROOT / "cnn_weights.safetensors"),
        "attn_implementation": "eager",
        "torch_dtype": "bfloat16",
        "use_liger_ce": False,
        "gradient_checkpointing": False,
        "compile": False,
        "train_vision_encoder": False,
        "train_pos_embed": True,
        "train_vision_boundary": True,
        "train_qwen3": True,
        "train_qwen3_layers": None,         # list of layer indices, e.g. [24,25,26,27]
        "train_qwen3_embed_tokens": False,
        "train_qwen3_final_norm": True,
        "train_lm_head": True,
        "base_checkpoint": "",              # path to stage-1 model.safetensors
    },
    "data": {
        "jsonl_glob": "/mnt/si001719kd1w/default/xjz/data/swift_jsonl/train.jsonl",
        "text_jsonl_path": "",              # text-only data for mixing (e.g. onesci_cc_150k)
        "text_ratio": 0.0,                  # 0.0=image-only, 0.2=20% text-only
        "max_length": 4096,
        "samples_per_epoch": 0,
        "shuffle_files": True,
        "shuffle_buffer": 0,
        "skip_bad_samples": True,
        "image_open_retries": 1,
        "cpu_patchify": False,
        "num_workers": 4,
        "prefetch_factor": 4,
        "pin_memory": True,
        "persistent_workers": True,
    },
    "training": {
        "seed": 42,
        "epochs": 3,
        "micro_batch_size": 2,
        "grad_accum_steps": 16,
        "lr": 1.0e-5,
        "min_lr": 0.0,
        "weight_decay": 0.1,
        "betas": [0.9, 0.95],
        "eps": 1.0e-8,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.01,
        "warmup_steps": 0,
        "lr_schedule": "cosine",
        "precision": "bf16",
        "log_interval": 10,
        "save_interval": 0,
        "save_each_epoch": True,
        "keep_last_n": 5,
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
        "enabled": False,
        "start_step": 10,
        "num_steps": 20,
        "record_shapes": False,
        "with_stack": False,
        "export_chrome_trace": True,
    },
    "output": {
        "output_dir": str(SCRIPT_DIR / "outputs" / "cpt"),
        "log_dir": "",
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    return deep_update(DEFAULT_CONFIG, user_cfg)


def setup_dist() -> Dict[str, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if is_dist():
        dist.barrier()


def is_main(rank: int) -> bool:
    return rank == 0


def seed_everything(seed: int, rank: int) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dtype_from_name(name: str):
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def setup_logging(output_dir: Path, rank: int) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("haina_train")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    if rank == 0:
        file_handler = logging.FileHandler(log_dir / "train.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


@contextmanager
def timer_bucket(stats: Dict[str, float], name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats[name] += time.perf_counter() - t0


class JsonlCptDataset(IterableDataset):
    """Stream swift_jsonl rows and interleave text-only samples for CPT."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        tokenizer,
        image_processor,
        rank: int,
        world_size: int,
        epoch: int = 0,
    ):
        super().__init__()
        files = sorted(glob.glob(str(cfg["jsonl_glob"]), recursive=True))
        if not files:
            raise FileNotFoundError(f"No JSONL files matched: {cfg['jsonl_glob']}")
        self.files = files
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch
        self.eos = tokenizer.eos_token_id
        self.text_ratio = float(cfg.get("text_ratio", 0.0))
        self.text_lines: List[str] = []
        text_path = cfg.get("text_jsonl_path", "")
        if text_path and self.text_ratio > 0.0:
            with open(text_path, "r", encoding="utf-8") as f:
                self.text_lines = [line for line in f if line.strip()]
            if not self.text_lines:
                raise RuntimeError(f"Text JSONL is empty: {text_path}")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _iter_files(self) -> List[str]:
        files = list(self.files)
        if self.cfg.get("shuffle_files", True):
            rng = random.Random(int(self.cfg.get("seed", 42)) + self.epoch)
            rng.shuffle(files)
        return files

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        shard_id = self.rank * num_workers + worker_id
        num_shards = max(1, self.world_size * num_workers)
        rng = random.Random(int(self.cfg.get("seed", 42)) + self.epoch + shard_id)
        yielded = 0
        max_samples = int(self.cfg.get("samples_per_epoch", 0) or 0)
        row_index = 0
        for path in self._iter_files():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    take = row_index % num_shards == shard_id
                    row_index += 1
                    if not take:
                        continue
                    if max_samples and yielded >= max_samples:
                        return

                    # ── text-only mixing ──
                    if self.text_lines and rng.random() < self.text_ratio:
                        sample = self._build_text_sample(rng)
                    else:
                        sample = self._sample_from_line(line)

                    if sample is None:
                        continue
                    yielded += 1
                    yield sample

    def _build_text_sample(self, rng: random.Random) -> Optional[Dict[str, Any]]:
        """Build a pure-text CPT sample from the preloaded text JSONL."""
        line = rng.choice(self.text_lines)
        try:
            row = json.loads(line)
            text = str(row.get("text", "") or "").strip()
        except Exception:
            return None
        if len(text) < 20:
            return None
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        max_text_len = int(self.cfg["max_length"]) - 1  # reserve 1 for eos
        if len(token_ids) > max_text_len:
            token_ids = token_ids[:max_text_len]
        if len(token_ids) < 4:
            return None
        ids = token_ids + [self.eos]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()  # all tokens contribute to LM loss
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "num_tokens": int(input_ids.numel()),
            "is_text_only": True,
        }

    def _sample_from_line(self, line: str) -> Optional[Dict[str, Any]]:
        try:
            row = json.loads(line)
            messages = row.get("messages") or []
            images = row.get("images") or []
            caption = next((m.get("content", "").strip() for m in messages if m.get("role") == "assistant"), "")
            if not caption or not images:
                return None
            image_path = images[0]
            image = None
            for _ in range(max(1, int(self.cfg.get("image_open_retries", 1)))):
                try:
                    image = Image.open(image_path).convert("RGB")
                    break
                except Exception:
                    image = None
            if image is None:
                return None
            return self._build_sample(image, caption)
        except Exception:
            if self.cfg.get("skip_bad_samples", True):
                return None
            raise

    def _build_sample(self, image: Image.Image, caption: str) -> Optional[Dict[str, Any]]:
        image_inputs = self.image_processor.preprocess(
            images=image,
            return_tensors="pt",
            cpu_patchify=bool(self.cfg.get("cpu_patchify", False)),
        )
        pixel_values = image_inputs["pixel_values"]
        if pixel_values.ndim == 4 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0]
        grid = image_inputs["image_grid_thw"][0]
        grid = [int(x) for x in grid]
        num_image_tokens = int(grid[0]) * int(grid[1]) * int(grid[2])

        answer_ids = self.tokenizer.encode(caption, add_special_tokens=False)
        vision_len = 1 + num_image_tokens + 1
        max_answer_len = int(self.cfg["max_length"]) - vision_len - 1
        if max_answer_len < 4:
            return None
        if len(answer_ids) > max_answer_len:
            answer_ids = answer_ids[:max_answer_len]

        ids = [VISION_START] + [IMAGE_PAD] * num_image_tokens + [VISION_END] + answer_ids + [self.eos]
        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[:vision_len] = -100
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values.contiguous(),
            "image_grid_thw": torch.tensor(grid, dtype=torch.long),
            "num_tokens": int(input_ids.numel()),
        }


def collate_cpt(samples: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(s["input_ids"].numel() for s in samples)
    batch = len(samples)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, sample in enumerate(samples):
        n = sample["input_ids"].numel()
        input_ids[i, :n] = sample["input_ids"]
        labels[i, :n] = sample["labels"]
        attention_mask[i, :n] = sample["attention_mask"]

    # ── Handle pixel_values: may be absent for text-only samples ──
    img_samples = [s for s in samples if not s.get("is_text_only", False)]
    if not img_samples:
        pixel_values = None
        image_grid_thw = None
    else:
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
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "num_tokens": num_tokens,
        # Track which samples are image vs text-only for the training loop
        "img_sample_indices": [i for i, s in enumerate(samples) if not s.get("is_text_only", False)],
    }


def build_model(cfg: Dict[str, Any], device: torch.device, logger: logging.Logger):
    model_cfg = cfg["model"]
    dtype = dtype_from_name(model_cfg.get("torch_dtype", "bfloat16"))
    config = HainaOCRNativePixelConfig.from_pretrained(model_cfg["model_dir"], trust_remote_code=True)
    config._attn_implementation = model_cfg.get("attn_implementation", "eager")
    config.use_liger_ce = bool(model_cfg.get("use_liger_ce", False))
    model = HainaOCRNativePixelForConditionalGeneration(config)

    # ── Load Qwen3 pretrained weights (essential: model init is random) ──
    llm_path = model_cfg.get("llm_path", "")
    if llm_path and Path(llm_path).exists():
        model.load_pretrained_components(llm_path, dtype=dtype)
        logger.info("Loaded Qwen3 pretrained weights from %s", llm_path)
    else:
        logger.warning("llm_path not set or not found, Qwen3 weights stay random!")

    cnn_weights = model_cfg.get("cnn_weights")
    if cnn_weights and Path(cnn_weights).exists():
        missing, unexpected = model.load_state_dict(load_file(cnn_weights), strict=False)
        logger.info("Loaded cnn/init weights: %s | missing=%d unexpected=%d", cnn_weights, len(missing), len(unexpected))
    elif Path(model_cfg["model_dir"], "model.safetensors").exists():
        missing, unexpected = model.load_state_dict(load_file(str(Path(model_cfg["model_dir"]) / "model.safetensors")), strict=False)
        logger.info("Loaded model weights from model_dir | missing=%d unexpected=%d", len(missing), len(unexpected))

    # ── Load Stage 1 base checkpoint (cpt_v1 or similar) ──
    base_ckpt = model_cfg.get("base_checkpoint", "")
    if base_ckpt and Path(base_ckpt).exists():
        sd = load_file(base_ckpt) if base_ckpt.endswith(".safetensors") else torch.load(base_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info("Loaded base checkpoint: %s | missing=%d unexpected=%d", base_ckpt, len(missing), len(unexpected))
        # Fix: old-architecture checkpoints may lack vision_start/end_embed.
        # Restore them from the trained embed_tokens (old arch used embed_tokens lookup).
        for boundary_name, boundary_id in [("vision_start_embed", 151652), ("vision_end_embed", 151653)]:
            if any(boundary_name in k for k in missing):
                emb = model.model.qwen3.embed_tokens.weight
                target = getattr(model.model, boundary_name)
                with torch.no_grad():
                    target.copy_(emb[boundary_id])
                logger.info("Restored %s from embed_tokens[%d]", boundary_name, boundary_id)

    set_trainable(model, model_cfg)
    if bool(model_cfg.get("gradient_checkpointing", False)) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    model = model.to(device=device, dtype=dtype)
    if bool(model_cfg.get("compile", False)) and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile")
        model = torch.compile(model)
    return model


def set_trainable(model, model_cfg: Dict[str, Any]) -> None:
    modules = {
        "vision_encoder": bool(model_cfg.get("train_vision_encoder", False)),
        "pos_embed_2d": bool(model_cfg.get("train_pos_embed", True)),
        "qwen3": bool(model_cfg.get("train_qwen3", True)),
        "lm_head": bool(model_cfg.get("train_lm_head", True)),
        "vision_boundary": bool(model_cfg.get("train_vision_boundary", True)),
    }
    # Layer-wise Qwen3 control (only active when train_qwen3=False)
    qwen3_layers: Optional[List[int]] = model_cfg.get("train_qwen3_layers") or None
    train_embed_tokens = bool(model_cfg.get("train_qwen3_embed_tokens", False))
    train_final_norm = bool(model_cfg.get("train_qwen3_final_norm", True))

    for name, param in model.named_parameters():
        train = False
        if ".vision_encoder." in name or name.startswith("model.vision_encoder."):
            train = modules["vision_encoder"]
        elif ".pos_embed_2d." in name or name.startswith("model.pos_embed_2d."):
            train = modules["pos_embed_2d"]
        elif ".qwen3." in name or name.startswith("model.qwen3."):
            if modules["qwen3"]:
                train = True
            elif qwen3_layers is not None:
                # Fine-grained layer control
                if "embed_tokens" in name:
                    train = train_embed_tokens
                elif "norm" in name and "layers." not in name:
                    # This is qwen3.final_norm (Qwen3RMSNorm after all layers)
                    # Pattern: model.qwen3.norm.weight (not layers.X.xxx_norm)
                    train = train_final_norm
                else:
                    # Check if this param belongs to one of the selected layers
                    train = any(f"layers.{idx}." in name for idx in qwen3_layers)
            # else: train_qwen3=False and no layers specified → all frozen
        elif name.startswith("lm_head."):
            train = modules["lm_head"]
        elif name.endswith("vision_start_embed") or name.endswith("vision_end_embed"):
            train = modules["vision_boundary"]
        param.requires_grad = train


def build_optimizer(model, cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "norm" in name.lower() or "rms" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(train_cfg["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(train_cfg["lr"]),
        betas=tuple(float(x) for x in train_cfg["betas"]),
        eps=float(train_cfg["eps"]),
    )


def build_scheduler(optimizer, cfg: Dict[str, Any], total_steps: int):
    train_cfg = cfg["training"]
    warmup_steps = int(train_cfg.get("warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.0)))
    min_lr_ratio = float(train_cfg.get("min_lr", 0.0)) / max(float(train_cfg["lr"]), 1e-20)
    schedule = str(train_cfg.get("lr_schedule", "cosine")).lower()

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        if schedule == "linear":
            value = 1.0 - progress
        elif schedule == "constant":
            value = 1.0
        else:
            value = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return max(min_lr_ratio, value)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    output_dir: Path,
    step: int,
    epoch: int,
    model,
    optimizer,
    scheduler,
    cfg: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    core = model.module if hasattr(model, "module") else model
    ckpt_dir = output_dir / f"checkpoint-{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().contiguous().cpu() for k, v in core.state_dict().items()}
    save_file(state, str(ckpt_dir / "model.safetensors"))
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
        },
        ckpt_dir / "trainer_state.pt",
    )
    latest = output_dir / "checkpoint-latest"
    tmp_latest = output_dir / "checkpoint-latest.tmp"
    if tmp_latest.exists() or tmp_latest.is_symlink():
        tmp_latest.unlink()
    try:
        tmp_latest.symlink_to(ckpt_dir.name, target_is_directory=True)
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        tmp_latest.rename(latest)
    except OSError:
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(ckpt_dir, latest)

    keep_last_n = int(cfg["training"].get("keep_last_n", 5))
    ckpts = sorted(
        [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-") and p.name != "checkpoint-latest"],
        key=lambda p: p.name,
    )
    for old in ckpts[:-keep_last_n]:
        shutil.rmtree(old, ignore_errors=True)
    logger.info("Saved checkpoint: %s", ckpt_dir)


def load_resume(output_dir: Path, model, optimizer, scheduler, resume: str, logger: logging.Logger):
    if not resume or str(resume).lower() in {"false", "none", "0"}:
        return 0, 0
    ckpt = output_dir / "checkpoint-latest" if str(resume).lower() == "auto" else Path(resume)
    state_path = ckpt / "trainer_state.pt"
    model_path = ckpt / "model.safetensors"
    if not state_path.exists() or not model_path.exists():
        logger.info("No resume checkpoint found at %s", ckpt)
        return 0, 0
    core = model.module if hasattr(model, "module") else model
    missing, unexpected = core.load_state_dict(load_file(str(model_path)), strict=False)
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    logger.info("Resumed %s | step=%s epoch=%s missing=%d unexpected=%d", ckpt, state["step"], state["epoch"], len(missing), len(unexpected))
    return int(state["step"]), int(state["epoch"])


def make_profiler(cfg: Dict[str, Any], output_dir: Path):
    prof_cfg = cfg["profile"]
    if not prof_cfg.get("enabled", False):
        return None
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    trace_dir = output_dir / "profile"
    trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(prof_cfg.get("start_step", 10)),
            warmup=1,
            active=int(prof_cfg.get("num_steps", 20)),
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
        record_shapes=bool(prof_cfg.get("record_shapes", False)),
        with_stack=bool(prof_cfg.get("with_stack", False)),
        profile_memory=True,
    )


def count_steps_per_epoch(cfg: Dict[str, Any], world_size: int) -> int:
    samples = int(cfg["data"].get("samples_per_epoch", 0) or 0)
    if samples <= 0:
        # Streaming mode cannot know total rows cheaply. This default keeps the
        # run epoch-based but asks users to set samples_per_epoch for exactness.
        samples = 1_051_594
    effective = int(cfg["training"]["micro_batch_size"]) * int(cfg["training"]["grad_accum_steps"]) * max(1, world_size)
    return max(1, math.ceil(samples / max(1, effective)))


def train(cfg: Dict[str, Any]) -> None:
    dist_info = setup_dist()
    rank, world_size, local_rank = dist_info["rank"], dist_info["world_size"], dist_info["local_rank"]
    seed_everything(int(cfg["training"]["seed"]), rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = Path(cfg["output"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output_dir, rank)
    if is_main(rank):
        with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    logger.info("rank=%d/%d local_rank=%d device=%s", rank, world_size, local_rank, device)
    llm_path = cfg["model"].get("llm_path", "")
    if not llm_path or not Path(llm_path).exists():
        llm_path = "/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b"
    tokenizer = load_tokenizer_with_fixes(llm_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = HainaOCRNativePixelImageProcessor.from_pretrained(cfg["model"]["model_dir"])
    image_processor.cpu_patchify = bool(cfg["data"].get("cpu_patchify", False))

    model = build_model(cfg, device, logger)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("parameters total=%d trainable=%d ratio=%.2f%%", total, trainable, 100.0 * trainable / max(1, total))
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=bool(cfg["ddp"].get("find_unused_parameters", True)),
            static_graph=bool(cfg["ddp"].get("static_graph", False)),
            gradient_as_bucket_view=bool(cfg["ddp"].get("gradient_as_bucket_view", True)),
            broadcast_buffers=bool(cfg["ddp"].get("broadcast_buffers", False)),
        )

    steps_per_epoch = count_steps_per_epoch(cfg, world_size)
    total_steps = steps_per_epoch * int(cfg["training"]["epochs"])
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    start_step, start_epoch = load_resume(output_dir, model, optimizer, scheduler, cfg["training"].get("resume", "auto"), logger)

    writer = None
    if is_main(rank):
        log_dir = cfg["output"].get("log_dir") or str(output_dir / "tb")
        writer = SummaryWriter(log_dir)
        logger.info("epochs=%d steps_per_epoch=%d total_steps=%d output=%s", cfg["training"]["epochs"], steps_per_epoch, total_steps, output_dir)

    precision = str(cfg["training"].get("precision", "bf16")).lower()
    amp_dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
    use_amp = torch.cuda.is_available() and precision in {"bf16", "bfloat16", "fp16", "float16"}
    scaler = torch.amp.GradScaler("cuda", enabled=precision in {"fp16", "float16"} and torch.cuda.is_available())
    prof = make_profiler(cfg, output_dir)

    global_step = start_step
    pbar = tqdm(total=total_steps, initial=start_step, disable=not is_main(rank), dynamic_ncols=True, desc="train")
    model.train()
    stats = defaultdict(float)
    tokens_window = 0
    samples_window = 0
    window_t0 = time.perf_counter()

    if prof:
        prof.__enter__()
    try:
        for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
            data_cfg = dict(cfg["data"])
            data_cfg["seed"] = int(cfg["training"]["seed"])

            # ── HTML online render path ──
            html_manifest = data_cfg.get("html_manifest_glob", "")
            if html_manifest and _has_html_render:
                if epoch == start_epoch:
                    logger.info("Using HTML online render pipeline: manifest=%s", html_manifest)
                import glob as _glob
                manifest_files = sorted(_glob.glob(html_manifest, recursive=True))
                if not manifest_files:
                    raise FileNotFoundError(f"No manifest files matched: {html_manifest}")
                renderer = HtmlOCRRenderer(
                    RenderConfig(output_mode="uint8"),
                    BrowserConfig(
                        executable_path=data_cfg.get("browser_path") or None,
                        katex_dist=data_cfg.get("katex_dist") or None,
                    ),
                )
                dataset = RenderDataset(
                    manifest_files, renderer=renderer,
                    base_seed=int(cfg["training"]["seed"]),
                    rank=rank,
                )
                collator = RenderCollator(
                    tokenizer=tokenizer,
                    vision_start_id=VISION_START,
                    image_pad_id=IMAGE_PAD,
                    vision_end_id=VISION_END,
                    eos_id=tokenizer.eos_token_id,
                    pad_id=tokenizer.pad_token_id,
                    max_length=int(data_cfg.get("max_length", 4096)),
                )
                loader = DataLoader(
                    dataset,
                    batch_size=int(cfg["training"]["micro_batch_size"]),
                    collate_fn=collator,
                    num_workers=int(data_cfg.get("num_workers", 1)),
                    prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
                    pin_memory=bool(data_cfg.get("pin_memory", True)),
                    persistent_workers=bool(data_cfg.get("persistent_workers", True)),
                    drop_last=True,
                )

                # ── Text mixing: preload text lines once per epoch ──
                text_ratio = float(data_cfg.get("text_ratio", 0.0))
                text_lines: List[str] = []
                _text_rng = random.Random(int(cfg["training"]["seed"]) + epoch)
                text_path = data_cfg.get("text_jsonl_path", "")
                if text_ratio > 0.0 and text_path:
                    if epoch == start_epoch:
                        logger.info("Text mixing: path=%s ratio=%.2f", text_path, text_ratio)
                    import glob as _tglob
                    _tfiles = sorted(_tglob.glob(text_path, recursive=True)) or [text_path]
                    text_lines = []
                    for _tp in _tfiles:
                        with open(_tp, "r", encoding="utf-8") as _tf:
                            text_lines.extend(line for line in _tf if line.strip())
                    if not text_lines:
                        raise RuntimeError(f"Text JSONL is empty: {text_path}")
                    logger.info("Text mixing loaded %d lines", len(text_lines))
                _use_text_mixing = text_ratio > 0.0 and len(text_lines) > 0
            else:
                dataset = JsonlCptDataset(data_cfg, tokenizer, image_processor, rank, world_size, epoch=epoch)
                loader = DataLoader(
                    dataset,
                    batch_size=int(cfg["training"]["micro_batch_size"]),
                    collate_fn=lambda batch: collate_cpt(batch, tokenizer.pad_token_id),
                    num_workers=int(cfg["data"].get("num_workers", 0)),
                    prefetch_factor=int(cfg["data"].get("prefetch_factor", 2)) if int(cfg["data"].get("num_workers", 0)) > 0 else None,
                    pin_memory=bool(cfg["data"].get("pin_memory", True)),
                    persistent_workers=bool(cfg["data"].get("persistent_workers", True)) and int(cfg["data"].get("num_workers", 0)) > 0,
                )
            accum_loss = 0.0
            accum_loss_count = 0
            micro_loss_sum = 0.0
            micro_loss_count = 0
            micro = 0
            optimizer.zero_grad(set_to_none=True)

            for batch in loader:
                if global_step >= total_steps or global_step >= (epoch + 1) * steps_per_epoch:
                    break

                # ── Text mixing: replace some samples with pure text ──
                if _use_text_mixing and html_manifest:
                    bs = batch["input_ids"].shape[0]
                    keep_img_indices: list[int] = []
                    for row in range(bs):
                        if _text_rng.random() >= text_ratio:
                            keep_img_indices.append(row)
                            continue
                        # Build text-only replacement for this row
                        text_line = _text_rng.choice(text_lines)
                        try:
                            txt = json.loads(text_line)
                            text_content = str(txt.get("t") or txt.get("text") or "").strip()
                        except Exception:
                            text_content = text_line.strip()
                        if len(text_content) < 20:
                            keep_img_indices.append(row)
                            continue
                        tids = tokenizer.encode(text_content, add_special_tokens=False)
                        max_t = max(1, int(data_cfg.get("max_length", 4096)) - 1)
                        tids = tids[:max_t]
                        tids = tids + [tokenizer.eos_token_id]
                        # Overwrite in the batch tensors
                        for t in range(len(tids)):
                            batch["input_ids"][row, t] = tids[t]
                            batch["labels"][row, t] = tids[t]
                        batch["input_ids"][row, len(tids):] = tokenizer.pad_token_id
                        batch["labels"][row, len(tids):] = -100
                        batch["attention_mask"][row, :len(tids)] = 1
                        batch["attention_mask"][row, len(tids):] = 0
                    # Rebuild pixel_values with only image samples
                    if not keep_img_indices:
                        batch["pixel_values"] = None
                        batch["image_grid_thw"] = None
                    elif len(keep_img_indices) < bs:
                        batch["pixel_values"] = batch["pixel_values"][keep_img_indices]
                        batch["image_grid_thw"] = batch["image_grid_thw"][keep_img_indices]
                    batch["img_sample_indices"] = keep_img_indices

                with timer_bucket(stats, "to_device"):
                    batch = {
                        k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in batch.items()
                    }
                    if batch.get("pixel_values") is not None:
                        pv = batch["pixel_values"]
                        if pv.dtype == torch.uint8:
                            pv = pv.to(device=device, non_blocking=True).to(dtype=dtype_from_name(cfg["model"]["torch_dtype"])).div_(255.0)
                        else:
                            pv = pv.to(dtype=dtype_from_name(cfg["model"]["torch_dtype"]))
                        batch["pixel_values"] = pv

                amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with timer_bucket(stats, "forward_backward"):
                    with amp_ctx:
                        model_kwargs = dict(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            use_cache=False,
                        )
                        if batch.get("pixel_values") is not None:
                            model_kwargs["pixel_values"] = batch["pixel_values"]
                            model_kwargs["image_grid_thw"] = batch["image_grid_thw"]
                        out = model(**model_kwargs)
                        loss = out.loss / int(cfg["training"]["grad_accum_steps"])
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                micro_loss_sum += float(loss.detach().cpu()) * int(cfg["training"]["grad_accum_steps"])
                micro_loss_count += 1
                if "num_tokens" not in batch:
                    batch["num_tokens"] = batch["attention_mask"].sum(dim=1)
                tokens_window += int(batch["num_tokens"].sum().item())
                samples_window += int(batch["input_ids"].shape[0])
                micro += 1

                if micro >= int(cfg["training"]["grad_accum_steps"]):
                    step_loss = micro_loss_sum / max(1, micro_loss_count)
                    with timer_bucket(stats, "optimizer"):
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            float(cfg["training"]["max_grad_norm"]),
                        )
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    micro = 0
                    micro_loss_sum = 0.0
                    micro_loss_count = 0
                    accum_loss += step_loss
                    accum_loss_count += 1

                    if prof:
                        prof.step()
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                    if global_step % int(cfg["training"]["log_interval"]) == 0:
                        elapsed = max(1e-9, time.perf_counter() - window_t0)
                        tok_s = tokens_window / elapsed
                        samp_s = samples_window / elapsed
                        mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0.0
                        if is_main(rank):
                            avg_loss = accum_loss / max(1, accum_loss_count)
                            logger.info(
                                "epoch=%d step=%d/%d loss=%.5f lr=%.3e grad_norm=%.3f samples/s=%.2f tokens/s=%.0f mem=%.2fGB time=%s",
                                epoch + 1,
                                global_step,
                                total_steps,
                                avg_loss,
                                scheduler.get_last_lr()[0],
                                float(grad_norm),
                                samp_s,
                                tok_s,
                                mem_gb,
                                dict((k, round(v, 3)) for k, v in stats.items()),
                            )
                            if writer:
                                writer.add_scalar("train/loss", avg_loss, global_step)
                                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
                                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                                writer.add_scalar("perf/samples_per_sec", samp_s, global_step)
                                writer.add_scalar("perf/tokens_per_sec", tok_s, global_step)
                                writer.add_scalar("perf/max_memory_gb", mem_gb, global_step)
                                for key, value in stats.items():
                                    writer.add_scalar(f"profile_time/{key}", value, global_step)
                        accum_loss = 0.0
                        accum_loss_count = 0
                        tokens_window = 0
                        samples_window = 0
                        stats.clear()
                        window_t0 = time.perf_counter()

                    save_interval = int(cfg["training"].get("save_interval", 0) or 0)
                    if save_interval > 0 and global_step % save_interval == 0:
                        barrier()
                        if is_main(rank):
                            save_checkpoint(output_dir, global_step, epoch, model, optimizer, scheduler, cfg, logger)
                        barrier()

                    empty_cache_interval = int(cfg["training"].get("empty_cache_interval", 0) or 0)
                    if empty_cache_interval > 0 and torch.cuda.is_available() and global_step % empty_cache_interval == 0:
                        torch.cuda.empty_cache()

            if bool(cfg["training"].get("save_each_epoch", True)):
                barrier()
                if is_main(rank):
                    save_checkpoint(output_dir, global_step, epoch + 1, model, optimizer, scheduler, cfg, logger)
                barrier()
    finally:
        if prof:
            prof.__exit__(None, None, None)
        pbar.close()
        if writer:
            writer.close()
        if is_dist():
            dist.destroy_process_group()

    if is_main(rank) and bool(cfg["training"].get("save_final", True)):
        save_checkpoint(output_dir, global_step, int(cfg["training"]["epochs"]), model, optimizer, scheduler, cfg, logger)
        logger.info("done step=%d output=%s", global_step, output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config_cpt.yaml"), help="YAML config path")
    parser.add_argument("--set", action="append", default=[], help="Override key=value, e.g. training.epochs=2")
    return parser.parse_args()


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override: {item}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        cursor = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return cfg


if __name__ == "__main__":
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.set)
    if bool(config["training"].get("detect_anomaly", False)):
        torch.autograd.set_detect_anomaly(True)
    train(config)
