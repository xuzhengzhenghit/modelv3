#!/usr/bin/env python3
"""Simple inference: image → text. Usage: python infer.py <checkpoint_path> <image_path>"""
import sys, os, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safetensors.torch import load_file
from PIL import Image
from hainaocr_nativepixel import (
    HainaOCRNativePixelConfig, HainaOCRNativePixelForConditionalGeneration,
    HainaOCRNativePixelImageProcessor,
)
from hainaocr_nativepixel.hainaocr_nativepixel.processing import load_tokenizer_with_fixes

# CKPT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/outputs/cpt1_html_stage1/checkpoint-00001000/model.safetensors'
CKPT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/outputs/cpt1_html_stage1/checkpoint-latest/model.safetensors'
IMAGE = sys.argv[2] if len(sys.argv) > 2 else '/mnt/si001719bp3c/default/XJZ/modelv3/tmp/00_CC-MAIN-20251204191828-20251204221828-00044.warc_processed.jsonl.gz_12-p00000.png'
LLM = '/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b'
PROJECT_ROOT = '/mnt/si001719bp3c/default/XJZ/modelv3/hainaocr_nativepixel'
DEV = 'cuda'
VS, VE, IP = 151652, 151653, 151655

# Load
config = HainaOCRNativePixelConfig.from_pretrained(PROJECT_ROOT, trust_remote_code=True)
config._attn_implementation = 'flash_attention_2'; config.use_liger_ce = False
model = HainaOCRNativePixelForConditionalGeneration(config)
model.load_state_dict(load_file(CKPT), strict=False)
model.to(DEV, dtype=torch.bfloat16).eval()
tok = load_tokenizer_with_fixes(LLM, trust_remote_code=True); tok.pad_token = tok.eos_token
eos = tok.eos_token_id
img_proc = HainaOCRNativePixelImageProcessor.from_pretrained(PROJECT_ROOT)

# Process image
img = Image.open(IMAGE).convert('RGB')
ipt = img_proc.preprocess(images=img)
pv = ipt['pixel_values'].to(DEV, dtype=torch.bfloat16)
n = int(ipt['image_grid_thw'][0][1]) * int(ipt['image_grid_thw'][0][2])
prompt_ids = torch.tensor([VS] + [IP]*n + [VE], dtype=torch.long).unsqueeze(0).to(DEV)

# Generate (manual loop to avoid DynamicCache bug)
with torch.no_grad():
    out = model(input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
                pixel_values=pv, use_cache=True)
pkv, logits = out.past_key_values, out.logits[:, -1, :]
gen_ids = []
for _ in range(2000):
    tid = logits.argmax(dim=-1).item()
    if tid == eos: break
    gen_ids.append(tid)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[tid]], device=DEV), past_key_values=pkv, use_cache=True)
    pkv, logits = out.past_key_values, out.logits[:, -1, :]
    if len(gen_ids) >= 8 and len(set(gen_ids[-5:])) == 1: break

print(tok.decode(gen_ids, skip_special_tokens=True))
