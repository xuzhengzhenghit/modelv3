# Text2Vision-PT 使用指南

## 安装

```bash
pip install playwright pillow numpy torch beautifulsoup4
playwright install chromium
```

KaTeX 使用 haina_html_render 已有的 node_modules，不需要额外安装。

## 三步使用

### 1. 数据预处理

```bash
cd /mnt/si001719bp3c/default/XJZ/modelv3/data/text2vision_pt

python preprocessing/repackage_shards.py \
  --input-glob '/path/to/onesci_cc_pages/pages-*.jsonl' \
  --output-dir /path/to/output \
  --shard-size-mb 256 \
  --max-pages 100000
```

输入可以是已有 compact JSONL (`{i, t, s}`)，也可以是原始 gzip 文件。

### 2. 渲染预览

```bash
python qc/render_preview.py \
  --manifest /path/to/output/train/train-00000.jsonl \
  --out-dir /path/to/preview \
  --num 12
```

输出：PNG + 同名 JSON（含 target_text、paper_size、visual_tokens）

### 3. Python API

```python
import sys
sys.path.insert(0, 'path/to/text2vision_pt')

from rendering.html_ocr_renderer import (
    HtmlOCRRenderer, RenderConfig, BrowserConfig,
    RenderUnit, NeedsSplit, TooWide
)

renderer = HtmlOCRRenderer(
    RenderConfig(output_mode="uint8"),
    BrowserConfig(
        executable_path="/path/to/chromium",
        katex_dist="/path/to/katex/dist",
    ),
)

# 定义渲染单元
blocks = [{"id": "b0", "kind": "paragraph", "parts": [{"kind": "text", "text": "你的文本"}]}]
unit = RenderUnit(
    sample_id="sample-001",
    blocks=tuple(blocks),
    target_text="你的文本",   # 渲染器绝不修改此值
)

# 渲染
result = renderer.render_dynamic(unit, seed=42)

# 结果
result["pixel_values"]       # torch.Tensor [3, H, W] uint8
result["target_text"]        # str — 和输入完全一致
result["num_visual_tokens"]  # int — 动态 visual token 数量
result["paper_size"]         # (width, height)
result["png_or_jpeg_bytes"]  # bytes

renderer.close()
```

## 异常处理

```python
try:
    result = renderer.render_dynamic(unit, seed)
except NeedsSplit as exc:
    # 内容超过 max_height(512)，需要上游 paginator 分割
    print(f"Split needed: {exc}")
except TooWide as exc:
    # 内容超过 max_width(1024)，所有宽度 bucket 都装不下
    print(f"Too wide: {exc}")
```

## 关键配置

```python
RenderConfig(
    output_mode="uint8",           # uint8 / float01 / float11 / pil
    width_buckets=(384,512,768,1024),  # 离散宽度
    height_buckets=(64,96,128,160,192,256,320,384,448,512),  # 离散高度
    patch_size=32,
    font_size_min=21, font_size_max=29,
    padding_x_range=(32,58), padding_y_range=(26,46),
    line_height_range=(1.30,1.55),
    backgrounds=("#ffffff", "#fcfcfa", "#faf9f4", "#f7f7f7"),
    font_families=(
        'Arial, "Noto Sans", sans-serif',
        '"Times New Roman", "Noto Serif", serif',
        'Georgia, "Noto Serif", serif',
    ),
)
```

## 设计原则

1. **RenderUnit 不可变** — 渲染器只负责完整渲染，绝不修改 target_text / 删除 block / 缩小字号
2. **溢出抛异常** — `NeedsSplit` / `TooWide`，不静默裁切
3. **单遍布局** — `layoutBlocks` 一次排版，`finalizePaper` 设置高度，无重复渲染
4. **离散 bucket** — 宽高从预定义集合中选择，方便 batch
5. **block ID 断言** — 渲染后的 block 集合必须与输入一致
