# 对接 train_haina_cpt.py

## 改法

`train_haina_cpt.py` 改 **4 处**，其余代码不动。

### 1. 顶部 import 块 (L53-65)

```python
# OLD
_HTML_RENDER_DIR = SCRIPT_DIR.parent / "data" / "haina_html_render"
if str(_HTML_RENDER_DIR) not in sys.path:
    sys.path.insert(0, str(_HTML_RENDER_DIR))
_has_html_render = True
try:
    from html_ocr_dataset import HtmlRenderedOCRDataset
    from html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig
    from haina_cpt_collator import HainaCPTCollator
except Exception:
    _has_html_render = False
    HtmlRenderedOCRDataset = None
    HtmlOCRRenderer = None
    HainaCPTCollator = None

# NEW
_T2V_DIR = SCRIPT_DIR.parent / "data" / "text2vision_pt"
sys.path.insert(0, str(_T2V_DIR))
_has_html_render = True
try:
    from dataset.render_dataset import RenderDataset
    from dataset.render_collator import RenderCollator
    from rendering.html_ocr_renderer import BrowserConfig, HtmlOCRRenderer, RenderConfig
except Exception:
    _has_html_render = False
    RenderDataset = None
    RenderCollator = None
    HtmlOCRRenderer = None
```

### 2. 渲染器创建 (L765-776)

```python
# OLD
renderer = HtmlOCRRenderer(
    RenderConfig(
        output_mode="uint8",
        width=int(data_cfg.get("render_width", 1024)),
        height=int(data_cfg.get("render_height", 512)),
        patch_size=int(data_cfg.get("render_patch_size", 32)),
    ),
    BrowserConfig(
        executable_path=data_cfg.get("browser_path") or None,
        katex_dist=data_cfg.get("katex_dist") or None,
    ),
)

# NEW — RenderConfig 用 bucket 替代固定宽高
renderer = HtmlOCRRenderer(
    RenderConfig(output_mode="uint8"),
    BrowserConfig(
        executable_path=data_cfg.get("browser_path") or None,
        katex_dist=data_cfg.get("katex_dist") or None,
    ),
)
```

### 3. Dataset 创建 (L777-781)

```python
# OLD
dataset = HtmlRenderedOCRDataset(
    manifest_files, renderer,
    base_seed=int(cfg["training"]["seed"]),
    rank=rank,
)

# NEW
dataset = RenderDataset(
    manifest_files, renderer=renderer,
    base_seed=int(cfg["training"]["seed"]),
    rank=rank,
)
```

### 4. Collator 创建 (L782-790)

```python
# OLD
collator = HainaCPTCollator(
    tokenizer=tokenizer,
    vision_start_id=VISION_START,
    image_pad_id=IMAGE_PAD,
    ...
)

# NEW — 完全相同的参数
collator = RenderCollator(
    tokenizer=tokenizer,
    vision_start_id=VISION_START,
    image_pad_id=IMAGE_PAD,
    vision_end_id=VISION_END,
    eos_id=tokenizer.eos_token_id,
    pad_id=tokenizer.pad_token_id,
    max_length=int(data_cfg.get("max_length", 4096)),
)
```

### 5. YAML 配置（无需改字段名）

```yaml
# config_cpt_html_stage1.yaml — 原有字段保持不变
data:
  html_manifest_glob: "/path/to/pages-*.jsonl"
  browser_path: "/path/to/chromium"
  katex_dist: "/path/to/katex/dist"
  num_workers: 2
  max_length: 4096
```

## 行为变化

| | 旧 (haina_html_render) | 新 (text2vision_pt) |
|---|---|---|
| 画布尺寸 | 固定 1024×512 | 动态 bucket (256~512) |
| 溢出处理 | 删 block / 缩字号 | 抛 NeedsSplit 跳过 |
| target_text | 渲染器可能修改 | 渲染前确定，不变 |
| font 字号 | 21→18px 自动缩小 | 固定采样值 |
| 单样本延迟 | ~260ms | ~280ms |
| 旧 `render()` | 可用 | 可用 (legacy renderBlocks) |
| 新 `render_dynamic()` | 无 | 主方法 |

## 验证

改完后运行 smoke test：

```bash
cd /mnt/si001719bp3c/default/XJZ/modelv3/data/text2vision_pt
python -c "
from dataset.render_dataset import RenderDataset
from dataset.render_collator import RenderCollator
from rendering.html_ocr_renderer import HtmlOCRRenderer, RenderConfig, BrowserConfig
import torch
from torch.utils.data import DataLoader

# Mock tokenizer for test
class MockTok:
    eos_token_id=151645; pad_token_id=151643
    def encode(self, t, **kw): return [ord(c)%1000 for c in t[:50]]

r = HtmlOCRRenderer(RenderConfig(output_mode='uint8'),
    BrowserConfig(executable_path='/root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome',
                  katex_dist='/mnt/.../katex/dist'))
ds = RenderDataset(['/mnt/.../tmp/train/train-00000.jsonl'], renderer=r)
collator = RenderCollator(tokenizer=MockTok())
loader = DataLoader(ds, batch_size=2, collate_fn=collator)
batch = next(iter(loader))
print('Keys:', list(batch.keys()))
print('input_ids:', batch['input_ids'].shape)
print('labels:', batch['labels'].shape)
print('pixel_values:', batch['pixel_values'].shape)
print('image_grid_thw:', batch['image_grid_thw'])
r.close()
"
```
