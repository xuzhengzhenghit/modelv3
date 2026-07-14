# text2vision_pt 迁移记录

**日期**: 2026-07-14

## 改动文件

### 1. `data/text2vision_pt/dataset/render_dataset.py`

新增 `_parse_text_to_blocks()` 函数，调用 `preprocessing/document_parser.parse_document()` 解析紧凑 JSONL 的 `t` 字段为结构化 block（paragraph/math_display/table），确保 KaTeX 正确渲染公式。

### 2. `haina_train/train_haina_cpt.py` (4 处改动)

| 位置 | 旧 (haina_html_render) | 新 (text2vision_pt) |
|------|----------------------|---------------------|
| imports | `HtmlRenderedOCRDataset`, `HainaCPTCollator` | `RenderDataset`, `RenderCollator` |
| renderer | `RenderConfig(width=1024, height=512, ...)` | `RenderConfig(output_mode="uint8")` (动态 canvas) |
| dataset | `HtmlRenderedOCRDataset(...)` | `RenderDataset(..., renderer=renderer)` |
| collator | `HainaCPTCollator(...)` | `RenderCollator(...)` 相同参数 |

额外修复：去掉 `shuffle=True`（IterableDataset 不支持）。

## 行为变化

| | haina_html_render | text2vision_pt |
|---|---|---|
| 画布 | 固定 1024×512 | 动态 bucket (384/512/768/1024 × 64~512) |
| visual token | 固定 512 | 动态，随内容变化 |
| 溢出 | 缩字号/删 block | 抛 NeedsSplit 跳过 |
| target_text | 渲染器可能修改 | 渲染前确定，绝不修改 |
| shuffle | DataLoader shuffle | RenderDataset 内部文件+行级 shuffle |
| 单样本延迟 | ~260ms | ~280ms |
| GPU 显存 | 11 GB (固定 512 tokens) | 7.7 GB (动态 bucket) |

## 验证结果

Smoke test: 50 samples, 3 epochs, 150 steps.

| 指标 | 值 |
|------|-----|
| loss | 4.05 → 0.50 |
| throughput | ~6.2 samples/s, ~3700 tokens/s |
| GPU | 7.69 GB |
| 状态 | PASSED |

## 运行命令（不变）

```bash
source MX_env.sh && NPROC_PER_NODE=1 CONFIG=config_cpt_html_stage1.yaml bash run_cpt_html.sh
```

YAML 配置字段完全兼容，无需修改。
