# HainaOCR HTML 在线渲染 PT

这套代码完成：

```text
JSON/JSONL 文本
  → 解析普通文本、LaTeX、HTML、Markdown 表格
  → 按页面 block 切分 manifest
  → DataLoader worker 内常驻 Chromium + KaTeX
  → 只替换 DOM
  → 截图到内存
  → uint8 Tensor
  → HainaOCR CPT Collator
```

## 目录

- `benchmark_html_render.py`：最小可行性和单 Page 速度测试。
- `prepare_html_manifest.py`：原始 JSON/JSONL 转 page-level manifest。
- `document_parser.py`：公式、HTML、Markdown 表格解析及 target 规范化。
- `html_ocr_renderer.py`：常驻 Chromium 在线渲染器。
- `html_ocr_dataset.py`：支持 DDP、persistent workers 和抽样预览的 Dataset。
- `haina_cpt_collator.py`：HainaOCR CPT 输入/标签组装。
- `benchmark_dataloader.py`：多 worker 端到端吞吐测试。
- `example_train_integration.py`：接入 `train_haina_cpt.py` 的参考代码。

## 1. 安装

```bash
pip install -r requirements.txt
playwright install chromium
npm install katex
```

服务器已经安装系统 Chromium 时，可以不下载 Playwright Chromium：

```bash
export PLAYWRIGHT_CHROMIUM_EXECUTABLE=/usr/bin/chromium
```

KaTeX 必须使用本地文件，训练时不要访问 CDN。默认搜索：

```text
./node_modules/katex/dist
脚本所在目录/node_modules/katex/dist
```

也可以明确传入：

```bash
--katex-dist /path/to/node_modules/katex/dist
```

## 2. 先跑最小基准

```bash
python benchmark_html_render.py \
  --count 300 \
  --warmup 20 \
  --browser-path /usr/bin/chromium \
  --katex-dist ./node_modules/katex/dist \
  --tensor-mode uint8 \
  --save-first 8
```

脚本打印：

- DOM 构建和 KaTeX 排版耗时；
- Chromium 截图耗时；
- PNG 解码耗时；
- Tensor 转换耗时；
- p50/p95；
- 页/秒；
- 溢出与 KaTeX 错误数量；
- `[3, 512, 1024]` 输出形状。

预览图和同名 JSON 保存到 `benchmark_output/`。

### 为什么默认输出 uint8

1024×512 RGB：

```text
uint8   ≈ 1.5 MiB/页
float32 ≈ 6.0 MiB/页
```

CPU worker 中直接转换 float32 会增加内存带宽和复制成本。推荐：

```python
pixel_values = batch["pixel_values"].to(device, non_blocking=True)
pixel_values = pixel_values.to(torch.bfloat16).div_(255.0)
```

即先把 `uint8` 异步传到 GPU，再转换为训练 dtype。

## 3. 原始数据生成 manifest

```bash
python prepare_html_manifest.py \
  '/data/science/**/*.json*' \
  --output /data/ocr_pt/train_pages.jsonl \
  --text-fields text,content,body,abstract,document.text \
  --subject-fields subject,category,discipline,field \
  --page-budget 950 \
  --max-block-chars 420
```

原始 81K 长文会被解析为 block 并切成多个 page record。不会从 `\begin{align}` 中间截断。

manifest 示例：

```json
{
  "id": "physics-001-p00003",
  "doc_id": "physics-001",
  "page_index": 3,
  "subject": "Physics",
  "blocks": [
    {
      "kind": "paragraph",
      "parts": [
        {"kind": "text", "text": "The resistance is "},
        {"kind": "math_inline", "tex": "r_s=1/g_m", "target": "\\(r_s=1/g_m\\)"}
      ]
    }
  ],
  "target_text": "The resistance is \\(r_s=1/g_m\\)",
  "has_math": true,
  "has_table": false
}
```

### 公式规范化

| 原始格式 | 视觉渲染 | target |
|---|---|---|
| `$r_s=1/g_m$` | 行内公式 | `\(r_s=1/g_m\)` |
| `\(...\)` | 行内公式 | `\(...\)` |
| `$$...$$` | 行间公式 | `\[...\]` |
| `\[...\]` | 行间公式 | `\[...\]` |
| `equation/align` | 行间公式 | `\[...\]` |
| `[latex]...[/latex]` | 按长度判定 | 统一定界符 |

单 `$...$` 会通过数学信号和学科字段判断，尽量排除 `$10`、`$5 million` 等货币表达。

HTML 不直接注入浏览器：代码先用 BeautifulSoup 提取安全的段落和表格 block，避免源数据中的脚本或样式进入渲染页。

## 4. 多 worker 吞吐测试

```bash
python benchmark_dataloader.py \
  --manifest /data/ocr_pt/train_pages.jsonl \
  --workers 2 \
  --batch-size 4 \
  --batches 100 \
  --warmup-batches 10 \
  --browser-path /usr/bin/chromium \
  --katex-dist ./node_modules/katex/dist \
  --output-mode uint8 \
  --preview-dir /data/ocr_pt/preview
```

依次测试：

```bash
--workers 1
--workers 2
--workers 4
--workers 6
```

不要假设 worker 越多越快。观察：

- 总页/秒；
- batch 等待时间；
- CPU 使用率；
- 系统内存；
- GPU 是否出现 DataLoader 等待。

## 5. 与 HainaOCR PT 对接

先检查三个特殊 token ID：

```text
vision_start_id
image_pad_id
vision_end_id
```

尤其确认 `image_pad_id` 当前到底是 151654 还是 151655，并确保它与 forward 中视觉 embedding 替换条件一致。

运行接线测试：

```bash
python example_train_integration.py \
  --manifest /data/ocr_pt/train_pages.jsonl \
  --tokenizer /path/to/Qwen3-1.7B \
  --vision-start-id 151652 \
  --image-pad-id YOUR_ACTUAL_ID \
  --vision-end-id 151653 \
  --workers 2 \
  --batch-size 4 \
  --browser-path /usr/bin/chromium \
  --katex-dist ./node_modules/katex/dist
```

1024×512、patch=32 时：

```text
image_grid_thw = [1, 16, 32]
image_pad 数量 = 512
```

CPT 序列：

```text
<vision_start> <image_pad>×512 <vision_end> target <eos>
```

视觉前缀 label 全部为 `-100`。

## 6. 训练配置建议

开始时：

```text
workers=2
prefetch_factor=2
persistent_workers=True
output_mode=uint8
batch_size=按显存确定
```

如果 2 workers 的渲染吞吐低于训练消费速度，再测试 4 workers。单 worker 内浏览器和 Page 都会常驻；不会每条样本重启 Chromium。

建议抽样预览：

```python
PreviewConfig(
    directory="/data/ocr_pt/render_preview",
    probability=0.0001,
    max_per_worker=20,
)
```

每张图片会带同名 JSON，其中保存实际 target、字体大小、是否溢出、KaTeX 错误和各阶段耗时。

## 7. 目前实现的边界

- 对复杂自定义 LaTeX 宏、TikZ、完整 TeX 宏包不保证支持；KaTeX 失败时会显示错误形式并在 metadata 中计数。
- HTML 表格支持常规行列；复杂 rowspan/colspan 会被安全解析成普通二维表。
- 如果一个最小 block 本身仍高于页面，metadata 的 `overflow=true` 会暴露该问题；应进一步降低 `max_block_chars` 或单独切公式/表格。
- 页面 target 是**实际保留下来的 block**对应文本；如果随机字号导致页面溢出，渲染器会缩小字号，仍装不下时移除尾部 block，避免图片与 label 不一致。
