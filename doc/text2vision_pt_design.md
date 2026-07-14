# Text2Vision-PT：面向 Encoder-Free OCR 多模态大模型的在线视觉预训练数据引擎

核心目标不是生成并保存一个静态 OCR 数据集，而是：

> **用有限的文本语料，在线产生近乎无限的视觉观测，并根据训练任务动态生成不同长度的视觉 token 序列。**

---

## 项目总架构

```
                   原始文本语料
              *.json.gz / *.jsonl.gz
                        │
                        ▼
┌─────────────────────────────────────────────┐
│  Stage 1. Corpus Preprocessor               │
│                                             │
│  流式解压 → JSON解析 → Unicode规范化        │
│  → HTML清洗 → LaTeX识别和规范化             │
│  → 表格识别 → 文档级去重                    │
│  → 按语义结构切成训练单元                   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
            分片 JSONL 页面/文档单元
          不保存图片，不保存完整 HTML
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 2. Task Sampler                      │
│                                             │
│  Task A: Full OCR                           │
│  Task B: Optical Continuation               │
│  Optional: Span Reconstruction              │
│  Optional: Text Replay                      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 3. Render Spec Sampler               │
│                                             │
│  字体 / 字号 / 行距 / 页面模板 / 排版       │
│  公式样式 / 表格样式 / OCR场景 / 难度等级   │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 4. Dynamic Layout Engine             │
│                                             │
│  HTML/CSS排版 → KaTeX公式                   │
│  → Chromium测量实际内容尺寸                 │
│  → 动态决定 W×H → 对齐到32的倍数            │
│  → 得到动态 visual token 数量               │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 5. Rasterization                     │
│                                             │
│  Chromium内存截图 → OCR退化增强             │
│  → uint8 Tensor → 不落盘                    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│  Stage 6. Bucket / Prefetch Pipeline        │
│                                             │
│  按图像尺寸/visual token分桶                │
│  → 同尺寸组成batch → 预取队列               │
│  → pinned memory → GPU                      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
              HainaOCR / Qwen3
        CNN → Dynamic Visual Tokens → LLM
```

---

## 原始数据层

### 保存

```
原始 gzip + 处理后的 JSONL shards + 少量固定验证图片 + 错误样本日志
```

### 不保存

```
训练 PNG / JPG / 每个 epoch 的渲染结果 / 完整 HTML 页面副本 / 重复的 source_text + target_text
```

### 目录结构

```
dataset/
├── train/
│   ├── shard-00000.jsonl
│   ├── shard-00001.jsonl
│   ├── ...
├── val/
├── test/
└── rejects/
```

每个 shard: 256 MB ~ 512 MB

---

## JSONL 存储格式

只保存规范化内容：

```json
{"i":"doc12-p3","t":"The resistance is \\(r_s=1/g_m\\).","s":"Physics","f":1}
```

- `i`: sample id
- `t`: normalized target text (with canonical LaTeX delimiters)
- `s`: subject
- `f`: flags bitmask (1=has_math, 2=has_table, 4=has_html, 8=multilingual)

---

## 预处理原则

1. 不能切断英文单词、Unicode 字符、`\(...\)`、`\[...\]`、`\begin{align}...\end{align}`、HTML/Markdown 表格
2. 先解析成结构化 Block (TextBlock / DisplayMathBlock / TableBlock)，再切训练单元
3. 公式统一规范：行内 `\(...\)`，行间 `\[...\]`
4. 表格 target 优先 TSV 格式

---

## 三个核心训练任务

### Task A: Full OCR

```
<vision_start> <image_pad>×N <vision_end> target_text <eos>
```

视觉前缀 label=-100，target_text 计算 loss。

### Task B: Optical Continuation

原始文本 A + B + C，将 B 渲染为图片：

```
A <vision_start> <image_pad>×N <vision_end> C
```

A 和 visual tokens label=-100，C 计算 loss。训练 LLM 把视觉 token 当作正常语言上下文的一部分。

### Task C: Visual Span Reconstruction (辅助)

```
A <vision_start> <image_pad>×N <vision_end>
→ 必须输出 B
```

防止模型忽略视觉内容。

### Text Replay

10%~20% 纯文本继续训练，防止语言能力退化。

---

## 任务比例

| 阶段 | Full OCR | Span Recon. | Opt. Cont. | Text Replay |
|------|----------|-------------|------------|-------------|
| Stage 1 | 60% | 20% | 10% | 10% |
| Stage 2 | 40% | 15% | 35% | 10% |
| Stage 3 | 30% | 10% | 50% | 10% |

---

## 动态图片尺寸

核心：文字内容 + 字体大小 + 布局 → 真实渲染后的 content bbox → 自动决定图片尺寸和 visual token 数量。

### 算法

1. 随机选择样式 (font_size, font, line_height, max_width, margin)
2. 浏览器排版，测量 content_width / content_height
3. 加入 margin: `raw_width = content_width + left_margin + right_margin`
4. 向 32 对齐: `width = ceil(raw_width / 32) * 32`
5. `N_visual = (H/32) * (W/32)`

### 尺寸范围

- 最小: 256×64
- 最大: 1024×512
- 全部 32 的倍数

### 随机空白

基础 margin 16~64 px，额外随机 slack 0~2 个 patch，避免完全 tight crop。

---

## 视觉位置编码

CNN output `[B, D, Gh, Gw]` + 2D Coordinate MLP:

```
p_{x,y} = MLP[x/(W-1), y/(H-1), log H, log W]
```

插入 Qwen3 前做 `visual embedding + 2D pos`。

---

## Batch 策略

### Bucket Batch Sampler

相同或接近 grid shape 进入同一个 batch。

### Token Budget Batch

```python
total_tokens = sum(visual_tokens + text_input_tokens + target_tokens)
# 限制 <= max_token_budget
```

---

## Loss 归一化

按有效 label token 平均，分别记录各任务 loss。

---

## 图片多样性（四层）

### 1. 字体层

Serif / Sans / Monospace / CJK / Math / Bold / Italic / Light / Condensed

### 2. 排版层

论文 30% / 教材 20% / 网页 20% / 普通文档 15% / 扫描文档 10% / 其他 5%

### 3. Raster 层

字体抗锯齿 / DPI感 / 截图感 / 打印感 / 低分辨率

### 4. OCR 退化层

Gaussian blur / motion blur / JPEG / noise / contrast / brightness / rotation / perspective / shadow / paper texture / screen moiré / scan streak

---

## Curriculum

| Phase | Clean | Mild | Hard |
|-------|-------|------|------|
| Phase 1 | 80% | 18% | 2% |
| Phase 2 | 55% | 35% | 10% |
| Phase 3 | 40% | 40% | 20% |

---

## 渲染架构

Persistent Renderer Workers:

```
JSONL Reader → Task Queue → [Renderer Worker 0..N (Chromium常驻)] → Prefetch Queue → Trainer
```

- 启动一次 Chromium + KaTeX + Fonts + Page Template
- 每个样本只做 inject content → layout → screenshot
- 禁止 CDN / 网络字体 / 每样本 reload browser
- 预取 4~16 batches，uint8 CPU → bf16 GPU

---

## 监控指标

```
render_qps / render_ms_mean / render_ms_p95
batch_wait_ms / queue_depth
samples_per_second / visual_tokens_per_second / target_tokens_per_second
gpu_data_wait_ratio
```

目标: `render throughput > global_batch / step_time`

---

## 缓存策略

**缓存**: 公式解析缓存 / 字体信息 / HTML template / Validation 图片

**不缓存**: 全部训练截图

---

## 质量控制系统

每样本 metadata:

```python
{
    "sample_id", "task_type", "width", "height",
    "grid_h", "grid_w", "visual_tokens",
    "font_size", "style", "difficulty",
    "overflow", "missing_glyph", "formula_error",
}
```

必须检测的错误: 文字裁切、内容溢出、missing glyph、公式渲染失败、表格超宽、空白图片、target 为空、图片和 target 不对应、visual token 数量不匹配、sequence 超过 max_length。

失败处理: retry with larger canvas → fallback renderer → reject log。

---

## 项目目录结构

```
text2vision_pt/
├── configs/          (data.yaml, render.yaml, tasks.yaml, train.yaml)
├── preprocessing/    (gzip_reader, json_reader, normalizer, html_sanitizer,
│                      latex_parser, table_parser, document_chunker, build_manifest)
├── rendering/        (font_registry, style_sampler, html_builder, layout_measure,
│                      dynamic_canvas, browser_renderer, formula_renderer, image_augment)
├── tasks/            (full_ocr, optical_continue, span_reconstruct, text_replay, task_sampler)
├── dataset/          (shard_reader, iterable_dataset, bucket_sampler,
│                      token_budget_sampler, collator, prefetch)
├── model/            (dynamic_2d_position, visual_token_utils)
├── qc/               (validate_manifest, render_preview, audit_fonts,
│                      audit_formulas, benchmark_pipeline)
└── tests/            (test_visual_token_count, test_no_overflow, test_task_labels,
                       test_formula_roundtrip, test_ddp_sharding)
```

---

## V1 必须完成

- gzip → JSONL shards
- 纯文本 / LaTeX / 表格解析
- HTML + KaTeX 在线渲染
- Chromium 常驻
- 动态 canvas + 动态 visual token
- Full OCR + Optical Continuation + Span Reconstruction + Text Replay
- bucket batching + prefetch
- uint8 CPU → bf16 GPU
- 固定 validation + visual dependency evaluation

---

## V2 再增加

- 真实扫描背景 / camera distortion
- 多栏复杂文档 / 高级表格
- MathJax fallback
- token-budget batching
- node-level renderer pool
- patch masking
- 真实 OCR 数据混合

---

## 五条核心原则

1. **文本内容有限，但视觉观测可以近乎无限** — 1M text × 不同 epoch × 不同字体 × 不同布局 × 不同尺寸 × 不同 OCR degradation
2. **图片尺寸由实际视觉内容决定** — 随机字体布局 → 浏览器排版 → 测量 content bbox → 量化32倍数 → 动态视觉token
3. **不要只做图→文** — 联合 Image→Text / Text+Image→Text / Visual Span Reconstruction / Text Replay
4. **所有训练图片在线生成** — JSONL 占几 GB，而不是几千万 PNG 占几百 GB
5. **整个数据系统必须跑在 GPU 前面** — GPU data wait ≈ 0
