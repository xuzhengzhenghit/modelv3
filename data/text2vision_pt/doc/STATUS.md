# Text2Vision-PT 项目状态

## 已完成 (已验证)

### 核心渲染 — `rendering/html_ocr_renderer.py`
- `HtmlOCRRenderer` — Chromium + KaTeX 常驻渲染器
- `RenderUnit` — 不可变渲染单元（blocks + target_text），渲染器不修改内容
- `render_dynamic()` — 动态画布单遍渲染：
  - 宽度离散 bucket (384/512/768/1024)，按内容选择
  - 高度离散 bucket (64~512)，由真实排版决定
  - `layoutBlocks()` + `finalizePaper()` — JS 端一次排版一次测量
  - 渲染器绝不删除 block / 缩小字号 / 修改 target_text
  - 溢出抛 `NeedsSplit` / `TooWide`，不静默裁切
  - block ID 一致性硬断言
- 旧 `render()` 方法保持向后兼容

### 数据预处理 — `preprocessing/`
- `repackage_shards.py` — 已有 JSONL → Text2Vision-PT 格式（补 `f` 字段 + 分片）
- `document_parser.py` — 公式/表格/HTML 解析
- `document_chunker.py` — 文档切页
- `normalizer.py` — Unicode 规范化
- `html_sanitizer.py` — HTML 安全清洗
- `gzip_reader.py` — 流式读 gzip

### 模型工具 — `model/`
- `Dynamic2DPositionEmbedding` — (x, y, logH, logW) 四维坐标 MLP，支持任意 grid
- `RowColumnPositionEmbedding` — 备用方案
- `validate_visual_token_alignment()` — 视觉 token 对齐硬断言

### 质量保证 — `qc/`
- `render_preview.py` — 渲染预览 PNG + JSON metadata
- `validate_manifest.py` — JSONL 格式校验

### 测试 — `tests/` (13/13 全过)
- `test_visual_token_count.py` (6 个) — 视觉 token 计数/前缀构建/对齐断言
- `test_formula_roundtrip.py` (5 个) — 公式规范化 roundtrip
- `test_task_labels.py` (2 个) — collator 四种任务 label 正确性

---

## 已完成 (未验证)

这些模块已写完但未跑过端到端测试，需要接入训练管线：

- `rendering/style_sampler.py` — 随机字体/字号/排版采样
- `rendering/dynamic_canvas.py` — 画布尺寸计算器（逻辑已内联到 render_dynamic）
- `rendering/html_builder.py` — 构建安全 HTML 模板
- `rendering/image_augment.py` — OCR 退化增强
- `rendering/font_registry.py` — 字体 Unicode 覆盖注册表
- `tasks/task_sampler.py` — 四任务系统 (Full OCR / Opt.Cont / Span Recon / TextReplay)
- `dataset/iterable_dataset.py` — Text2VisionDataset (IterableDataset)
- `dataset/collator.py` — 四种任务格式 collator
- `dataset/bucket_sampler.py` — 同 shape 分桶
- `dataset/prefetch.py` — 渲染/训练解耦线程
- `qc/benchmark_pipeline.py` — 吞吐测试

---

## 未完成

- 对接 `train_haina_cpt.py` 训练脚本
- Sequence Budget Controller（渲染前 tokenize 检查）
- Visual Dependency Score（correct/wrong/blank/noise 四图对比）
- Hard Span Sampling for Task B
- Formula parse cache
- Token budget batching
- `qc/audit_fonts.py` / `qc/audit_formulas.py`
- OCR 退化增强接入渲染 pipeline
- 真实扫描背景 / 纸张纹理
- Camera distortion / 多栏文档 / 高级表格
- test_no_overflow.py / test_ddp_sharding.py
