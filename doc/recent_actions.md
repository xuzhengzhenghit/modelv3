# 近期行动记录

## 1. Qwen3-1.7B → Qwen3-0.6B 模型切换

**改动文件**（详见 `qwen3_1.7B_to_0.6B_migration.md`）:

| 文件 | 改动 |
|------|------|
| `hainaocr_nativepixel/hainaocr_nativepixel/configuration.py` | 新增 `QWEN3_0_6B_DEFAULTS`，hidden_size→1024, intermediate_size→3072 |
| `hainaocr_nativepixel/config.json` | 路径 + 架构参数 |
| `hainaocr_nativepixel/model_register.py` | `DEFAULT_LLM_DIR` |
| `haina_train/train_haina_cpt.py` | 默认 `llm_path` |
| `haina_train/train_haina_sft.py` | 默认 `llm_path` |
| `haina_train/config_cpt.yaml` | `llm_path` |
| `haina_train/config_cpt_stage2.yaml` | `llm_path` |
| `haina_train/config_sft.yaml` | `llm_path` |

架构参数对比：

| 参数 | 1.7B | 0.6B |
|------|------|------|
| hidden_size | 2048 | 1024 |
| intermediate_size | 6144 | 3072 |
| 其他 | 相同 | 相同 |
| 总参数 | ~1.7B | ~598M |

验证结果：配置加载、模型创建、权重加载、forward/backward 全部通过。

---

## 2. HTML Render 吞吐测试

**测试文件**: `data/haina_html_render/benchmark_html_render.py`, `benchmark_dataloader.py`

**单 Page 基准**: 3.98 pages/s, 瓶颈在 Chromium 截图 (~78%)

**多 Worker 扩展**（详见 `html_render_throughput_benchmark.md`）:

| workers | throughput |
|---------|-----------|
| 1 | 7.93 pages/s |
| 2 | 15.78 pages/s |
| 4 | 31.96 pages/s |
| 6 | 49.07 pages/s |

---

## 3. onesci_cc_core 数据预处理

**输入**: `/mnt/si001719bp3c/default/wbc/ocrsft/dataset/text_data/onesci_cc_core/` (21 个 .jsonl.gz, 3.4 GB, 1,007,980 篇文档)

**输出**: `/mnt/si001719kd1w/default/xjz/data/onesci_cc_pages/` (20 个 .jsonl 分片, 8.34 GB, 9,733,672 页)

**脚本**: `/tmp/build_sharded_manifest.py`

**格式变化**:
```
源: {"text":"...", "id":"...", "final_subjects":[...], ...}
输出: {"i":"doc-id-p00000", "t":"页面规范文本", "s":"Physics"}
```

**处理流程**: parse_document (识别公式/表格/HTML) → pack_pages (按 950 预算切页) → 紧凑 JSONL 输出

---

## 6. 权重初始化说明

当前配置 (`config_cpt_html_stage1.yaml`)：

| 组件 | 权重来源 | 训练 |
|------|---------|------|
| Qwen3-0.6B | `llm_path: qwen3_0_6b` (pretrained) | `train_qwen3: true` |
| CNN vision encoder | `cnn_weights: cnn_weights.safetensors` (17.5 MB) | `train_vision_encoder: true` |
| lm_head | 同 Qwen3 pretrained | `train_lm_head: true` |

如需全部随机初始化：

```yaml
llm_path: ""
cnn_weights: ""
```

### llm_path 为空时 tokenizer 回退修复

**问题**: `llm_path: ""` 时 `AutoTokenizer.from_pretrained("")` 报 `HFValidationError`

**修复** (`train_haina_cpt.py` 第 700-703 行): 当 `llm_path` 为空或路径不存在时，tokenizer 自动回退到 `/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b`。tokenizer 与模型权重解耦——随机初始化模型仍需正确 tokenizer。

### samples_per_epoch / epochs / steps 计算关系

```text
effective_batch = micro_batch_size × grad_accum_steps × world_size
steps_per_epoch = ceil(samples_per_epoch / effective_batch)
total_steps     = steps_per_epoch × epochs
```

当前配置：

```text
effective_batch  = 2 × 16 × 1 = 32
samples_per_epoch = 9,733,672
steps_per_epoch   = ceil(9,733,672 / 32) = 304,178
total_steps       = 304,178 × 1 = 304,178
```

即：每 epoch 消耗全部 970 万页，共 30.4 万步。

---

## 7. 纯文本混合训练

**目的**: 图文训练中混入纯文本样本，防止 Qwen3 遗忘语言能力（catastrophic forgetting）。

**实现**:

| 文件 | 改动 |
|------|------|
| `haina_cpt_collator.py` | 支持 `is_text_only: True` 样本，跳过视觉前缀，全部 token 参与 loss |
| `train_haina_cpt.py` | HTML render 路径新增文本混合：从 `text_jsonl_path` 加载文本，按 `text_ratio` 随机替换 batch 样本 |

**配置** (`config_cpt_html_stage1.yaml`):

```yaml
text_jsonl_path: "/mnt/si001719kd1w/default/xjz/data/onesci_cc_pages/pages-00001.jsonl"
text_ratio: 0.2           # 20% 纯文本, 80% 图文渲染
```

### 当前训练配置总览

| 参数 | 值 |
|------|-----|
| 数据 | onesci_cc_pages, 9,733,672 页 |
| 每页视觉 token | 512 (1024×512, patch=32) |
| 有效 batch | 2 × 16 = 32 |
| 每 epoch 步数 | 304,178 |
| epochs | 1 |
| 图文:纯文本比例 | 80:20 |
| Qwen3 | 随机初始化 (llm_path: "") |
| CNN | cnn_weights.safetensors |
| 可训参数 | 598M (100%) |
| LR | 1e-3, cosine |

---

## 4. CPT 训练框架适配 HTML Render

**修改**: `haina_train/train_haina_cpt.py`

- 新增 HTML 在线渲染数据管线分支：配置 `data.html_manifest_glob` 时自动使用 `HtmlRenderedOCRDataset` + `HainaCPTCollator`
- 自动处理 uint8→bfloat16 pixel_values 转换
- 自动补充 `num_tokens` 字段
- 新增 `processing.py` 符号链接 (`hainaocr_nativepixel/processing.py` → 内层 processing.py)
- 安装 hainaocr-nativepixel 为 editable package

**修改**: `data/haina_html_render/html_ocr_dataset.py`

- 支持紧凑 manifest 格式（`i/t/s` 字段），自动调用 `parse_document` 解析

**新增配置**:

| 文件 | 用途 |
|------|------|
| `haina_train/config_cpt_html_stage1.yaml` | Stage 1: train vision encoder + Qwen3, HTML 渲染数据 |
| `haina_train/run_cpt_html.sh` | 训练启动脚本 |

**Smoke test 验证**: 100 pages, 1000 steps, ~2.5min, 管线全通（见最后一条）

---

## 5. 运行命令

### 单卡 CPT 训练（HTML 在线渲染）
```bash
source /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/MX_env.sh && \
NPROC_PER_NODE=1 CONFIG=/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/config_cpt_html_stage1.yaml \
bash /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/run_cpt_html.sh
```

### 多卡 CPT 训练
```bash
source /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/MX_env.sh && \
NPROC_PER_NODE=8 CONFIG=/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/config_cpt_html_stage1.yaml \
bash /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/run_cpt_html.sh
```

### HTML 多 Worker 吞吐测试
```bash
cd /mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render
python benchmark_dataloader.py \
  --manifest /mnt/si001719kd1w/default/xjz/data/onesci_cc_pages/pages-00001.jsonl \
  --workers 4 --batch-size 4 --batches 30 --warmup-batches 5 \
  --browser-path /root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
  --katex-dist ./node_modules/katex/dist --output-mode uint8
```

### 单 Page 基准测试
```bash
cd /mnt/si001719bp3c/default/XJZ/modelv3/data/haina_html_render
python benchmark_html_render.py \
  --count 100 --warmup 10 \
  --browser-path /root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
  --katex-dist ./node_modules/katex/dist --tensor-mode uint8 --save-first 8
```
