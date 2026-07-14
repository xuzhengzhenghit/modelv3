# 训练配置文件说明

## 配置文件总览

| 文件 | 用途 | 数据源 | 状态 |
|------|------|--------|------|
| `config_cpt.yaml` | CPT Stage 1 (原始) | 预渲染 swift JSONL 图文数据 | 旧，1.7B |
| `config_cpt_stage2.yaml` | CPT Stage 2 (原始) | swift JSONL + 纯文本混合 | 旧，1.7B |
| `config_cpt_html_stage1.yaml` | **CPT Stage 1 (HTML渲染)** | HTML 在线渲染 | 当前使用 |
| `config_cpt_html_stage2.yaml` | **CPT Stage 2 (HTML渲染+文本)** | HTML 渲染 + 纯文本混合 | 新建 |
| `config_sft.yaml` | SFT 微调 | VQA 图文 + PEFT 纯文本 | 旧，1.7B |

---

## config_cpt_html_stage1.yaml — Stage 1 Vision Pretraining

**目的**: 训练 CNN vision encoder，将像素转化为 LLM 可理解的视觉 token。

```
onesci_cc_pages (970万页)
  → Chromium + KaTeX 在线渲染
  → 1024×512 PNG → 512 visual tokens
  → <vision_start> + 512×<image_pad> + <vision_end> + target_text
  → 模型 forward/backward
```

**关键参数**:

| 分类 | 参数 | 值 | 说明 |
|------|------|-----|------|
| 模型 | `llm_path` | `""` | Qwen3 随机初始化 |
| 模型 | `cnn_weights` | `cnn_weights.safetensors` | CNN 从预训练权重加载 |
| 训练 | `train_qwen3` | `true` | 训练 Qwen3 |
| 训练 | `train_lm_head` | `true` | 训练 lm_head |
| 训练 | `train_vision_encoder` | `true` | 训练 CNN |
| 数据 | `html_manifest_glob` | `pages-*.jsonl` | 全 20 个分片 |
| 数据 | `samples_per_epoch` | `9733672` | 全量 |
| 数据 | `text_ratio` | `0.2` | 20% 纯文本 |
| 训练 | `lr` | `1e-3` | |
| 训练 | `epochs` | `1` | |
| 训练 | `micro_batch_size` × `grad_accum_steps` | `2 × 16 = 32` | 有效 batch |
| 训练 | `steps_per_epoch` | `304,178` | |
| 输出 | `output_dir` | `cpt1_html_stage1` | |

---

## config_cpt_html_stage2.yaml — Stage 2 Text Mixing

**目的**: 在 Stage 1 基础上加入纯文本混合，防止 Qwen3 遗忘语言能力。

```
Stage 1 checkpoint 权重
  + HTML 渲染 (80%)
  + 纯文本 (20%)
  → 继续训练
```

**与 Stage 1 的关键差异**:

| 参数 | Stage 1 | Stage 2 | 说明 |
|------|---------|---------|------|
| `llm_path` | `""` | `""` | 都不从原始 Qwen3 加载 |
| `cnn_weights` | `safetensors` | `""` | Stage 2 不从文件加载 |
| `base_checkpoint` | 无 | `model.safetensors` | **从 Stage 1 加载全部权重** |
| `resume` | `auto` | `false` | Stage 2 重新初始化 optimizer |
| `lr` | `1e-3` | `5e-5` | Stage 2 更低的 LR |
| `output_dir` | `cpt1_html_stage1` | `cpt2_html_stage2` | |

### base_checkpoint 加载逻辑 (`train_haina_cpt.py`)

```text
if base_checkpoint 存在:
  model.load_state_dict(load_file(base_checkpoint))
  # 加载 CNN + Qwen3 + lm_head 全部权重
  # 不恢复 optimizer/scheduler
```

---

## config_cpt.yaml / config_cpt_stage2.yaml — 原始 CPT（旧）

基于预渲染 swift JSONL 数据 (`train_shuffled.jsonl`)，使用 `JsonlCptDataset`。适用于已有预渲染图片的场景。

| 参数 | Stage 1 | Stage 2 |
|------|---------|---------|
| `llm_path` | `Qwen3-1.7B` | `Qwen3-1.7B` |
| `train_qwen3` | `false` (冻结) | `true` (部分层) |
| `train_lm_head` | `false` | `true` |
| LR | `1e-3` | `5e-5` |
| `jsonl_glob` | swift JSONL | swift JSONL |
| `text_jsonl_path` | 无 | `onesci_cc_150k.jsonl` |
| `text_ratio` | `0` | `0.2` |

---

## config_sft.yaml — SFT 微调（旧）

ChatML 格式多轮对话微调，图文 VQA + 纯文本 PEFT 混合。

| 参数 | 值 | 说明 |
|------|-----|------|
| 格式 | `<\|im_start\|>user\n...<\|im_end\|>\n<\|im_start\|>assistant\n...` | ChatML |
| labels | 仅 assistant 部分 | |
| 数据 | VQA JSONL + PEFT JSONL | |
| `text_ratio` | `0.3` | |
| `epochs` | `2` | |
| LR | `5e-5` | |

---

## 通用参数速查

### 训练参数计算

```text
effective_batch = micro_batch_size × grad_accum_steps × world_size
steps_per_epoch = ceil(samples_per_epoch / effective_batch)
total_steps     = steps_per_epoch × epochs
```

当前 (micro_batch=2, grad_accum=16, samples_per_epoch=9,733,672):

| GPU 数 | 全局 batch | steps/epoch | 每卡 samples/step |
|--------|-----------|-------------|-------------------|
| 1 | 32 | 304,178 | 32 |
| 8 | 256 | 38,023 | 32 |
| 16 | 512 | 19,012 | 32 |

### 视觉 token 规格

```text
canvas:       1024 × 512
patch_size:   32
grid:         1024/32 × 512/32 = 32 × 16
tokens/page:  512
```

### 序列结构

**图文样本**:

```text
<vision_start>(1)  <image_pad>(512)  <vision_end>(1)  target_text  <eos>(1)
|<---- labels=-100 (视觉前缀不参与loss) ---->|<- labels=target_ids ->|
```

**纯文本样本**:

```text
target_text  <eos>(1)
|<-- labels=target_ids (全部参与loss) -->|
```

### 启动命令

**关键**: `source MX_env.sh` 必须在同一行，变量用 `&& \` 串联，否则不生效。

```bash
# 单卡
source /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/MX_env.sh && \
NPROC_PER_NODE=1 CONFIG=/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/config_cpt_html_stage1.yaml \
bash /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/run_cpt_html.sh

# 8卡
source /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/MX_env.sh && \
NPROC_PER_NODE=8 CONFIG=/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/config_cpt_html_stage1.yaml \
bash /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/run_cpt_html.sh

# 16卡
source /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/MX_env.sh && \
NPROC_PER_NODE=16 CONFIG=/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/config_cpt_html_stage1.yaml \
bash /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/run_cpt_html.sh

# 自定义配置
source /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/MX_env.sh && \
NPROC_PER_NODE=4 CONFIG=/mnt/si001719bp3c/default/XJZ/modelv3/haina_train/config_cpt_html_stage2.yaml \
bash /mnt/si001719bp3c/default/XJZ/modelv3/haina_train/run_cpt_html.sh
```

**错误示范**（分三行写会丢失变量，变成单卡）：

```bash
# ❌ 这样不行！NPROC_PER_NODE 没传给 bash
source MX_env.sh
NPROC_PER_NODE=8 CONFIG=xxx.yaml
bash run_cpt_html.sh
```
