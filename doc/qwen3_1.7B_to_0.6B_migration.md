# Qwen3-1.7B → Qwen3-0.6B 迁移记录

## 架构参数变更

| 参数 | 1.7B | 0.6B |
|------|------|------|
| `hidden_size` | 2048 | **1024** |
| `intermediate_size` | 6144 | **3072** |
| `num_hidden_layers` | 28 | 28 |
| `num_attention_heads` | 16 | 16 |
| `num_key_value_heads` | 8 | 8 |
| `head_dim` | 128 | 128 |
| `vocab_size` | 151936 | 151936 |

总参数量：~1.7B → ~598M

## 0.6B 模型路径

```
/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b
```

## 改动文件清单

### 1. `hainaocr_nativepixel/hainaocr_nativepixel/configuration.py`

- 将 `QWEN3_1_7B_DEFAULTS` 重命名为 `QWEN3_0_6B_DEFAULTS`
- `hidden_size`: 2048 → 1024
- `intermediate_size`: 6144 → 3072
- 默认 `llm_model_name` 指向 `qwen3_0_6b`
- 所有 `_default_if_none` 引用更新为 `QWEN3_0_6B_DEFAULTS`

### 2. `hainaocr_nativepixel/config.json`

- `llm_model_name` → `/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b`
- `source_llm_model_name` → 同上
- `hidden_size` → 1024
- `intermediate_size` → 3072

### 3. `hainaocr_nativepixel/model_register.py`

- `DEFAULT_LLM_DIR` → `/mnt/si001719kd1w/default/xjz/model/qwen3_0_6b`

### 4. `haina_train/train_haina_cpt.py`

- 默认配置 `llm_path` → `qwen3_0_6b`

### 5. `haina_train/train_haina_sft.py`

- 默认配置 `llm_path` → `qwen3_0_6b`

### 6. `haina_train/config_cpt.yaml`

- `llm_path` → `qwen3_0_6b`

### 7. `haina_train/config_cpt_stage2.yaml`

- `llm_path` → `qwen3_0_6b`

### 8. `haina_train/config_sft.yaml`

- `llm_path` → `qwen3_0_6b`

## 验证结果

| 测试项 | 结果 |
|--------|------|
| 配置参数与 qwen3_0_6b/config.json 一致性 | PASSED |
| config.json 加载 | PASSED |
| 模型实例化 (598,808,832 参数) | PASSED |
| 0.6B 权重加载 (28 层全加载) | PASSED |
| 前向传播 | PASSED |
| 反向传播 | PASSED |
