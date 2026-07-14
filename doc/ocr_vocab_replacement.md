# OCR 词表替换

**状态**: 已完成 — 词表、tokenizer、模型配置全部实施，smoke test 通过

**日期**: 2026-07-14

## 动机

Qwen3-0.6B 原始词表 151936 tokens（BPE subword），对纯 OCR 任务过大：
- 每个输出位置需计算 151936 类 softmax
- embedding + lm_head 占 **311M 参数**

## 方案

用 **PP-OCRv5 字符级词表** + 全量语料扫描 替代 Qwen3 BPE 词表。

## 词表构建

### 数据源

| 来源 | 字符数 |
|------|--------|
| PP-OCRv5 官方字典 | 18,382 |
| onesci_cc_core (970万页) | 11,899 唯一字符 |
| onesci_cc_base (104万文档) | 27,092 唯一字符 |
| 空白字符 (space, tab, newline) | 3 |
| 合并去重 | — |

### 特殊 Token

| ID | Token | 用途 |
|----|-------|------|
| 0 | `<\|pad\|>` / `<\|image_pad\|>` | 填充 / 图像占位 |
| 1 | `<\|eos\|>` | 序列结束 |
| 2 | `<\|unk\|>` | 未知字符 |
| 3 | `<\|im_start\|>` | 对话开始 |
| 4 | `<\|im_end\|>` | 对话结束 |
| 5 | `<\|vision_start\|>` | 视觉开始 |
| 6 | `<\|vision_end\|>` | 视觉结束 |
| 7 | `<\|image_pad\|>` | 图像 token (同 pad) |
| 8 | `<\|ocr\|>` | OCR 标记 |
| 9 | `\n` | 换行 |

## 最终词表

| 指标 | 值 |
|------|-----|
| **总 tokens** | **32,647** |
| 特殊 token | 10 |
| 字符 token | 32,637 |
| 语料字符覆盖 | **100%** (零缺失) |

## 参数影响

| 层 | 旧 (151936) | 新 (32647) | 节省 |
|----|------------|-----------|------|
| embed_tokens | 155.6M | 33.4M | 122.2M |
| lm_head | 155.6M | 33.4M | 122.2M |
| **合计** | **311.2M** | **66.9M** | **244.3M** |

总模型参数: ~598M → **~477M**

## 实施改动

| 文件 | 改动 |
|------|------|
| `ocr_tokenizer.py` | 新增 — HF 兼容字符级 tokenizer |
| `configuration.py` | vocab_size → 32647, token ID 更新 |
| `config.json` | 同上 |
| `modeling.py` | `load_pretrained_components()` 处理 vocab 不匹配 |
| `train_haina_cpt.py` | token ID 常量 (5/6/7) + 自动加载 OCR tokenizer |
| `render_collator.py` | token ID 默认值 |
| `data/ocr_vocab/vocab.txt` | 最终 32647 行词表 |
| `data/ocr_vocab/tokenizer/` | 构建好的 HF tokenizer |

## 词表扩充方法

加入新语料时，扫描字符集 → 找到 vocab.txt 中缺失的 → 追加到文件末尾 → 重新运行 `build_and_save_tokenizer()`。

```python
from ocr_tokenizer import build_and_save_tokenizer
build_and_save_tokenizer('data/ocr_vocab/vocab.txt', 'data/ocr_vocab/tokenizer')
```

## 回溯

- 词表来源: [PP-OCRv5 dict](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt)
- 构建逻辑: 本项目 `data/ocr_vocab/` 目录
- Tokenizer: `hainaocr_nativepixel/hainaocr_nativepixel/ocr_tokenizer.py`
- 设计讨论: 本文件
