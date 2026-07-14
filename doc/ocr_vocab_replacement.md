# OCR 词表替换方案

**状态**: 准备阶段 — 词表已构建，tokenizer/model 适配待实施

## 背景

Qwen3-0.6B 原始词表 151936 tokens（BPE subword），对纯 OCR 任务来说过大。每个输出位置需计算 151936 类 softmax，embedding + lm_head 占 **311M 参数**。

## 方案

用 **PP-OCRv5 字符级词表** 替代 Qwen3 BPE 词表。

## 词表来源

| 来源 | 文件 | 大小 |
|------|------|------|
| PP-OCRv5 官方字典 | `data/ocr_vocab/ppocrv5_dict.txt` | 18383 字符 |
| + 空白字符 (space, tab) | — | 2 |
| + 语料缺失字符 | onesci_cc_pages 扫描 | 608 |
| + 特殊 token | pad, eos, unk, im_start/end, vision tokens, ocr | 10 |

**最终词表**: `data/ocr_vocab/vocab.txt` — **19002 tokens**

## 特殊 token ID 映射

| Token | ID |
|-------|-----|
| `<\|pad\|>` | 0 |
| `<\|eos\|>` | 1 |
| `<\|unk\|>` | 2 |
| `<\|im_start\|>` | 3 |
| `<\|im_end\|>` | 4 |
| `<\|vision_start\|>` | 5 |
| `<\|vision_end\|>` | 6 |
| `<\|image_pad\|>` | 7 |
| `<\|ocr\|>` | 8 |
| `\n` (newline) | 9 |
| ` ` (space) | 18392 |

## 参数节省

| 层 | 旧 (151936) | 新 (19002) | 节省 |
|----|------------|-----------|------|
| embed_tokens | 155.6M | 19.5M | 136.1M |
| lm_head | 155.6M | 19.5M | 136.1M |
| **合计** | **311.2M** | **38.9M** | **272.3M** |

总模型参数: ~598M → ~**326M**

## 待实施步骤

1. 构建 HF 兼容 character-level tokenizer（`encode`/`decode`）
2. 更新 `configuration.py`: `vocab_size` → 19002
3. 更新 `config.json`: `vocab_size`, token IDs
4. 更新 `train_haina_cpt.py`: `VISION_START`/`VISION_END`/`IMAGE_PAD` 常量 → 读 config
5. 更新 `haina_cpt_collator.py` / `render_collator.py`: token IDs → 参数化
6. 重新初始化 embedding/lm_head（其余权重可复用）
7. Smoke test

## 代码回溯

- 词表构建脚本: `data/ocr_vocab/build_vocab.py`（待创建）
- 词表来源: [PP-OCRv5 dict](https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/utils/dict/ppocrv5_dict.txt)
- 设计讨论: 本文件
- 实施 commit: GitHub history
