# HTML Render 吞吐测试结果

**测试日期**: 2026-07-13
**测试环境**: Python 3.10.10, PyTorch 2.6.0+metax3.3.0.2, Chromium 149.0.7827.55

## 测试配置

| 参数 | 值 |
|------|-----|
| canvas | 1024 × 512 |
| visual tokens@32 | 512 |
| tensor mode | uint8 |
| batch size | 4 |
| manifest 样本 | 19650 pages (from onesci_cc_core) |

## 单 Page 吞吐（无 DataLoader）

| 指标 | 值 |
|------|-----|
| throughput | **3.98 pages/s** |
| DOM/KaTeX mean | 44.80 ms |
| screenshot mean | 194.85 ms |
| PNG decode mean | 8.34 ms |
| to tensor mean | 1.22 ms |
| total mean | 249.21 ms |
| overflow pages | 0 |
| KaTeX errors | 0 |
| avg PNG size | 41.6 KiB |

## 多 Worker 吞吐

| workers | throughput | mean batch wait | 相对扩展 |
|---------|-----------|-----------------|----------|
| 1 | **7.93** pages/s | 504.59 ms | 1× |
| 2 | **15.78** pages/s | 253.42 ms | 2.0× |
| 4 | **31.96** pages/s | 125.14 ms | 4.0× |
| 6 | **49.07** pages/s | 81.51 ms | 6.2× |

## 结论

- 缩放接近线性，每增加 2 worker 吞吐翻倍
- 瓶颈在 Chromium 截图（~78% 耗时），非 DOM/KaTeX/PNG 解码
- 按 global batch=32, step=2s 估算，**4 workers 即可喂满训练**（需 16 pages/s，实际 31.96）

## 运行命令

```bash
# 单 Page 基准
python benchmark_html_render.py \
  --count 100 --warmup 10 \
  --browser-path /root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
  --katex-dist ./node_modules/katex/dist \
  --tensor-mode uint8 --save-first 8

# 多 Worker 基准
python benchmark_dataloader.py \
  --manifest /tmp/bench_manifest.jsonl \
  --workers 4 --batch-size 4 --batches 30 --warmup-batches 5 \
  --browser-path /root/.cache/ms-playwright/chromium-1228/chrome-linux64/chrome \
  --katex-dist ./node_modules/katex/dist \
  --output-mode uint8
```
