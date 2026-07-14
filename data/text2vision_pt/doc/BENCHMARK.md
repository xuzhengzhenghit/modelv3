# Benchmark Results

**Date**: 2026-07-14
**Data**: onesci_cc_pages (730 chars avg/page, 200 samples tested)

## Single-worker throughput

| Stage | Mean | P50 | P95 | Share |
|-------|------|-----|-----|:-----:|
| Layout (DOM + KaTeX) | 35.1ms | 21.9ms | 90.2ms | 13% |
| Screenshot | 238.7ms | 239.6ms | 291.5ms | 85% |
| Decode (PNG → tensor) | 5.4ms | 4.8ms | 6.4ms | 2% |
| **Total** | **279.2ms** | **287.5ms** | **310.8ms** | — |

Single-worker throughput: **3.6 samples/s**, **1,512 visual tokens/s**

## Multi-worker throughput

| Workers | Throughput | Efficiency | Per-sample latency |
|:-------:|-----------:|:---------:|:------------------:|
| 1 | 3.6 samples/s | 100% | 274ms |
| 2 | 5.7 samples/s | 79% | 349ms |
| 4 | 6.9 samples/s | 47% | 583ms |

Efficiency drops above 2 workers due to Chromium CPU contention.

## Training feasibility

| Training scenario | Required throughput | 2 workers sufficient? |
|-------------------|--------------------:|:---:|
| batch=32, step=2s | 16 samples/s | Yes (but some wait) |
| batch=32, step=3s | 11 samples/s | Yes |
| batch=16, step=2s | 8 samples/s | Yes |
| 8-GPU DDP (effective batch=256) | ~1 sample/s per GPU | Yes |

2 workers can comfortably feed most single-GPU training configurations.
The bottleneck is Chromium screenshot (~85% of total time).

## Dynamic canvas distribution

From 200-sample test:

| Paper size | Count | Share | Visual tokens |
|-----------:|------|:-----:|:------------:|
| 1024×256 | 3 | 2% | 256 |
| 1024×320 | 20 | 12% | 320 |
| 1024×384 | 52 | 31% | 384 |
| 1024×448 | 57 | 34% | 448 |
| 1024×512 | 34 | 20% | 512 |
| (SPLIT) | 38 | 19% | — |

Average: **422 visual tokens/sample** (vs fixed 512, saving 18%)
