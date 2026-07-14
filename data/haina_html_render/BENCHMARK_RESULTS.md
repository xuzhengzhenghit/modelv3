# 本环境可行性测试结果

测试环境：

- Python 3.13.5
- PyTorch 2.10.0 CPU build
- 系统 Chromium `/usr/bin/chromium`
- 本地 KaTeX
- 画布 1024×512
- 输出 `[3, 512, 1024]` `torch.uint8`
- 样本包含普通科学正文、行内公式、行间公式和表格

## 单个常驻 Chromium Page

20 个正式样本，3 个 warm-up：

```text
wall time        : 6.599 s
throughput       : 3.03 pages/s
average PNG size : 48.3 KiB
overflow pages   : 0
KaTeX errors     : 0
DOM/KaTeX mean   : 112.19 ms
screenshot mean  : 121.38 ms
PNG decode mean  : 12.36 ms
to uint8 tensor  : 28.20 ms
total mean       : 274.13 ms
```

## 每条样本重新加载模板的反例

```text
throughput       : 2.81 pages/s
DOM/KaTeX mean   : 224.42 ms
total mean       : 355.36 ms
```

常驻模板将 DOM/KaTeX 阶段从约 224 ms 降到约 112 ms。这里 reload 测试仍复用了浏览器进程；如果每条样本连 Chromium 进程也重启，代价会更高。

## 两个 DataLoader 渲染 worker

batch size=2，20 页，完成 warm-up 后：

```text
workers           : 2
throughput        : 12.52 pages/s
mean batch wait   : 159.71 ms
median batch wait : 158.69 ms
```

先前的小规模对照：

```text
1 worker : 约 6.57 pages/s
2 workers: 约 12.09–12.52 pages/s
4 workers: 约 12.34 pages/s
```

当前容器在 2 workers 左右已经接近 CPU/Chromium 饱和，继续增加 worker 没有明显收益。训练服务器需要重新测试；不要直接照搬这个 worker 数。

## 结论

- 方案已经完成从混合文本到 HTML/KaTeX，再到内存截图和 Tensor 的全链路验证。
- 单 Page 渲染的主要瓶颈是 DOM/公式排版与 Chromium 截图，不是 PNG 解码。
- CPU worker 中保留 `uint8` 很重要；应在异步传入 GPU 后再转 bf16/float。
- 是否能完全喂满 GPU，要用：

```text
所需页/秒 = 全局 batch size ÷ 单步训练秒数
```

与服务器实测的多 worker 页/秒比较。
