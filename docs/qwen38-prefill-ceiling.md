# B60 的 Qwen3.8-27B 预填充算力上界

分析日期：2026-10-25。结论：**可以估算乐观上界，不能仅凭规格预测实际吞吐。** 对本机这份混合 IQ GGUF，主要矩阵计算与理想因果注意力折算出的上界为 32K 约 1,783 tok/s、96K 约 1,445 tok/s；当前默认实测分别约 571、449 tok/s。差距值得分析，但不等于软件能追回全部差距。

## 显卡峰值选哪个

Intel [B60 规格](https://www.intel.com/content/www/us/en/products/sku/243916/intel-arc-pro-b60-graphics/specifications.html)列出 20 个 Xe2 核、2.4 GHz、197 INT8 TOPS、12.28 FP32 TFLOPS、456 GB/s。
[Intel Xe2 架构材料第 27 页](https://cdrdv2-public.intel.com/824443/2024_Intel_Tech%20Tour%20TW_Lunar%20Lake%20Architecture%20Session%20Highlights-3.pdf#page=27)给出每核 INT8 4096、FP16 2048 ops/clock。依同架构吞吐比例推算：

```text
FP16 XMX 峰值 = 20 × 2048 × 2.4 GHz = 98.304 TFLOPS
INT8 XMX 峰值 = 20 × 4096 × 2.4 GHz = 196.608 TOPS ≈ 标称 197
```

98.304 是按架构与标称频率推导的矩阵峰值，不是本机实测持续算力。200 W 档位、实际时钟、温度及算子实现都会影响持续性能。不要混用 FP32 向量吞吐、FP16 向量吞吐和 XMX 矩阵吞吐。

当前 SYCL F16/oneDNN 的大批量矩阵路径会解量化后计算。GGUF 存储为 IQ3/IQ2 等，不意味着直接按 INT8 峰值执行，也不意味着参数运算量按位宽同比缩小。

## 按实际模型结构计算

尺寸来自[已保存 GGUF 检查](../results/2026-10-25-qwen38/metadata.json)，图结构参照固定版本 [qwen35.cpp](https://github.com/ggml-org/llama.cpp/blob/d327bf1f0780f87b8c01b367b61853b0c291d7c4/src/models/qwen35.cpp)。主模型 64 层：16 层完整注意力、48 层 Gated DeltaNet；额外 1 层 MTP 在此估算中省略。FMA 按 2 FLOPs。

| 每 token 主模型线性投影 | GFLOPs |
|---|---:|
| 64 层 FFN | 34.2255 |
| 16 层完整注意力的 Q/gate、K、V、输出投影 | 3.3554 |
| 48 层循环注意力的投影 | 11.1201 |
| 合计 | **48.7011** |

已额外读取实际 GGUF 张量描述符，汇总主模型二维矩阵（排除卷积）得到 24,350,556,160 个元素，即每 token 48,701,112,320 FLOPs，与上述尺寸公式一致。没有读取整个权重数据区。

零前缀缓存、长度 N 的冷输入，完整注意力有 N(N+1)/2 个因果位置对；Q 头数 24、头维 256，QK 与 AV 的平均计算量为：

```text
注意力 FLOP/token = 16 × 24 × (256 + 256) × (N + 1)
所计 GFLOP/token = 48.70111232 + 0.000196608 × (N + 1)
乐观 tok/s = 98,304 / 所计 GFLOP/token
乐观耗时下界 = N / 乐观 tok/s
```

这只统计主要投影与理想因果 QK/AV，并假设所计运算始终达到 FP16 XMX 峰值。**未计入** DeltaNet 状态与分块计算、MTP、激活/归一化、末端选定输出的词表投影、解量化、显存读写、额外 masked/padding 运算、kernel 启动、同步及主机调度。64 层 FFN 均按全部 token 计算，未模拟最后一层输出裁剪。某些实际算子使用其他精度或执行单元，不能达到该矩阵峰值。因此这是简化计算模型的乐观参考上界，不是精确完整 roofline，也不是可实现速度承诺。

## 与当前默认测量对照

12K、32K、96K 分别是 12,288、32,768、98,304 tokens。实测为已热身、前缀缓存为零的请求；当前默认开启 MTP3。每项仅一次测量。

| 冷输入 | 所计 GFLOP/token | 乐观峰值折算 tok/s | 乐观耗时下界 | 当前实测 tok/s | 当前预填充耗时 |
|---|---:|---:|---:|---:|---:|
| 12K | 51.12 | 1,923 | 6.39 s | 561 | 21.91 s |
| 32K | 55.14 | 1,783 | 18.38 s | 571 | 57.34 s |
| 96K | 68.03 | 1,445 | 68.03 s | 449 | 218.85 s |

实测出处：[12K](../results/2026-10-25-qwen38/qwen38-mtp3-draft512/summary.json)、[32K](../results/2026-10-25-qwen38/long-mtp3-q8-32k/summary.json)、[96K](../results/2026-10-25-qwen38/long-mtp3-q8-96k/summary.json)。表格使用服务端 prompt_ms，首字时间还会稍长。

所计运算量乘以实测 tok/s，折算约 29–32 TFLOPS、约峰值的 29%–32%。**这不是 profiler 测得的 XMX 利用率**：分子不含完整工作量，分母也不是所有算子都能使用的持续算力。

从 32K 增至 96K，所计总算量增长约 **3.70 倍**，实测预填充时间增长约 **3.82 倍**。额外变慢与完整注意力的增长量级接近；这一吻合既不能定位瓶颈，也不能证明当前实现已最佳。

## 如何使用这个估算

- 当前 450–570 tok/s 不能直接称为 B60 的硬件极限；同样也不能承诺可达到 1,400–1,800 tok/s。
- 456 GB/s 除以模型大小，主要描述逐 token 反复读权重时的带宽约束；大批量预填充会复用权重，不能沿用该式估算预填充。
- 更小量化可能增加解量化成本；不能由文件大小判断预填充速度。
- 下一步应对 32K/96K 做算子时间线，区分 GEMM、DeltaNet、注意力、转换/拷贝与主机间隙；再比较兼容量化与后端。仅从峰值差距不能确定哪一项值得改。
- MTP 的既有优势主要体现为生成速度；本表没有把它当成冷预填充加速器。本次分析未重启服务或运行新的 GPU 基准。

复算：

```powershell
python scripts/estimate_qwen38_prefill.py
python scripts/estimate_qwen38_prefill.py --tokens 32768 65536 98304 --peak-tflops 98.304
```

[脚本](../scripts/estimate_qwen38_prefill.py)只读已保存元数据，输出明确标注省略项的 JSON。[默认参数输出](data/qwen38-prefill-ceiling.json)是分析数据，不是新测量。
