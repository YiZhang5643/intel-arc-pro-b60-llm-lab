# Qwen3.8-27B：纯 Q4 与 1200 tok/s 目标

目标按单张 B60 24GB、Windows、27B、**96K 完整冷输入**理解。当前推荐的 SYCL 配置保持 MTP 开启，实测约 **472 tok/s / 208.17 秒**；1200 tok/s 要把同样 98,304 tokens 的预填充压到 **81.92 秒**，需要约 **2.54 倍**吞吐。尚未达到这个目标。

## 纯 Q4 能解决多少

纯 Q4 值得作为兼容性与解码实验，但现有证据不支持仅换格式即可得到 2.54 倍预填充。

当前文件是混合 IQ/K/BF16 量化；大批量矩阵计算已经经过 GPU 解量化，使用 FP16 XMX。去掉 IQ 不等于首次启用 XMX。Q4_0、Q4_K 和 OpenVINO INT4 的分组、零点与执行路径也不同，不能统称“纯 Q4”后直接推断速度。

本机独立 oneDNN 3.11.4 探针，batch 8192、FP32 输出；每项预热后计时五次取均值：

| 计算路径 | FFN up：5120 → 17408 | FFN down：17408 → 5120 |
|---|---:|---:|
| 当前后端方向的 FP16 GEMM | 20.497 ms | 20.308 ms |
| 交换矩阵方向的 FP16 | 20.442 ms | 20.264 ms |
| oneDNN Q4 权重解压路径，W4A16 | 21.327 ms | 20.186 ms |
| 同一 U4 权重路径，输入改为 INT8 | 23.997 ms | 27.103 ms |
| 真正 W8A8 GEMM，标量缩放 | 10.834 ms | 10.281 ms |
| W8A8 + 独立逐行/逐列缩放内核 | 14.594 ms | 11.346 ms |
| W8A8 + 融合逐行/逐列缩放后处理 | 10.931 ms | 10.265 ms |

FP16 大矩阵约 71–72 TFLOPS；W8A8 约 135–142 TOPS，计数均为 `2MNK / 时间`，两种单位表示不同运算精度。INT8 数字展示了进一步研究的空间；它是算子结果，尚未接入本模型。

探针共 28 个可执行案例，合成数据与 FP16 参考的差异均为零；另 4 个逐轴 scale 属性描述符被当前 oneDNN 拒绝，已保留错误。**合成输入刻意选择可精确表示的数值**，未验证真实激活与当前 GGUF 的重新量化误差。计时不含 IQ/Q4 解包、激活量化与 scale 计算，也没有包含注意力、GDN 或模型显存管理。不能把这里的约两倍矩阵加速当成模型两倍加速或无损量化。

由 IQ 转成 Q4 也不会恢复上游 BF16 精度，可能再引入误差。更大的权重还会挤占长上下文缓存和计算批量。

## 为什么仍需要完整图优化

以下只是替换已测分类耗时的条件计算，保持其他耗时不变，**不是新后端速度预测**：

| 假设 | 32K 输入 | 96K 输入 |
|---|---:|---:|
| 当前实测 | 610 tok/s | 472 tok/s |
| 解量化全部免费 | 665 tok/s | 505 tok/s |
| GEMM 快两倍，解量化全部免费 | 871 tok/s | 615 tok/s |
| GEMM 与解量化全部免费 | 1260 tok/s | 787 tok/s |

32K 使用 blocked-GDN 新 trace；96K 复用此前 trace 的 GEMM/解量化分类，没有假装重新测过新版 96K trace。分类是 kernel 时间之和，可能重叠，所以这些值不是严格硬件上界。它们说明：96K 还需要同时处理注意力和图调度等耗时，单独针对权重解包不够。

计算脚本：[estimate_prefill_target.py](../scripts/estimate_prefill_target.py)。完整测量、版本与限制：[Q4 探针快照](../results/2026-10-25-qwen38-q4-probe/metadata.json)。

## 框架线索

- [Torchit1 的固定提交](https://github.com/Torchit1/llama.cpp/tree/8f09ea4166e8207c701671d77cd30b12b67a0295)：Q4_0/Q8_0 的 oneDNN 权重压缩接口，是探针的属性配置来源。本机 W4A16 探针没有得到明显提升；这不等于测过该分支的完整模型表现。
- [Mach X](https://github.com/Red-Weasel/machx-inference-engine)：作者报告了两张 B70 的高预填充；单张 B70、Q6_K 的公开值约 823 tok/s。硬件、长度与后端都不同，目前没有本机 Windows/B60/96K 的验证，不能拿双卡数字承诺 1200。
- [Arcint](https://github.com/marfrit/arcint)：作者的单 B60 27B Q4_K_M，55K 约 709 tok/s、138K 约 499 tok/s；README 明确将 Windows 列为非目标。它提供有用的 Linux 对照，还不是本机已部署方案。
- [Arc Tuned GGUF](https://huggingface.co/Greatjedi/Qwen3.8-27B-Intel-Arc-Tuned-GGUF)：主要展示去掉 IQ 后的解码收益，公开预填充值约 244–246 tok/s，测试配置不同。Swift 变体由已有低比特权重转量化，不应作为恢复原始精度的办法。
- OpenVINO：用户此前在本机观察到较快预填充，优先进行完整长输入对照，详见后续 [OpenVINO 验证](qwen38-openvino.md)。

## 复现矩阵探针

```powershell
scripts\build_q4_probe.cmd
python scripts/run_gpu_probe.py --deployment-dir C:/MyLLM/nemotron-sycl --executable runs/q4-gemm-probe.exe --output runs/q4-reproduction
python scripts/estimate_prefill_target.py --target-tps 1200
```

构建依赖和路径沿用本项目 oneAPI/oneDNN 环境；新输出目录必须不存在。探针运行器先验证记录中的服务空闲，暂停后执行，最后恢复原运行库、预设、设备和监听端口。当前推荐服务仍监听 `0.0.0.0:5001`。
