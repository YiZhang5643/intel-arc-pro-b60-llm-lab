# Intel Arc Pro B60 24GB · 本地大模型实验室

[English](README.en.md) · [完整结果](docs/nemotron.md) · [Windows 安装](docs/setup-windows.md) · [测试方法](docs/methodology.md) · [工具与来源](docs/tools.md)

在一台普通 Windows 电脑上，记录 B60 运行本地大模型时真正需要等多久。重点是 **32K–96K 长输入、首字等待、连续对话缓存和 MTP**。每个结论附配置、原始测量和复现脚本，后续可增加不同模型。

新增：[Qwen3.8-27B 混合 IQ 权重调优](docs/qwen38.md)，包含 32K/96K、嵌入式 MTP 与独立草稿批量对照；新增 [算子时间线分析](docs/qwen38-profile.md)与完整 trace；后续 [内核改进实测](docs/qwen38-kernel-experiments.md)：保持 MTP 开启，32K 预填充约 53.7 秒、96K 约 208.2 秒。

后续研究：[纯 Q4 与 1200 tok/s 目标](docs/qwen38-1200-target.md)、[OpenVINO 格式兼容与对照准备](docs/qwen38-openvino.md)。

首个模型：**NVIDIA Nemotron 3.5 Lightning 30B-A3B，Q4_0 主模型 + 可选 Q8_0 MTP**。这不是模型能力排行榜，也没有证明它是 B60 上最聪明的模型。

## 先看结果

测试机：Arc Pro B60 24GB，i5-12600K，32GB DDR4，Windows 11；驱动 `32.0.101.9033`。2026-10-24 测量、10-25 整理。

| 冷输入 | 官方 SYCL 基线：处理输入 / 请求完成 | 调优后的快速预填充：处理输入 / 首字 / 请求完成 |
|---|---:|---:|
| 32K（32,768 tokens） | 482 tok/s / 69.38 s | **1,454 tok/s / 22.61 s / 24.08 s** |
| 96K（98,304 tokens） | 455 tok/s / 218.00 s | **1,184 tok/s / 83.30 s / 85.28 s** |

这是两套完整配置的对比：优化后采用 FP16/oneDNN 构建、Q4_0 MoE reorder，并关闭 MTP，将 batch/ubatch 提至 8192；基线开启 MTP、batch 2048 / ubatch 1024。**约 133 秒的 96K 请求耗时差，不能归功于某一个补丁。** 两边均找对全部 3 个隐藏条目；每个长度各一次测量，尚无置信区间。

同为优化构建、batch/ubatch 2048、相同 96K token 数组时，MTP 的结果更复杂：

| 场景 | MTP 关 | MTP 开 | 如何理解 |
|---|---:|---:|---|
| 冷输入首字 | 93.24 s | 104.65 s | MTP 增加首次等待 |
| 缓存后检索生成 | 34.07 tok/s | 49.24 tok/s | 开启约快 45% |
| 缓存后代码生成 | 33.86 tok/s | 34.75 tok/s | 约快 3%；均达到 768 token 上限 |
| 缓存后英文说明 | 34.08 tok/s | 27.11 tok/s | 开启约慢 20%；输出长度不同 |

因此提供两套配置：**长对话 MTP 开启**，以及 **新长文档优先的快速预填充（MTP 关闭）**。MTP 不是长上下文必然加速的开关。缓存命中的 96K 后续提问，首字可以低于 1 秒；它与每次重新处理整篇文档是两种工作负载。

## 复现

需要 Python 3.10+，基准脚本只用标准库，可选绘图需 Matplotlib；模型和运行库另行下载，见 [安装步骤](docs/setup-windows.md)。

```powershell
# 默认前台运行，MTP 开启；--dry-run 可先检查命令
python scripts/serve.py --preset presets/nemotron-long-chat-mtp.json --runtime-dir D:/llm/runtime --model-dir D:/llm/models --host 0.0.0.0 --port 5001

# 另一个终端；请为每轮使用新的输出目录
python scripts/bench_chat.py --label mtp-on --output runs/mtp-on --preset presets/nemotron-long-chat-mtp.json
python scripts/bench_prefill.py --label prefill --output runs/prefill --sizes 12288,32768,98304 --repeats 3
```

启动器默认监听 `127.0.0.1:5001`；示例中的 `--host 0.0.0.0` 用于局域网访问。它不修改防火墙，也不接管其他服务器。运行基准前停用其他推理请求，确认选中了 B60，而非核显。

## 仓库里有什么

- [原始结果与版本指纹](results/2026-10-24-nemotron/metadata.json)、[逐项测量 CSV](results/2026-10-24-nemotron/measurements.csv)，以及可检查完整性的 SHA256 清单。
- `scripts/`：启动、文件校验、冷预填充、长对话、历史检索、短生成和基础正确性检查。
- [完整调优记录](docs/nemotron.md)：12K batch 扫描、MTP 对照、数值验证及未采用的实验。
- [新增模型指南](CONTRIBUTING.md) 和 [结果模板](templates/model-result.md)。不同 tokenizer 的 tok/s 不直接代表相同文字吞吐。

新增 [Intel Vulkan FA 补丁的本机实测](docs/qwen38-kernel-experiments.md)：当前混合 IQ 权重上仍慢于 SYCL；没有对未打补丁的上游做独立回归。Intel 专用框架的适配调查见 [工具与来源](docs/tools.md)。Qwen 的算子时间线已定位主要耗时，但尚未证明预填充已到硬件极限。

代码许可：MIT；第三方补丁、模型和二进制的范围见 [来源说明](THIRD_PARTY_NOTICES.md)。
