# 工具、下载与框架调查

以下是 2026-10-24/25 查阅的一手入口。链接指向会更新的页面；已测版本与文件哈希见 [metadata](../results/2026-10-24-nemotron/metadata.json)。

| 工具 | 用途与入口 |
|---|---|
| llama.cpp b11869 | [固定发布](https://github.com/ggml-org/llama.cpp/releases/tag/b11869)、[Windows SYCL ZIP](https://github.com/ggml-org/llama.cpp/releases/download/b11869/llama-b11869-bin-win-sycl-x64.zip)、[SYCL 文档](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md) |
| Nemotron GGUF | [Bartowski 模型页](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF)，需要主模型 Q4_0 与可选 mtp Q8_0 两个文件 |
| Q4 MoE reorder | [Steve Miles 的补丁目录及说明](https://github.com/srmiles/local-llm-benchmarks/tree/main/configs/patches)，[原始 Q4_0/Q8 patch](https://raw.githubusercontent.com/srmiles/local-llm-benchmarks/main/configs/patches/0001-sycl-extend-MoE-expert-tensor-reorder-to-Q4_0-and-Q8.patch)；适配器校验已测文件哈希 |
| oneAPI DPC++/C++ | [官方编译器下载](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html)、[Windows 单组件安装](https://www.intel.com/content/www/us/en/docs/oneapi-toolkit/installation-guide-windows/latest/install-standalone-tools.html)；本轮 2026.1.1 |
| oneDNN | [官方项目与安装说明](https://github.com/uxlfoundation/oneDNN)；本轮 Intel 组件 2026.0.2 |
| Level Zero | [SDK 发布](https://github.com/oneapi-src/level-zero/releases)；本轮 SDK 1.32.0 |
| Visual Studio C++ | [官方 Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022)；需要 x64 MSVC/Windows SDK |
| CMake / Ninja | [CMake 下载](https://cmake.org/download/)、[Ninja 发布](https://github.com/ninja-build/ninja/releases) |
| Intel 驱动 | [官方驱动下载入口](https://www.intel.com/content/www/us/en/download-center/home.html)；已测 B60 版本 32.0.101.9033 |
| Python / Git | [Python](https://www.python.org/downloads/)、[Git for Windows](https://gitforwindows.org/)；基准和分析脚本仅用标准库；可选绘图需 Matplotlib |
| Intel unitrace | [固定源码与使用说明](https://github.com/intel/pti-gpu/tree/307d457292976823546d3b30b141402de44f724f/tools/unitrace)，本机编译2.4.1，采集Windows Level Zero内核/API时间线；[本机分析](qwen38-profile.md) |

## Intel 专用框架是否更合适

| 方案 | 对当前 Windows + Nemotron + MTP 的判断 |
|---|---|
| llama.cpp SYCL FP16/oneDNN + Q4 reorder | 已在本机部署、测试长输入，是当前有直接证据的方案。 |
| [Intel LLM Scaler / vLLM](https://github.com/intel/llm-scaler/tree/main/vllm) | 面向 Intel GPU 的优化值得后续测试；查阅文档以 Ubuntu 24.04 裸机为基础。未验证本目标模型、量化、Windows、单 B60 与 MTP 全组合，不能由其他模型数据推定收益。 |
| [ArcInt](https://github.com/marfrit/arcint) | 有专用 OpenCL/XMX 内核；查阅时支持架构与部署目标不覆盖本次 Windows Nemotron 组合。适合另建兼容模型/Linux 对照。 |
| [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) | 尚未验证目标 Nemotron 与 MTP 的完整转换和运行路径。 |

未验证不代表永远不支持。没有为了本轮调查更换系统或启用 WSL/Docker。新的框架结果应附可复现命令、模型格式及端到端 TTFT。

Windows Vulkan 的 >32K 历史退化没有本机重测。[Intel FA PR #29357](https://github.com/ggml-org/llama.cpp/pull/29357) 和 [PR #30351](https://github.com/ggml-org/llama.cpp/pull/30351) 可追踪进展，但 PR 状态不能证明问题已在本机解决。
