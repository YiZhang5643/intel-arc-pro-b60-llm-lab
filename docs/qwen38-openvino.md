# OpenVINO：先解决格式兼容，再测 96K 冷输入

用户此前在本机观察到 OpenVINO 较快的预填充，因此将它作为 1200 tok/s 目标的优先验证路线。当前完成兼容性检查，速度结果另行记录。

## 已确认的格式限制

OpenVINO IR 是专用的图与权重表示：XML 描述计算图、BIN 保存数据，结构有[公开文档](https://docs.openvino.ai/2023.3/openvino_ir.html)。它需要额外的模型副本，但并不限于官方预制权重；可以从受支持的原始 Hugging Face 模型导出并压缩。

[最新 GenAI 文档](https://openvinotoolkit.github.io/openvino.genai/docs/guides/gguf/)也提供直接 GGUF 路线，属于预览功能。必须核对已安装包，不能根据主分支文档认定正式 wheel 支持了全部架构和量化。

本机隔离安装 `openvino-genai==2026.4.1.0`，自动匹配 `openvino==2026.4.1`、`openvino-tokenizers==2026.4.1.0`，检测结果：

- `GPU.0` 是 UHD 770，**`GPU.1` 才是 B60**；本次只选择 B60。
- wheel 的前端列表没有 `gguf`，也没有新版 `GGUFTokenizer` 接口。
- 正式版 GenAI 自带的旧 GGUF 读取器尝试当前 Swift/Huihui 混合 IQ 权重，约 0.46 秒后在 `gguf_tensor_to_f16` 解包处失败；没有进入推理，不能据此评价 OpenVINO 的速度。
- [该发布版源码](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/src/gguf_utils/gguf.cpp)对没有等价数据类型的量化权重先转 FP16。不能假定在 GGUF 中压缩到十余 GB，就能以同样大小执行。
- [新前端支持矩阵](https://github.com/openvinotoolkit/openvino/blob/fde316cf48b371dd495e2e0ce840056a6f8a8356/src/frontends/gguf/docs/supported_models.md)列出了 `qwen35` 混合架构；[量化列表](https://github.com/openvinotoolkit/openvino/blob/fde316cf48b371dd495e2e0ce840056a6f8a8356/src/frontends/gguf/docs/quantization.md)仍未列出本文件使用的 IQ 系列。源码覆盖不等于本机 wheel 可用。

原始错误、版本和完整输入文本的 tokenizer 一致性校验见[兼容性证据](../results/2026-10-25-qwen38-openvino-compatibility/metadata.json)。测试结束已恢复原 SYCL 服务。

## 优先使用已有 INT4 IR 做框架对照

官方已有 [OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov/tree/7b20ea8c7df7223a85ff27bab5c473f0cb038943)，包含独立 MTP 图，无须本机先加载约 54GB BF16 再导出。固定版本 `7b20ea8c7df7223a85ff27bab5c473f0cb038943`。

这是原版 Qwen 的 INT4 转换，**不是当前 Swift/Huihui 衍生权重**，因此测试比较完整部署方案，不能把差距全归因于某个内核或量化格式。保持原 GGUF 和默认服务，另存 IR 用于实验。

当前官方 MTP 路径要求贪心解码、静态草稿数和关闭前缀缓存。冷输入实验适用；长对话前缀复用与现有 SYCL 配置不同，不能在没有额外验证时替换生产服务。

复现准备：

```powershell
python -m venv C:/MyLLM/nemotron-sycl/openvino-venv
C:/MyLLM/nemotron-sycl/openvino-venv/Scripts/python.exe -m pip install openvino-genai==2026.4.1.0
python scripts/download_openvino_snapshot.py --revision 7b20ea8c7df7223a85ff27bab5c473f0cb038943 --output C:/MyLLM/nemotron-sycl/models/Qwen3.8-27B-int4-ov
```

[下载器](../scripts/download_openvino_snapshot.py)要求固定提交，校验每个文件大小和 LFS SHA256；模型文件不入库。[基准脚本](../scripts/bench_openvino.py)记录批量、缓存、MTP、实际输入数量、文本哈希、TTFT 与检索结果。输入经过解码与重新分词，会明确记录其与原 token 数组的差异。

实际计时使用关闭前缀缓存后的新请求；指标 `prefill_plus_first_token_tps` 将首 token 工作一起计入，比单独的 kernel 预填充时间更保守。首次编译/加载和预热另记；使用相同完整文档，不使用预存前缀后的短追加来代替 96K 冷输入。
