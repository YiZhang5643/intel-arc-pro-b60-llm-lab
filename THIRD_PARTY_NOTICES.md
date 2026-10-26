# Attribution and scope

This repository's original scripts and documentation use the MIT license. Model weights and Intel binaries are not distributed here and retain their own terms. The explicitly identified third-party patches below retain their original licence and attribution.

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT: inference engine and backend tests. The pinned source is b11869, commit `d327bf1f0780f87b8c01b367b61853b0c291d7c4`.
- [Steve Miles / local-llm-benchmarks](https://github.com/srmiles/local-llm-benchmarks/tree/main/configs/patches): source of the Q4_0 MoE expert reorder function used in the measured local build. Our adapter extracts that function from a separately obtained patch and adapts its call sites to b11869. We do not vendor or relicense that implementation. Consult the original author/repository for its terms before redistribution.
- [Bartowski's Nemotron GGUF](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF): the model card declares `openmdw-1.1`. Consult that card and the original model terms.
- Intel oneAPI, oneDNN, Level Zero SDK and graphics drivers: obtain from their official sources and follow their respective terms.

Historical output samples are machine-generated benchmark responses, not verified reference explanations. Local filesystem paths and unrelated files have been excluded; profiling traces retain process/thread IDs to correlate events. This is an independent community experiment, not an Intel or NVIDIA project.

Qwen experiment weights are the [Huihui Swift-1.5 GGUF derivative](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF). This particular file embeds a reference to the [Swift upstream license](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE); its metadata is preserved in the experiment snapshot. These model weights are not distributed or relicensed by this repository.

GPU profiling uses [Intel PTI / unitrace](https://github.com/intel/pti-gpu), whose tool source carries MIT SPDX notices. Source commit, tool hashes and measured configuration are recorded with the traces. Its source and compiled binaries are not vendored here.

The October 25 kernel experiments vendor these MIT-licensed llama.cpp changes; see [the retained licence](manifests/llama-MIT-LICENSE.txt):

- [Torchit1's token-blocked SYCL GDN](https://github.com/Torchit1/llama.cpp/tree/ff6c13e692f06cfc9e55e26b088f74aca48c7e7a): the local adaptation in `manifests/qwen38-gdn-blocked.patch` adds an opt-in switch and preserves the current cache-fusion path. The companion test patch adds B60/Qwen and MTP snapshot cases. Performance comments copied from that implementation describe its author's B580 measurements, not our B60 results.
- Intel/Russell Liu's Vulkan FA prefill commits `61cc4c25a6878bfcff2d2d76870ba1948ee0f3e2` and `1a5137a44fe7b2c921fe6b5f7d62704226bac599`, obtained from [edagher92-coder/arc-llm-lab](https://github.com/edagher92-coder/arc-llm-lab/tree/6d26bb1980854b7dfde77817653f0e10b123818a/engine/patches). The original patch bytes, author lines and SHA256SUMS are retained under `manifests/intel-vulkan-fa/`. A separately identified local third patch logs the first Intel prefill dispatch once during warmup; it does not change shader arithmetic. No binary is redistributed. This reference project is an independent community lab.
- The standalone SDPA probe follows llama.cpp's oneDNN graph pattern; it does not contain or redistribute model weights.

The optional portable Vulkan toolchain is built from Google shaderc, Khronos glslang, SPIRV-Headers, SPIRV-Tools and Vulkan-Headers. Their source is downloaded separately using pinned commits/checksums in `manifests/vulkan-toolchain-sources.json`; consult each upstream repository's licence. The build derives an import library from the Vulkan loader already installed in Windows and does not replace that loader.

The standalone Q4/INT8 GEMM probe uses oneDNN primitive attributes from [Torchit1/llama.cpp](https://github.com/Torchit1/llama.cpp/tree/8f09ea4166e8207c701671d77cd30b12b67a0295), MIT; the retained llama.cpp licence above applies to that reference. Synthetic input generation and comparisons are local probe code. OpenVINO software and downloaded models remain external to the repository and retain their upstream terms.

OpenVINO IR experiments use the external [OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov/tree/7b20ea8c7df7223a85ff27bab5c473f0cb038943) snapshot. Its card declares Apache-2.0. The unmodified native chat template is rendered from that separately downloaded snapshot; weights, template, OpenVINO binaries and Jinja2 are not vendored. Consult their upstream licences.

The local GenAI dense-tail correction and regression tests adapt OpenVINO GenAI commit ab01d1771e4adaa0e1e08e9d630def170572506c, Apache-2.0. See manifests/openvino-genai-dense-tail.patch and the retained manifests/openvino-genai-Apache-2.0.txt. The independent experimental DLL is built externally and is not distributed here.
