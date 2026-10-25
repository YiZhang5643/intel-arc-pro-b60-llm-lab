# Attribution and scope

This repository's original scripts and documentation use the MIT license. Model weights, Intel binaries and third-party patches are not distributed here and retain their own terms.

- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT: inference engine and backend tests. The pinned source is b11869, commit `d327bf1f0780f87b8c01b367b61853b0c291d7c4`.
- [Steve Miles / local-llm-benchmarks](https://github.com/srmiles/local-llm-benchmarks/tree/main/configs/patches): source of the Q4_0 MoE expert reorder function used in the measured local build. Our adapter extracts that function from a separately obtained patch and adapts its call sites to b11869. We do not vendor or relicense that implementation. Consult the original author/repository for its terms before redistribution.
- [Bartowski's Nemotron GGUF](https://huggingface.co/bartowski/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF): the model card declares `openmdw-1.1`. Consult that card and the original model terms.
- Intel oneAPI, oneDNN, Level Zero SDK and graphics drivers: obtain from their official sources and follow their respective terms.

Historical output samples are machine-generated benchmark responses, not verified reference explanations. Local paths, process IDs and unrelated files have been excluded. This is an independent community experiment, not an Intel or NVIDIA project.

Qwen experiment weights are the [Huihui Swift-1.5 GGUF derivative](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF). This particular file embeds a reference to the [Swift upstream license](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-27b/blob/main/LICENSE); its metadata is preserved in the experiment snapshot. These model weights are not distributed or relicensed by this repository.
