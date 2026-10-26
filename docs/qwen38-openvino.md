# OpenVINO：从格式兼容到 96K 冷输入加速

用户此前在本机观察到 OpenVINO 较快的预填充，因此将它作为 1200 tok/s 目标的优先验证路线。现已在 Windows/B60 上完成官方 INT4 IR 实测。下列结果均重新处理完整输入，关闭前缀缓存。


**当前结论：停止稀疏阈值调优，不采用这些稀疏候选。** 用户要求质量优先；完整注意力基线为 32K 约 982 tok/s、96K 约 646 tok/s，不能把两个长度混作一次加速比较。1200 tok/s 仍是研究目标，目前未得到质量可接受的达标配置。

[简洁阶段总结：32K 保留全部注意力块达到约 982 tok/s](2026-10-25-stage-summary.md)。

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


## IR 实测：全块路径与稀疏路径分开看

运行环境为 OpenVINO 2026.4.1 / GenAI 2026.4.1.0，GPU.1，U8 KV，单请求。主要指标是 **实际输入 tokens / 流式首个非空文本等待时间**，包含分词、调度和首 token 工作；内部 TTFT 也单独保存。32K 重新分词为 32,767 tokens，96K 为 98,304 tokens；与原测试解码出的完整文本一致。每配置每长度目前各一次计时，无置信区间。

| 配置 | 32K tok/s | 96K tok/s | 96K 首字 | 备注 |
|---|---:|---:|---:|---|
| SYCL blocked-GDN + MTP，原混合 IQ 衍生权重 | 610 | 472 | 208.2 s | 此列为此前处理输入的计时，口径略有不同 |
| IR 普通注意力，b8192，DQ32，MTP 开 | 542 | 未完成 | 超过 337 s 未出字 | 已中止，不能当完成吞吐 |
| IR 普通注意力，b2048，DQ0，MTP 关 | 604 | 未测 | — | 32K 筛选 |
| IR 普通注意力，b2048，DQ128，MTP 关 | 690 | 未测 | — | 32K 筛选 |
| XAttention 阈值 1.0，b2048，DQ128 | **982** | **646** | **152.2 s** | 保留所有注意力块 |
| XAttention 阈值 0.9 | 1051 | 701 | 140.3 s | 近似注意力 |
| XAttention 阈值 0.3 | 1256 | 1142 | 86.1 s | 近似注意力 |
| XAttention 阈值 0.15 | 1273 | 1192 | 82.5 s | 近似注意力 |
| XAttention 阈值 0.1，b4096 | — | 1135 | 86.6 s | 复杂质量 1/24，不采用 |

阈值 1.0 的 [GPU 发布版实现](https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/plugins/intel_gpu/include/intel_gpu/runtime/options.inl)允许跳过稀疏块选择；它用于保留全块的路径对照。其内核、缓存布局与数值舍入仍不同于普通 paged attention，不宣称逐位一致。此路径 32K 将 DQ128 改为 DQ1024 为 976 tok/s，改 b4096 为 918 tok/s，均未改善。

小于 1 的阈值会省略部分注意力计算，必须标为近似方案。[官方算法说明](https://openvinotoolkit.github.io/openvino.genai/docs/concepts/optimization-techniques/sparse-attention-prefill/)中，阈值是累计估计注意力质量的门限，不是固定保留百分比。KV 本身不丢弃，解码阶段仍使用完整 KV；这不意味着预填充产生的表示必定不受影响。

所有上表 XAttention 实验均 **MTP 关闭**。当前发布版 MTP 构造路径禁止 XAttention，不能把本表速度与 MTP 解码收益叠加承诺。96K 全块方案生成约 11.7 tok/s，0.3/0.15 约 12 tok/s；预填充收益明显，解码仍需优化。

早期普通注意力 MTP 实验的部分采样曾看到较高共享显存，但 Windows 共享显存计数包含主机映射等用途，不足以证明发生显存换页。全块 XAttention 的部分采样专用显存峰值约 22.31 GiB、共享约 2.17 GiB，同样不作为完整峰值或故障诊断。

## 质量检查和复现

三个代码检索项只是基础筛选，不能证明复杂任务质量不变。新增测试生成两份不同随机种子的约 96K 原生模板文档，每份包含四个三跳关联、四个版本冲突选择和四个顺序库存计算，共 24 个答案。事实分布在全文约 2.5%–97.5%，文档顺序与版本/事件顺序独立。答案由脚本生成，与模型输出无关。

发布版全块基线两份文档各 8/12，合计 **16/24**：三跳关联 8/8、版本选择 6/8、库存计算 2/8。阈值 0.1、b4096 的候选仅 **1/24**，损失基线原本正确的 15 项，没有新增正确项；原有三个代码检索仍全对。此候选不建议用于日常长文档，不能只凭简单检索判定质量合格。0.3 / 0.15 的早期速度筛选也尚不能据此视为质量验证通过。

模板直接使用下载模型的未修改 `chat_template.jinja`、`enable_thinking=false`，通过 Jinja2 渲染以避开当前 GenAI Minja 对 `is undefined` 的兼容错误。严格评分第一份答案对象的键、值与 JSON 类型；额外记录是否仅输出 JSON。围栏内 JSON 可以语义通过，但格式不通过。两份文档实际重新分词为 98,295 / 98,301 tokens，保留全部解码文本。

```powershell
# 基准控制器会确认原服务空闲，暂停服务，结束后恢复原运行库和监听配置。
# 首次复现时，先在原 SYCL 服务运行期间生成两份规范输入：
python scripts/bench_chat.py --label canonical32 --size 32768 --prepare-only --preset presets/qwen38-recommended.json --output runs/canonical32
python scripts/bench_chat.py --label canonical96 --size 98304 --prepare-only --preset presets/qwen38-recommended.json --output runs/canonical96

python scripts/run_openvino_bench.py --deployment-dir C:/MyLLM/nemotron-sycl --model C:/MyLLM/nemotron-sycl/models/Qwen3.8-27B-int4-ov --preset presets/openvino/qwen38-full-blocks.json --prompts runs/canonical32/prompts.json --prompts runs/canonical96/prompts.json --quality-in-last-timing --output runs/ov-full-reproduction

# 只做 CPU 分词准备；大体积 prompts.json 放在忽略的 runs/ 下。
C:/MyLLM/nemotron-sycl/openvino-venv/Scripts/python.exe -m pip install Jinja2==3.1.6
C:/MyLLM/nemotron-sycl/openvino-venv/Scripts/python.exe scripts/prepare_openvino_quality.py --model C:/MyLLM/nemotron-sycl/models/Qwen3.8-27B-int4-ov --seed 20261025 --output runs/quality-seed-20261025
```

[运行器](../scripts/run_openvino_bench.py)支持 `--dry-run`，保存解析后的实验配置、执行源码副本和 SHA256。`--quality-in-all-timings --quality-tokens 512` 可在每份长文档的冷输入计时中直接生成并评分，避免另做一次完整预填充。超时会停止该实验拥有的 Windows 子进程树，再恢复原服务，防止 venv 启动器退出后 GPU 子进程残留。

[原始 IR 测量与版本](../results/2026-10-25-qwen38-openvino-ir/metadata.json)和 [逐项计时](../results/2026-10-25-qwen38-openvino-ir/comparison.json)保留单次测量及中止记录。早期 MTP 关闭的筛选配置曾将 `cache_budget_scope` 文本误标为主模型与草稿合计；原始文件保留，实际 MTP 标志和 scheduler 字段才决定分配，后续脚本已修正标签。

默认 SYCL/MTP 服务继续监听 `0.0.0.0:5001`，实验预设不自动替换日常服务。


## 末尾全注意力的边界与 GPU 阈值约定

质量回归后检查 [GenAI 发布版调度器](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/src/continuous_batching/scheduler.hpp)发现两个需要一起处理的问题：

1. 判断使用 `processed_after_chunk < prompt_len`，没有覆盖最后一块正好到达末尾的情况。对于完整 96K、b2048、默认 dense tail=100，倒数第二块剩余 2048，大于 100；最后一块又被前一条件排除，因此没有预期的末尾全注意力。
2. 调度器用 0.0 表示关闭稀疏；[GPU CM 旁路](https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/plugins/intel_gpu/src/graph/impls/cm/paged_attention_gen.cpp)判断阈值 >=1。[块选择实现](https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/plugins/intel_gpu/src/graph/impls/cm/include/find_block.hpp)直接将阈值乘到估计总质量上，0 不能当作 GPU 多 token 全块旁路。

[本地实验补丁](../manifests/openvino-genai-dense-tail.patch)将边界改为包含最后一块，并对 dense prefill 返回显式全块阈值 1.0。解码与显式 dense tail=0 的行为保留；补上三种公共调度接口回归测试。它不是官方已发布修复，当前只针对本机 GPU 路线验证。完整预填充仍然近似，不能因为末尾恢复全注意力就声称无损。

构建固定在与 wheel 相同的 GenAI 提交，使用 MSVC Release、UTF-8 与 C++ 异常处理。独立构建关闭不参与本轮任务的 XGrammar，使用已安装 OpenVINO 运行库和 Python 绑定；源码补丁、编译参数和 DLL 哈希另存。程序不覆盖安装的 wheel，不自动改变日常服务。

```powershell
python scripts/build_openvino_dense_tail.py --deployment-dir C:/MyLLM/nemotron-sycl --jobs 4
# 命令输出 OVERLAY_READY 路径；后续运行器使用 --genai-overlay 传入该路径。
```

需要 Git、CMake >=3.26 与 VS2022 C++ 工具。构建只用 CPU，应在正式 GPU 计时前完成；构建器校验已有源码、补丁哈希与构建目录归属，保留不匹配的用户文件并报错。依赖修正的稀疏预设会在暂停服务前验证 overlay 的补丁和 DLL 指纹，防止误用旧 wheel。


## 质量优先后的最终处置

| 同题非思考模式质量测试 | 答对数 | 处置 |
|---|---:|---|
| 发布版，全块注意力 | 16/24 | 作为本轮参考，基线自身也有错误 |
| 发布版，阈值 0.1、b4096 | 1/24 | 淘汰 |
| 修复末尾路径，阈值 0.05、b2048 | 5/24 | 淘汰；原始 96K 首字 83.93 s、1171 tok/s |
| 修复末尾路径，阈值 0.3、b2048 | 第一份 5/12，参考为 8/12 | 停止路线；第二份按用户指示中止，无完成分数 |

修正调度边界并不能消除稀疏正文计算的质量影响。以上只是两份合成文档的诊断结果，不是模型总体准确率；但已经足以否决这些候选。用户没有设定可以接受的具体百分点，不能自行将较大的损失判为可接受。

后续先固定模型、输入、量化、生成参数与完整注意力，验证算子正确性和同题质量，再比较速度。明显、可复现的质量退化直接淘汰；小差异增加样本并逐项检查，在证据足够前不切换默认服务。研究重点回到完整注意力内核、矩阵计算、数据搬运与调度的实际耗时，不再通过降低稀疏阈值追求数字。

[修复版的原始证据与处置](../results/2026-10-25-qwen38-openvino-dense-tail/metadata.json)保留失败结果和中止原因。所有稀疏预设标为归档、不推荐；独立库仅用于复现。原 SYCL/MTP 日常服务保持 `0.0.0.0:5001`。
