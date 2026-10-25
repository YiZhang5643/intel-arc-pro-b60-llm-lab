# Qwen3.8：从时间线到内核改进

2026-10-25，Windows / Arc Pro B60 24GB / 原有混合 IQ 权重。**已采用分块 GDN：32K 少等约 3.8 秒，96K 少等约 11.1 秒。MTP 保持开启。**

[逐次测量](../results/2026-10-25-qwen38-kernels/gdn-cold.json) · [汇总](../results/2026-10-25-qwen38-kernels/comparison.json) · [配置与指纹](../results/2026-10-25-qwen38-kernels/metadata.json) · [检索输出](../results/2026-10-25-qwen38-kernels/gdn-retrieval.json) · [完整 trace](../results/2026-10-25-qwen38-kernels/gdn-profile-32k/trace.json.gz)

## 已确认的改进

| 冷输入 | 原内核：平均秒 / tok/s | 分块 GDN：平均秒 / tok/s | 节省等待 / 吞吐提升 |
|---|---:|---:|---:|
| 32,768 tokens | 57.49 / 570.0 | **53.71 / 610.1** | **3.78 秒 / 7.0%** |
| 98,304 tokens | 219.24 / 448.4 | **208.17 / 472.2** | **11.07 秒 / 5.3%** |

同一份实验 DLL，只切换 `GGML_SYCL_GDN_BLOCKED=0/1`，按关、开、开、关分别重启，每边每个长度测两次。每次先热身 4096 tokens；正式请求使用此前保存的相同 token 数组，`cache_n=0`，只生成 1 token，表中是服务端预填充时间。客户端总时长也保存在原始记录中；这不是完整回答的耗时，也不是流式首字的独立测量。测量期间未同时编译或运行其他基准。

32K 开启补丁的两次为 53.679 / 53.744 秒；关闭为 57.611 / 57.371 秒。96K 开启为 208.197 / 208.142 秒；关闭为 219.440 / 219.039 秒。样本仍少，不提供统计置信区间。

每次冷请求之后，复用缓存继续检索：**8/8 找对全部三个条目**，输出一致；生成速度基本一致，约 32K 20.9 tok/s、96K 14.3 tok/s。MTP 草稿均 57/57 接受。这组检索任务不代表其他代码、说明或模型能力测试；之前模型输出 JSON 围栏的问题仍然存在。

补丁基于 [Torchit1 的分块 SYCL GDN](https://github.com/Torchit1/llama.cpp/tree/ff6c13e692f06cfc9e55e26b088f74aca48c7e7a)：把多 token 的数据分块预取到片上共享内存，并让一个子组计算多列状态，以减少逐 token 等待。它保留相同递推公式，不改变权重或模型结构。这里适配到当前 b11869，保留现有 cache fusion，增加开关，只用于非 KDA、状态维度 128、批次至少 16 tokens 的路径。

[58/58 算子检查](../results/2026-10-25-qwen38-kernels/gdn-correctness.txt)通过，新增了实际 Qwen 的重复 head、8192-token 批次、尾块、多序列、非连续布局、MTP 历史状态与 cache fusion；短解码继续使用原内核。这些检查和检索不能替代广泛的质量评估。

## 时间线解释了收益

| 32K 内核合计 | 原始时间线 | 分块 GDN 时间线 |
|---|---:|---:|
| Gated DeltaNet | 8.05 s | **4.40 s** |
| 主矩阵乘 | 23.26 s | 23.27 s |
| 注意力 | 8.81 s | 8.81 s |
| 客户端请求 | 58.02 s | **53.98 s** |

[新时间线分析](../results/2026-10-25-qwen38-kernels/gdn-profile-32k/analysis.json)确认执行了新内核。GDN 本身耗时减少约 45%，其他两大项接近；这解释了整模型约 4 秒的改善。新采集相对同配置两次非采集平均请求约慢 0.5%；只有 32K 重新采集了时间线，没有将它当成 96K 的新算子测量。内核时长有重叠的可能，不能当成严格可相加的关键路径。

## 未采用的 SDPA 布局实验

将 oneDNN 的五维 GQA 图去掉最前面的长度 1 维，得到等价四维图；[独立微基准](../results/2026-10-25-qwen38-kernels/sdpa-layout-probe.txt)中输出逐元素一致、无非有限值，但没有实质提速：

| Q / KV 长度 | 五维 / 四维：平均毫秒 |
|---|---:|
| 8192 / 32768 | 199.853 / 199.817 |
| 8192 / 98304 | 635.037 / 633.162 |

大形状各热身后计时 5 次。它比较的是两种布局，不是 CPU 数学参考验证，也不等于整个模型的加速；因此没有修改后端布局。

## arc-llm-lab 的线索

参考项目：[edagher92-coder/arc-llm-lab](https://github.com/edagher92-coder/arc-llm-lab/tree/6d26bb1980854b7dfde77817653f0e10b123818a)。它保存了 Intel 未合并的 [Vulkan FA prefill 补丁](https://github.com/edagher92-coder/arc-llm-lab/blob/6d26bb1980854b7dfde77817653f0e10b123818a/docs/INTEL-FA.md)，能改善其长上下文下的追加输入速度。主模型需 F16 KV，Q8/Q4 KV 不会触发这条快速路径。

对照时必须区分：其 32K 深度的 420 tok/s 是**已有 32K 后追加 2048 tokens**，不是从零处理整段 32K；其高生成速度还涉及独立 DFlash2 草稿和 Q4_K_M 权重，与本机混合 IQ + 内嵌 MTP 的组合不同。因此它的数字不是本机应有速度的直接基线。

本机已校验原始补丁 SHA256，两个补丁均能直接应用到 b11869；固定源码、原作者信息、许可与校验值保存在 `manifests/`。构建使用现有 MSVC 和隔离工具目录，不修改驱动或系统 Vulkan loader。

### 本机 Vulkan 结果：保留实验配置，继续使用 SYCL

[计时、配置、输出与显存样本](../results/2026-10-25-qwen38-vulkan-intelfa/metadata.json)。明确选择 `Vulkan1 = Arc Pro B60`，`Vulkan0` 是本机 UHD 770；日志确认 66/66 层卸载，并在较大批量实验中打印 **Intel FA prefill active**。增加的第三个本地补丁只在热身时记录首次进入快速路径，不改变计算。

| 配置 | 32K 冷预填充 | 96K 冷预填充 | 缓存检索生成：32K / 96K |
|---|---:|---:|---:|
| 已采用 SYCL + 分块 GDN，Q8 KV，ubatch8192 | **53.71 s** | **208.17 s** | 约 20.9 / 14.3 tok/s |
| Intel Vulkan，F16 KV，ubatch256 | 96.57 s | 未完成，主动停止 | 15.34 / 未测 |
| Intel Vulkan，F16 KV，ubatch4096 | 90.45 s | 385.40 s | 16.75 / 14.89 tok/s |

Vulkan 每个完成的长度各一次；较大批量原计划重复两轮，第一轮完整 32K/96K 均明显落后后停止第二轮，不把中断请求计入结果。三次完成的缓存检索均找对全部条目，MTP 草稿均 57/57 接受；不是广泛质量测试。SYCL 数字为两次平均，Vulkan 为一次，样本量不同。

本机混合 IQ 权重比参考项目的 Q4_K_M 更小，因此也尝试了更大的 ubatch4096。专用显存采样最大观察值约 19.45 / 21.60 GiB，共享占用约 386 / 631 MiB；采样只覆盖部分窗口，不能证明瞬时峰值或 WDDM 页迁移情况。32K 增大批量略有改善，仍未追上 SYCL；96K 的一次完整测量也不支持替换默认。

这是整套配置对比，没有重新跑未打补丁的同版 Vulkan，也没有采集 Vulkan 算子时间线。因此这里只确认当前权重和配置的选择，不能据此判定补丁本身的收益、其他量化的表现，或未打补丁的上游是否已经解决历史退化。继续分析 Vulkan 应先分离混合 IQ 矩阵路径、注意力与状态检查点等成本，不能只凭显存占用认定瓶颈。

当前服务已恢复到 SYCL + 分块 GDN。Vulkan 独立运行库和实验预设保留，供后续诊断或换量化复测。


## 使用与回退

当前默认使用 `runtime-qwen-gdn-blocked` 与 [推荐预设](../presets/qwen38-recommended.json)。`Start-Qwen.cmd` 和兼容入口都由预设选运行库；MTP3、Q8 KV、主批量8192、草稿批量512、context114688、监听 `0.0.0.0:5001` 保持原设置。旧 `runtime-patched-f16` 的 DLL 原样保留。

```powershell
# 当前默认
python scripts/manage_server.py restart --deployment-dir C:/MyLLM/nemotron-sycl --preset presets/qwen38-recommended.json

# 回退到本轮前的内核与配置
python scripts/manage_server.py restart --deployment-dir C:/MyLLM/nemotron-sycl --preset presets/qwen38-baseline.json --runtime runtime-patched-f16
```

## 复现工具

```powershell
# 服务已运行时，只生成固定输入，不执行推理
python scripts/bench_chat.py --label input32 --output runs/input32 --size 32768 --prepare-only
python scripts/bench_chat.py --label input96 --output runs/input96 --size 98304 --prepare-only

# 先在已初始化 VS x64 / oneAPI 的终端构建；源码按 Windows 安装文档固定到 b11869
$lab = (Get-Location).Path
$src = 'C:/MyLLM/nemotron-sycl/source/llama.cpp-b11869'
git -C $src apply --check "$lab/manifests/qwen38-gdn-blocked.patch"
git -C $src apply "$lab/manifests/qwen38-gdn-blocked.patch"
git -C $src apply "$lab/manifests/qwen38-gdn-tests.patch"
powershell -File scripts/build-sycl.ps1 -SourceDir $src -BuildDir C:/MyLLM/nemotron-sycl/build-gdn-reproduction
# 将新 ggml-sycl.dll 和 test-backend-ops.exe 放到匹配 b11869 的独立运行库中，再做算子检查

# 四轮对照；实验脚本会暂停并恢复记录中的服务
python scripts/bench_gdn_blocked.py --deployment-dir C:/MyLLM/nemotron-sycl --runtime runtime-qwen-gdn-blocked --prompts runs/input32/prompts.json --prompts runs/input96/prompts.json --output runs/gdn-compare

# 独立 SDPA 探针：脚本中 oneAPI 路径按本机调整
scripts\build_sdpa_probe.cmd

# 从固定提交与校验值准备 Vulkan 源码；必须使用新目录
python scripts/prepare_vulkan_intelfa.py --root C:/MyLLM/nemotron-sycl/toolchain/vulkan-reproduction
scripts\build_vulkan_intelfa.cmd C:\MyLLM\nemotron-sycl\toolchain\vulkan-reproduction

# 将构建出的 bin/*.exe 与 *.dll 放进独立运行库后，先核对设备编号
C:\MyLLM\nemotron-sycl\runtime-vulkan-intelfa\llama-server.exe --list-devices
python scripts/bench_candidate.py --deployment-dir C:/MyLLM/nemotron-sycl --runtime runtime-vulkan-intelfa --preset presets/qwen38-vulkan-intelfa-experimental.json --device Vulkan1 --prompts runs/input32/prompts.json --prompts runs/input96/prompts.json --repeats 1 --output runs/vulkan-compare
```

GDN 补丁、源码指纹与新增测试分别为 [实现补丁](../manifests/qwen38-gdn-blocked.patch)、[指纹](../manifests/qwen38-gdn-blocked.json)、[测试补丁](../manifests/qwen38-gdn-tests.patch)。构建参数与此前相同：`GGML_SYCL=ON`、`GGML_SYCL_F16=ON`、`GGML_SYCL_DNN=ON`，先在单独运行库中完成 `test-backend-ops test -b SYCL0 -o GATED_DELTA_NET,GATED_DELTA_NET_CACHE_FUSION`，再测模型。测试时 `GGML_SYCL_GDN_BLOCKED=1`；旧后端会忽略此开关，不能据此声称已启用新内核。

独立运行库的依赖准备见 [Windows 安装步骤](setup-windows.md)。已有补丁的源码应另取干净副本，不要重复应用。

Intel Vulkan 原始补丁见 [保留的校验值](../manifests/intel-vulkan-fa/SHA256SUMS)；工具链源码版本见 [固定清单](../manifests/vulkan-toolchain-sources.json)，第三方许可与作者见 [来源说明](../THIRD_PARTY_NOTICES.md)。所有新证据均有 SHA256 清单，未改写历史结果。

实验切换还暴露了 Windows 进程退出竞态：查询正在终止的进程时可能先返回拒绝访问，稍后进程句柄才变为已退出。管理器现在只读等待退出，仍然只终止已核实路径的进程；实验脚本也在停止操作报错时尝试恢复原服务。新增的原生子进程回归检查覆盖了这一情形。
