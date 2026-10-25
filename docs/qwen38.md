# Qwen3.8-27B · B60 24GB 调优

测试日期：2026-10-25。[逐项测量 CSV](../results/2026-10-25-qwen38/measurements.csv) · [权重、硬件与配置指纹](../results/2026-10-25-qwen38/metadata.json)

实际文件为 **Huihui / Swift-1.5 / GSQ-RCO 的混合量化衍生版**，不是原版 Qwen 的单一 IQ3_S 量化。保留用户原文件，未转换、重下载或替换权重。它是约 27B 的稠密模型；此前 Nemotron 是每 token 仅激活部分参数的 MoE，总参数接近并不意味着生成成本相近。

## 最后采用的配置

- 主模型与内置 MTP 共用同一 GGUF，`--spec-type draft-mtp`，不指定 Nemotron drafter。
- context 114,688；batch / ubatch **8192 / 8192**；Q8_0 K/V；MTP n-max **3**、p-min **0.6**。
- 关键参数：**`--spec-draft-n-ubatch 512`**，独立缩小草稿工作缓冲。
- SYCL 图执行关闭，oneDNN 注意力保留。选择 B60 `SYCL0 / level_zero:0`，日志报告 66/66 层卸载至 GPU。
- 正式服务监听 **`0.0.0.0:5001`**；恢复用户原本已启用的 MCP proxy。

## 用户会等多久

32K 各组使用完全相同的输入 token 数组，非热身、缓存数经过断言检查。每场景一次测量。

| 32K 场景 | 原脚本：MTP 关闭 | 当前默认：MTP3 + 独立草稿批量 |
|---|---:|---:|
| 冷输入首字 | 57.54 s | 57.36 s |
| 缓存后检索生成 | 7.33 tok/s | 20.80 tok/s |
| 缓存后代码生成 | 7.31 tok/s | 16.16 tok/s |
| 缓存后英文说明生成 | 7.31 tok/s | 13.37 tok/s |
| 缓存后检索请求总时长 | 11.38 s | 5.06 s |

**默认方案保留了原来的首字等待水平，同时明显加快生成。** 代码和英文说明都截断到 128 tokens，只用于计时，不能当成完整答案质量测试。

96K 当前默认配置也通过了冷输入与缓存检索的 3/3 条目检查：

| 96K 场景 | 首字 s | 总时长 s | 生成 tok/s | 缓存 tokens |
|---|---:|---:|---:|---:|
| cold-retrieval | 219.11 | 224.45 | 14.26 | 0 |
| cached-retrieval | 1.96 | 7.02 | 14.07 | 98,300 |
| cached-code | 2.11 | 12.89 | 11.79 | 98,300 |
| cached-prose | 2.08 | 16.87 | 8.59 | 98,300 |

首次处理整篇 96K 文档仍需约 3.7 分钟；命中缓存的后续提问，首字约 2 秒。**本次没有把冷预填充变成几秒钟**，也未证明当前速度已到硬件极限。原配置未测 96K，因此不声称当前默认相对它有多少 96K 提速。

## 哪些尝试不适合当默认

| 尝试 | 观察与取舍 |
|---|---|
| 直接沿用大批量并开启 MTP | 主、草稿微批量都继承 8192，目标工作区约 4065 MiB、草稿约 3072 MiB；记录到约 23.10 GiB 专用显存和 508 MiB 共享占用。12K 请求前 4096 tokens 就用了 63.88 秒，停止该退化实验，未伪装成完整 12K 结果。 |
| 主、草稿 ubatch 都改 1024 | 12K 首字约 30–31 秒，原脚本约 21 秒；MTP 加快生成，但小主批量拖慢预填充。 |
| 主 8192 / 草稿 512 | 草稿工作区降至约 192 MiB，12K 首字恢复约 22 秒，专用显存观察值约 20.55 GiB；最终采用。 |
| 草稿长度 7 | 在同样独立草稿批量下，没有超过草稿长度 3；未采用。 |
| SYCL graph | 12K 对照基本无收益，部分图反复启停；未采用。 |
| F16 KV、主 ubatch4096 | 32K 首字约 64 秒；96K oneDNN 约 245 秒、原生注意力约 259 秒，均慢于默认 Q8 方案的约 219 秒。部分生成任务稍快，但占用更多显存；保留数据作为对照。 |
| 关闭 oneDNN 注意力 | 在 F16 KV 上，32K 生成约快 8%–9%；96K 冷输入更慢、英文说明也更慢，未设为通用默认。 |

显存数字是单点观察，不是峰值采样或页面迁移跟踪。大工作区与退化同时出现，缩小草稿工作区后恢复；没有据此断言某个 WDDM 机制或单个算子就是最终瓶颈。

## 使用与复现

本机原来的 `Start-Qwen.cmd` 和 `qwen26b_server.py` 已转发到仓库中的统一实现，原文件留有 `.before-tuning-时间戳` 备份。名字中的 26b 仅为兼容旧入口，实际加载 27B 文件。默认配置位于 [qwen38-recommended.json](../presets/qwen38-recommended.json)。后续改这个仓库并本地提交，阶段完成再发布。

```powershell
# 仓库中启动；默认部署目录是同级的 nemotron-sycl
./Start-Qwen.cmd

# 其他目录布局请明确指定；runtime 参数是部署目录内的运行库文件夹名
python scripts/manage_server.py restart --deployment-dir D:/llm --runtime runtime-optimized --preset presets/qwen38-recommended.json

# 模型检查与文件校验（SHA256 会完整读取约 12.1GB 文件）
python scripts/inspect_gguf.py D:/llm/models/Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_S-mtp.gguf
python scripts/verify_files.py --directory D:/llm/models --name Huihui-Qwen3.8-27B-abliterated-Swift-1.5-GSQ-RCO-IQ3_S-mtp.gguf

# 复现长上下文；用新的输出目录，先热身，单槽、单客户端
python scripts/bench_chat.py --label qwen32k --output runs/qwen32k --preset presets/qwen38-recommended.json --size 32768 --generation-tokens 128
python scripts/bench_chat.py --label qwen96k --output runs/qwen96k --preset presets/qwen38-recommended.json --size 98304 --generation-tokens 128
python scripts/bench_probe.py --label probe --output runs/probe --preset presets/qwen38-recommended.json

# 代码与数据检查
python -m unittest discover -s tests -v
python scripts/check_evidence.py
python scripts/check_evidence.py --directory results/2026-10-25-qwen38
```

`qwen38-cold-prefill.json` 保留 MTP 关闭的原始配置作为可选对照，不保证请求总时长更短。切换配置会重启服务并清空推理缓存，应在会话开始前选择。

## 正确性与局限

- 所有已完成 32K/96K 长对话组，冷检索和缓存检索均为 3/3；输入哈希已逐组核对。
- 正式默认配置的 `compress_ranges` 通过 8/8 功能样例。两道数值题的值正确，但模型加了 Markdown 代码围栏，**两项严格纯 JSON 格式检查失败**，原始失败结果与解释一并保留，未改评分规则让它通过。这些小题不是全面能力评测。
- 每配置/场景一次测量，没有随机顺序、多轮分布、温度/功耗控制；百分比是观察值。没有测另一份量化、Vulkan 或 Linux/XPU 框架。
- 这份 GGUF 混合 IQ1/IQ2/IQ3/IQ4/K-quants；更小的文件不保证同后端更快。继续优化冷输入，应做算子时间线及同模型不同量化/后端的同输入对照，不能把“Intel 优化版”模型卡上的其他环境数字当成本机结果。
- 测量使用直接推理请求，MCP 工具自身耗时不计入吞吐与首字时间；正式部署恢复了原来的代理能力。

来源：[用户所用 GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF)、[上游 IQ3 优化 b11636](https://github.com/ggml-org/llama.cpp/releases/tag/b11636)、[IQ3 多列 MMVQ](https://github.com/ggml-org/llama.cpp/pull/29500)。这些 IQ3 优化已在本次使用的 b11869 源码中，未重复打补丁。
