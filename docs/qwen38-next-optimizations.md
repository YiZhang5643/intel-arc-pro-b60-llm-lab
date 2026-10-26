# 下一阶段：保留质量的预填充优化

2026-10-25。阶段基线已保存；本轮只查资料和源码，没有运行新 GPU 实验或替换日常服务。下表是研究顺序，不是已获得的提速。

## 目标需要省下多少时间

| 同一完整冷输入 | 全块基线首字等待 | 1200 tok/s 对应等待 | 还需减少 |
|---|---:|---:|---:|
| 32,767 tokens | 33.37 s，981.95 tok/s | 27.31 s | 6.06 s，约 18.2% |
| 98,304 tokens | 152.19 s，645.91 tok/s | 81.92 s | 70.27 s，约 46.2% |

来源：[原始计时](../results/2026-10-25-qwen38-openvino-ir/openvino-off-xattn1-dq128/summary.json)。每项目前一次测量。96K 仍是主要研究场景，不能把 32K 的 982 外推过去。

若只优化一个占比为 f 的部分，使其快 s 倍，总耗时变为 `T × (1-f+f/s)`。例如把一个算子快一倍，要单独完成 96K 目标，它在当前请求中必须占约 92.3% 的耗时。**这只是条件预算；尚没有本路径的算子占比，不能承诺达到目标。**

## 先取得这条路径自己的时间线

旧 [SYCL 时间线](qwen38-profile.md)不能当成 OpenVINO 时间线；旧 unitrace 构建明确关闭了 OpenCL。先保持同一官方 IR、完整注意力、U8 KV、DQ128、b2048、MTP 关闭，区分以下耗时：

- 矩阵乘与权重/激活转换。
- 完整注意力：按输入块的已累计上下文长度拆分，观察后半段增长。
- GatedDeltaNet、KV 更新、重排与其他融合算子。
- 主机调度、同步和 GPU 空闲间隔。

可实施的两个采集入口：

1. OpenVINO 官方 [InferRequest.get_profiling_info](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.InferRequest.html#get-profiling-info) 和 [ProfilingInfo](https://docs.openvino.ai/2026/api/ie_python_api/_autosummary/openvino.ProfilingInfo.html)。在隔离 GenAI 构建中启用 profiling，并在每次 `m_request.infer()` 后采集节点名、类型、执行类型和时间；逐块保存，不只取生成结束的最后一次。源码入口：[model_runner.hpp](https://github.com/openvinotoolkit/openvino.genai/blob/ab01d1771e4adaa0e1e08e9d630def170572506c/src/cpp/src/continuous_batching/model_runner.hpp)；编译属性入口是同目录的 `pipeline_impl.cpp`。这是待实现的采集补丁，当前 Python 基准没有直接取得这个内部 InferRequest。
2. 在独立目录重建支持 OpenCL 的 [unitrace](https://github.com/intel/pti-gpu/tree/307d457292976823546d3b30b141402de44f724f/tools/unitrace)，核对实际提交 API，采集内核、拷贝与主机事件。现有构建保留供 SYCL 使用。逐层时间可能包含融合或重叠，不能直接把累加值当成请求墙钟时间。

采集有开销；使用同输入的采集/非采集对照。正式速度只报告关闭采集后的多次结果，记录中位数和范围。第一次采集先 32K，确认分类与开销可解释后再做 96K。

## 根据时间占比选择实现优化

| 优先方向 | 具体检查 | 质量与执行边界 |
|---|---|---|
| 完整注意力 CM 内核 | 工作组覆盖的序列长度、寄存器压力、KV 访问复用、softmax 归约与融合 | 保留全部因果注意力块；不降低阈值、不丢 token |
| GDN 与图融合 | 实际是否进入融合 GDN，短算子/重排是否过多，是否存在可消除的中间读写 | 保持原模型递推、状态与回滚语义 |
| 矩阵与数据转换 | 仅在占比足够时调整 shape/分组/融合；核对真实激活误差 | 不把合成 INT8 GEMM 探针等同于无损模型加速 |
| 上游新实现 | 隔离比较已固定版本与候选提交；保留可回退运行库 | 先验证 Windows/B60/本 IR 兼容，再测质量和速度 |

已核对发布版 [CM paged attention 生成器](https://github.com/openvinotoolkit/openvino/blob/e213a14725769a6a2bf86fc86650aa78a9c05d3e/src/plugins/intel_gpu/src/graph/impls/cm/paged_attention_gen.cpp)：`bypass_xattn` 使用阈值 `>=1.0` 旁路块选择；多 token 路径由 `CMPA_WG_SEQ_LEN` 等 JIT 常量控制，并按 KV 是否压缩选择 tree softmax。它们是源码研究入口，不是已经验证可调、可提速的发布版参数。`TILE_Q` / `OV_GPU_PA_ONLINE_TILE_STEPS` 出现在 small-Q 推测生成路径，不能当作长输入预填充开关。

已有全块筛选中，b4096 约 918 tok/s、DQ1024 约 976，均未超过 b2048 / DQ128 的约 982。继续盲目扩大 batch 或量化分组优先级低。

## 新进展怎么用

- [edagher92-coder/arc-llm-lab](https://github.com/edagher92-coder/arc-llm-lab)：当前 README 报告单 B60、Windows、Intel FA + F16 KV 的 32K 预填充约 420 tok/s；DFlash2 的主要收益是生成。值得参考其质量评测、显存余量和可复现补丁做法；这些数字不证明会超过本机 OpenVINO 全块基线。不同权重、上下文、后端不能作严格排名。
- [B70 OpenVINO 调优记录](https://github.com/SergiioB/intel-arc-pro-b70-inference-cookbook/blob/master/docs/qwen38-27b/OPENVINO-B70-TUNING.md)：B70、2026.5 nightly、INT8 VL split 的结果可提供排查线索，但不是 B60 INT4 结果。其 main U8 / draft F16 KV 方案值得后续单独验证 MTP；不能用于宣称冷预填充已加速。
- [官方发布列表](https://github.com/openvinotoolkit/openvino/releases)本次查询最新稳定版仍是 2026.4.1，正是阶段基线版本。nightly 只能作为隔离候选，不直接覆盖已验证环境。

## 每项优化的采用条件

固定模型 revision、完整输入文本与 token 检查、模板、精度和生成设置；单次只变更一个可解释因素。先跑现有多跳关联、修订冲突、顺序计算配对用例，再增加长文问答/真实任务。完整注意力也会因浮点归约顺序改变答案，因此仍须回归验证。明显、可复现的退化直接淘汰；小差异增加样本，不自行规定用户可接受的损失。

全块参考的 16/24 是小型非思考模式诊断分数，不是模型总体准确率。更换检查点、开启思考、改模板或采用更低精度都另列实验。MTP 后续独立评估生成速度与质量，不能把它算作冷预填充优化。重复文档的前缀复用可以另报交互延迟收益，但不计入完整冷输入 1200 tok/s 目标。
