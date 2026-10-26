# 阶段总结：Qwen3.8-27B 保留全部注意力块，32K 达到 982 tok/s

Windows 11、单张 Intel Arc Pro B60 24GB，使用官方 INT4 IR 和 OpenVINO GenAI。保留全部注意力块的预填充路径已经得到可复现的阶段基线；没有采用质量明显退化的稀疏候选。

## 已测结果

| 完整冷输入 | 首个非空文本等待 | 输入 tokens / 等待时间 |
|---|---:|---:|
| 32K：实际 32,767 tokens | **33.37 s** | **981.95 tok/s** |
| 96K：98,304 tokens | **152.19 s** | **645.91 tok/s** |

关闭前缀缓存，每次处理完整文档。速度包含分词、调度、预填充和首 token 工作；加载与预热另记。每个长度目前一次测量，没有置信区间。**982 是 32K 的结果，不能当作 96K 结果。**

同一 IR 的 32K 筛选中，普通注意力、b2048、DQ128 为约 690 tok/s；全块 CM 路径为约 982 tok/s，提高约 42%。这是注意力实现与缓存布局一起变化的配置对照，不能全部归因于一个算子。

## 复现配置

- 模型：[OpenVINO/Qwen3.8-27B-int4-ov](https://huggingface.co/OpenVINO/Qwen3.8-27B-int4-ov/tree/7b20ea8c7df7223a85ff27bab5c473f0cb038943)，固定提交 `7b20ea8c7df7223a85ff27bab5c473f0cb038943`。
- 运行库：OpenVINO 2026.4.1、GenAI 2026.4.1.0；驱动 32.0.101.9033。
- 设备：`GPU.1` 是 B60；`GPU.0` 是本机 UHD 770。
- b2048、动态量化分组 128、U8 KV、400 个 256-token KV 块，总容量 102,400 tokens。
- XAttention 接口的阈值 **1.0**，保留全部注意力块并进入 GPU 全块旁路；MTP 关闭、前缀缓存关闭。内核与量化舍入仍可能变化，不宣称逐位等同于其他后端。

[固定预设](../presets/openvino/qwen38-full-blocks.json) · [下载与完整命令](qwen38-openvino.md#质量检查和复现) · [隔离基准运行器](../scripts/run_openvino_bench.py) · [原始计时](../results/2026-10-25-qwen38-openvino-ir/openvino-off-xattn1-dq128/summary.json) · [环境与限制](../results/2026-10-25-qwen38-openvino-ir/metadata.json)

该官方模型与原 Swift/Huihui 混合 IQ GGUF 不是同一个检查点，因此跨框架对比属于完整部署方案对比。模型、安装的运行库和编译产物均不入库。

## 质量验证决定了哪些结果可以采用

96K 基础检索找对 3/3 个代码。更严格的两份非思考模式文档含三跳关联、修订冲突和顺序库存计算，共 24 个字段；全块参考答对 16/24，说明参考本身也存在能力限制，不能称为模型总体准确率。

降低稀疏阈值后，候选分别降到 1/24、5/24；修复末尾调度后仍存在明显退化。因此这条阈值调优路线停止，相关预设归档、不推荐。修复源码和失败记录保留用于复现，不作为速度推荐。

[质量生成脚本](../scripts/prepare_openvino_quality.py) · [严格评分](../scripts/score_long_answer.py) · [同输入比较](../scripts/compare_openvino_quality.py) · [逐项退化记录](../results/2026-10-25-qwen38-openvino-dense-tail/quality-fixed005-comparison.json)

## 下一阶段的约束

先保持同一模型、完整注意力、相同输入和生成参数，定位矩阵计算、完整注意力、GDN 与数据搬运耗时，再做实现优化。明显、可复现的质量退化直接淘汰；小差异增加样本复核，不自行指定用户可以接受的损失。

**1200 tok/s 仍是研究目标，目前未得到质量可接受的达标配置。** 本阶段只是记录可复现的性能基线；日常 SYCL/MTP 服务继续监听 `0.0.0.0:5001`，没有被实验配置替换。
