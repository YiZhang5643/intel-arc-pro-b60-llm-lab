# Intel Arc Pro B60 24GB Local LLM Lab


Milestone: [Qwen3.8-27B with all attention blocks retained](docs/2026-10-25-stage-summary.md): **982 tok/s at 32K (33.37 s to first text)** and **646 tok/s at 96K (152.19 s)**. Sparse candidates with substantial quality regressions were rejected. The quality-constrained 1200 tok/s target remains open.
[中文](README.md) · [Windows setup](docs/setup-windows.md) · [Measurements](docs/nemotron.md) · [Methodology](docs/methodology.md) · [Tools](docs/tools.md)

New: [Qwen3.8-27B mixed-IQ tuning](docs/qwen38.md), including 32K/96K inputs, embedded MTP and independent draft microbatch sizing. New: [operator timelines and traces](docs/qwen38-profile.md). Follow-up [kernel experiments](docs/qwen38-kernel-experiments.md): MTP stays on; blocked GDN reduces mean cold prefill to 53.71 s at 32K and 208.17 s at 96K (two timings per arm).

Reproducible community experiments on a single B60: long prompt prefill, time to first content, prefix reuse and MTP. First model: Nemotron 3.5 Lightning 30B-A3B Q4_0, optional Q8_0 MTP. More models can be added using the contribution template. This is not an intelligence leaderboard.

Windows 11, i5-12600K, 32GB DDR4, B60 24GB, driver 32.0.101.9033. Measured October 24, 2026.

| Cold prompt | Official SYCL baseline | Tuned cold-prefill profile |
|---|---:|---:|
| 32,768 tokens | 482 prompt tok/s; 69.38 s total | 1,454 prompt tok/s; 22.61 s TTFT; 24.08 s total |
| 98,304 tokens | 455 prompt tok/s; 218.00 s total | 1,184 prompt tok/s; 83.30 s TTFT; 85.28 s total |

These are **whole-profile comparisons**, not isolated patch effects: FP16/oneDNN, local Q4 expert reorder, batch/ubatch and MTP settings differ. Retrieval found all three needles in both profiles. One run per length; no confidence intervals.

A fair 96K MTP comparison uses identical token arrays and batch/ubatch 2048 on the same optimized build. MTP improved cached retrieval generation by 45%, code by 3%, but reduced English prose generation throughput by 20%. Cold TTFT increased from 93.24 to 104.65 seconds. Prose lengths differed, so total durations are not a fixed-output comparison.

Two explicit profiles are included: MTP on for long conversations; MTP off with batch/ubatch 8192 for processing fresh long documents. Cached continuation TTFT was below one second in the measured 96K conversation. MTP is workload-dependent, not universally beneficial at long context.

Python 3.10+; benchmark scripts need no Python packages; optional plots need Matplotlib. Download runtime and weights separately following the setup guide.

```powershell
python scripts/serve.py --preset presets/nemotron-long-chat-mtp.json --runtime-dir D:/llm/runtime --model-dir D:/llm/models --host 0.0.0.0 --port 5001
python scripts/bench_chat.py --label mtp-on --output runs/mtp-on --preset presets/nemotron-long-chat-mtp.json
```

The launcher defaults to 127.0.0.1:5001. Explicitly use 0.0.0.0 for LAN access. It runs in the foreground and does not change firewall settings or stop other services.

Historical measurements, sanitized configurations and response samples are under `results/`. Scripts rebuild prompts with the current server tokenizer and save input hashes. For MTP A/B tests, compare those hashes. Numerical Q4 tests passed 106/106; small model smoke checks are not broad capability evaluations. A patched Intel Vulkan configuration was tested on the current Qwen weights and was slower than SYCL; stock Vulkan was not retested. Qwen operator timelines identify the main costs but do not establish a hardware ceiling.

Original code: MIT. External models, binaries and patch implementations retain their own terms; see [attribution](THIRD_PARTY_NOTICES.md).

New: [OpenVINO full-attention baseline and rejected sparse prefill experiments](docs/qwen38-openvino.md), including complete cold 96K requests, runtime/model pins, and strict paired long-document quality checks. Experimental presets leave the daily SYCL/MTP service unchanged.

Sparse threshold tuning was stopped after substantial paired quality regressions. Performance gains must pass quality validation before adoption; no sparse preset replaced the daily service.
