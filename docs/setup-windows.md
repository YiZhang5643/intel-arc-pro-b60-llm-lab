# Windows 复现

所有下载入口见 [工具表](tools.md)。不要把不同 commit 的 ggml DLL 混用。以下用 `D:/llm` 举例，自行替换为工作目录；不会修改本机已有部署。

## 1. 下载与核验

从官方 b11869 发布页下载 `llama-b11869-bin-win-sycl-x64.zip`，从模型页下载以下两项：

- `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf`（19,063,900,128 bytes）
- `mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf`（2,175,713,312 bytes，仅 MTP 需要）

具体 URL、尺寸和 SHA256 在 [downloads.json](../manifests/downloads.json)。HF 页的 metadata revision 曾无法直接 resolve，因此这里使用 main 下载地址并以完整 SHA256 锁定内容。若哈希变化，不应当作相同模型直接比较。

```powershell
# 按下载位置分别验证；--name 可以重复
python scripts/verify_files.py --directory D:/llm/downloads --name llama-b11869-bin-win-sycl-x64.zip
python scripts/verify_files.py --directory D:/llm/models --name NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_0.gguf --name mtp-NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf
Expand-Archive D:/llm/downloads/llama-b11869-bin-win-sycl-x64.zip D:/llm/runtime-official
```

保留官方运行库作基线。确认安装 Intel B60 驱动与所需 SYCL/oneAPI 运行库。先执行 `llama-server.exe --list-devices`，确保 SYCL0/Level Zero 0 对应 B60；核显存在时编号不一定相同。启动器的 `--device` 与 `--device-selector` 可分别指定。

## 2. 编译优化 SYCL DLL

需要 Visual Studio 2022 C++ x64 环境、oneAPI DPC++ 2026.1.1、oneDNN 2026.0.2、Level Zero SDK 1.32.0、CMake 和 Ninja。先在开发者终端调用 VS `vcvarsall.bat x64` 与 oneAPI `setvars.bat`，再进入 PowerShell；设置 `DNNLROOT` 和 `LEVEL_ZERO_V1_SDK_PATH` 为实际安装目录。若 `cl`、`icx`、`ninja` 不在 PATH，先完成环境初始化。版本不同必须记录。

```powershell
git clone https://github.com/ggml-org/llama.cpp D:/llm/source
git -C D:/llm/source checkout d327bf1f0780f87b8c01b367b61853b0c291d7c4
# 从工具表中的补丁目录另行下载原始 0001 Q4_0/Q8 patch，保存为下面的文件
python scripts/adapt_q4_patch.py --source-dir D:/llm/source --patch-file D:/llm/downloads/sycl-moe-q4.patch --output-patch D:/llm/downloads/adapted-q4.patch
./scripts/build-sycl.ps1 -SourceDir D:/llm/source -BuildDir D:/llm/build
```

适配器锁定原始补丁 SHA256 `5150ad8b8655ae80cd3f921a17605a5586fc8d976f43f3ce555ef92568885b87`。它只提取 Q4_0 分支；b11869 已有 Q8_0 分支。补丁实现不随本仓库分发，来源与许可范围见 [说明](../THIRD_PARTY_NOTICES.md)。适配失败时不要绕过哈希与 anchor 检查。

复制完整官方运行库到 `D:/llm/runtime-optimized`，然后以本次编译的 `build/bin/ggml-sycl.dll` 替换其中同名文件，并复制 `test-backend-ops.exe`；保留官方 CPU backend，供正确性对照。新构建若额外依赖 oneAPI DLL，使用对应版本运行库或已初始化的 oneAPI PATH。不要以网上来历不明的 DLL 补缺。

```powershell
Copy-Item D:/llm/runtime-official D:/llm/runtime-optimized -Recurse
Copy-Item D:/llm/build/bin/ggml-sycl.dll D:/llm/runtime-optimized/ggml-sycl.dll -Force
Copy-Item D:/llm/build/bin/test-backend-ops.exe D:/llm/runtime-optimized/test-backend-ops.exe -Force
$env:ONEAPI_DEVICE_SELECTOR='level_zero:0'
$env:GGML_SYCL_ENABLE_HOST_PINNED_MEM='0'
$env:GGML_SYCL_MOE_Q4_REORDER='1'
Push-Location D:/llm/runtime-optimized
./test-backend-ops.exe -b SYCL0 -o MUL_MAT_ID -p 'type_a=q4_0'
Pop-Location
```

本轮正式 DLL 指纹 `c12dab95d5db6414dd7856dd150b720569656589cea825756afcdd2295456b7b`，数值检查 106/106。编译不可保证逐字节一致；仍应记录自己的哈希并运行检查。本仓库没有分发优化 DLL，公共脚本本轮仅做离线验证，没有重新跑 GPU 编译/基准；历史测试证据另附。

## 3. 启动和比较

```powershell
# 默认使用长对话 MTP；trim 是本轮采用过的一次性加载后主存整理
python scripts/serve.py --preset presets/nemotron-long-chat-mtp.json --runtime-dir D:/llm/runtime-optimized --model-dir D:/llm/models --host 0.0.0.0 --port 5001 --trim-working-set
```

Ctrl+C 结束自己启动的前台进程。新长文档优先时，将 preset 改为 `nemotron-cold-prefill.json`。严格 MTP A/B 时，关闭组使用 `nemotron-mtp-off-comparison.json`，两边 batch/ubatch 都是 2048；不要用 8192 配置代替关闭组。

另开终端，对两个组分别使用新目录：

```powershell
python scripts/bench_chat.py --label mtp-on --output runs/mtp-on --preset presets/nemotron-long-chat-mtp.json
# 手工重启为同 batch 的 MTP off 配置，然后运行：
python scripts/bench_chat.py --label mtp-off --output runs/mtp-off --preset presets/nemotron-mtp-off-comparison.json
python -c "import json; from pathlib import Path; a=json.loads(Path('runs/mtp-on/prompt-hashes.json').read_text()); b=json.loads(Path('runs/mtp-off/prompt-hashes.json').read_text()); assert a==b; print('identical inputs')"
python scripts/bench_prefill.py --label prefill --output runs/prefill --sizes 12288,32768,98304 --repeats 3
python scripts/bench_context.py --label retrieval --output runs/retrieval --sizes 32768,65536,98304 --stream
python scripts/bench_short.py --label short --output runs/short
python scripts/bench_smoke.py --label smoke --output runs/smoke
```

缓存对照要求单客户端、单槽，不能同时跑其他请求。若模型/模板不支持缓存断言或 MTP，记录失败；不要删除断言后报告成成功。`bench_smoke.py --execute-generated-code` 的执行范围见 [测试方法](methodology.md)。

## 4. 项目自检（无需 GPU）

```powershell
python -m unittest discover -s tests -v
python scripts/check_evidence.py
```

Mock HTTP 测试检查流式 TTFT、冷输入断言、哈希一致性和拒绝覆盖；它不验证 GPU 性能。
