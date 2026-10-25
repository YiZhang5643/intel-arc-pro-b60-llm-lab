# 本地开发与分阶段发布

本地仓库：`intel-arc-pro-b60-llm-lab`。日常以这里的文件为准，完成一项连贯修改后提交到本地 Git；积累到一个可分享的阶段，再按用户的发布指令推送到 GitHub。默认不自动推送，也不通过网页或 connector 把未发布的本地改动提前公开。

## 日常修改

1. 在本仓库内修改代码、配置、文档和测试，检查差异，不混入无关改动。
2. 运行与修改相符的检查；纯文档修改通常只需校对和差异检查。
3. 明确选择相关文件提交，不把模型、运行库、构建产物或私密信息加入 Git。

```powershell
git status --short
git diff
git diff --check
git add <本次修改的文件>
git commit -m "Explain the completed change"
```

`commit` 只保存在本地，不会更新 GitHub。可以用下面的命令查看尚未发布的提交：

```powershell
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
```

## 阶段发布

适合发布的阶段包括：完成一个模型的可复现测试、确认一项调优结论，或完成一批文档与工具改进。阶段完成后，用户要求发布时再执行下面的流程：

```powershell
git fetch origin
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main..HEAD
python -m unittest discover -s tests -v
python scripts/check_evidence.py
git push origin main
```

推送前确认远端是否有新提交并处理分歧，不强制覆盖远端；检查通过后发布选定阶段。推送后核对远端 commit，报告本次公开的内容。GitHub 登录和推送凭据与本地提交相互独立，尚未登录不妨碍日常本地提交。
