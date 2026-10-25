# Working in this repository

This directory is the canonical local source repository for the Intel Arc Pro B60 LLM Lab.

- Make project code, documentation, presets and test changes here. Keep model weights, runtimes, toolchains, temporary uploads and unrelated deployments outside version control.
- Commit completed, coherent changes locally, with a short message explaining the change. Inspect the diff and run checks appropriate to the change before committing. Do not include unrelated user changes.
- The default workflow ends at a local commit. Do not automatically push after a commit or publish local work through the GitHub website or connector.
- Publish a completed stage only when the user asks to publish or push that stage. Review the commits and checks, then push the selected work to the existing origin remote.
- Preserve historical measurement snapshots. New experiments get new result directories and must distinguish measured results from assumptions.

See [docs/workflow.md](docs/workflow.md) for commands and release checkpoints.
