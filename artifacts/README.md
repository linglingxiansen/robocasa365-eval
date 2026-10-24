Place the private, portable model bundle here. Only this README is tracked.

```text
200k/
  checkpoints/model.pt
  config.full.yaml
  dataset_statistics.json
  base_vlm/                 # exact architecture, processor and tokenizer assets
  SHA256SUMS.json
```

Use `scripts/export_artifacts.py` to create the bundle. Do not substitute a
generic Qwen tokenizer for the checkpoint's matching processor/tokenizer.
The complete trained VLM and action-head tensors are restored from `model.pt`.
Raw weights and generated bundles are excluded from Git; distribute the bundle
through a separate private checkpoint repository.
