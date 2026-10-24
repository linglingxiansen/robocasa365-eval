# RoboCasa365 policy evaluation

Private reviewer package for the 200,000-step MR0 / StarVLA-based Native12
policy. It contains inference and evaluation code, portable configuration,
normalization statistics, and the complete self-evaluated reference results.
Weights are distributed separately through a private model repository.

The reference completed 50 tasks × 50 episodes on RoboCasa 1.0.1:

| Split | Successes / episodes | Success rate |
| --- | ---: | ---: |
| Atomic-Seen | 711 / 900 | 79.00% |
| Composite-Seen | 432 / 800 | 54.00% |
| Composite-Unseen | 218 / 800 | 27.25% |
| Overall | 1361 / 2500 | 54.44% |

These are author-run results, not official verification. Reference evaluation
used pretraining kitchens, 16 executed actions per request, five vector
environments per shard, ten Euler denoising steps, and official v1.0.1 task
horizons. Eighteen atomic, sixteen seen-composite and sixteen unseen-composite
tasks are included. Model input uses camera order left external, right external,
wrist; RGB/state history offsets are `[-6,-4,-2,0]` at 20 Hz; image size is
640×320; state dimension is 20 and action dimension is 12. Prediction horizon
is 50. No task/subtask predictions are fed into the action policy.

The reference did not record per-episode seeds or videos. A new run is a
statistical reproduction, not an exact episode-by-episode replay. Do not add
another 1.5× horizon multiplier: the RoboCasa 1.0.1 registry already includes it.

## 1. Environments

Linux, Python 3.11, NVIDIA GPU/CUDA and EGL are required for live evaluation.
Use separate environments for model inference and simulation. The reference
layout uses eight GPUs: GPU 0 renders and GPUs 1–7 host models. A two-GPU
configuration is provided for lower-throughput verification. Never assign the
same GPU to both roles in these configurations.

```bash
conda create -n policy-model python=3.11 -y
conda activate policy-model
python -m pip install -r requirements/model.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
python -m pip install --no-deps -e .

conda create -n policy-sim python=3.11 -y
conda activate policy-sim
python -m pip install -r requirements/simulation.txt
python -m pip install --no-deps -e .
```

Use a torch build compatible with the host CUDA runtime. Qwen3.5 may also
benefit from matching `flash-linear-attention` and `causal-conv1d` builds.

Install the simulator outside this repository, following the official
[RoboCasa installation guide](https://robocasa.ai/docs/introduction/installation.html).
The public compatible source pins are:

- [robosuite](https://github.com/ARISE-Initiative/robosuite): `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`
- [RoboCasa](https://github.com/robocasa/robocasa): `b4684e6ee37d377cc392e98302a6b916d588b415`

Install both source checkouts with `python -m pip install -e <checkout>` in
the simulation environment. Run `python -m robocasa.scripts.setup_macros`
and `python -m robocasa.scripts.download_kitchen_assets` there. Training
demonstrations are not needed for evaluation. The simulator must expose the
native PandaOmron 12D interface and the target50 task groups.
For existing checkouts that are not installed editable, pass
`--robocasa-root /path/to/robocasa --robosuite-root /path/to/robosuite`
to `scripts/run_eval.py`. No machine-specific paths are embedded in the code.

## 2. Private checkpoint bundle

The matching bundle is hosted in the private Hugging Face model repository
[`llxs-robocasa-review/robocasa365-policy-200k`](https://huggingface.co/llxs-robocasa-review/robocasa365-policy-200k).
The reviewer must first receive access to that repository. In the model
environment, authenticate with your own authorized Hugging Face account and
download the pinned revision from this repository's root:

```bash
hf auth login
hf download llxs-robocasa-review/robocasa365-policy-200k \
  --revision 2deea919023acc8e49a5588c1406fa6c5a139e24 \
  --local-dir artifacts/200k
```

The bundle contains 11 files, including an 11,365,636,299-byte checkpoint with
SHA-256 `b6c2d13e1ec13b0bf755e33cb54a8e9d53dbe576280dcf330e5199fc9957e2c9`.
See [artifacts/README.md](artifacts/README.md) for the directory layout.
Keep the exact Qwen3.5-4B-Action tokenizer/processor assets supplied with the
checkpoint; the trained checkpoint contains both VLM and action-head tensors.

For the submitter, create a portable bundle from local assets:

```bash
python scripts/export_artifacts.py \
  --checkpoint /path/to/trained/checkpoint.pt \
  --base-vlm /path/to/matching/Qwen3.5-4B-Action \
  --output artifacts/200k --copy-weights
```

This copies inference-only metadata, strips cached source-location metadata
from JSON assets, renames the checkpoint to `checkpoints/model.pt`, and writes
SHA-256 checksums. It does not copy original training configs, logs, optimizer
states, datasets, credentials or source-control history. The model architecture
is initialized from its supplied config and the complete trained state dict is
then restored; a second copy of base-model weights is unnecessary.

## 3. Inspect the run before launching

Run from the repository root. Replace both interpreter paths with their local
absolute paths, e.g. the output of `python -c 'import sys; print(sys.executable)'`
in each environment. Set `--config configs/eval_2gpu.yaml` for two GPUs.

```bash
python scripts/run_eval.py \
  --checkpoint artifacts/200k/checkpoints/model.pt \
  --model-python /path/to/policy-model/bin/python \
  --simulation-python /path/to/policy-sim/bin/python \
  --config configs/eval_8gpu.yaml \
  --output outputs/eval_200k --dry-run
```

This checks checkpoint metadata, horizon and shard assignment without loading
weights or initializing EGL. Missing, duplicate or mixed results are rejected.
Each output directory is bound to its checkpoint path and evaluation protocol;
do not replace a checkpoint in-place when resuming a run.

## 4. Rendering preflight

First inspect `nvidia-smi` and ensure the selected render GPU is not assigned
to an active job. In the simulation environment, run the quick and
layout-matched stress probes before a full evaluation on a new/restarted host:

```bash
python -m examples.robocasa365.eval_files.official.preflight.check_render_gpu \
  --gpus 0 --mode auto --n-envs 5 --stress-workers 7 \
  --json-output outputs/preflight/render.json
```

For two GPUs, use `--stress-workers 1`. Both probe phases must pass and the
driver must remain responsive. See [the preflight workflow](docs/PREFLIGHT.md).
Configuration inspection and CPU tests require no live GPU probe.

## 5. Evaluate

After preflight passes, repeat the command from section 3 without `--dry-run`.
Foreground mode waits for all shards and computes metrics automatically.
Add `--background` to detach; after completion aggregate results explicitly:

```bash
python -m examples.robocasa365.eval_files.compute_metrics outputs/eval_200k
```

Rerunning the same config/output resumes completed tasks. Logs go to
`outputs/eval_200k/sharded_logs/`; aggregate results are `metrics.json` and
`metrics_per_task.csv`. A result is complete only when all 50 tasks contain
exactly 50 valid episodes. For failure videos, copy the evaluation YAML to an
ignored `configs/*.local.yaml`, set `save_video: true`, and use a new output
directory. Keep the official full-evaluation protocol at 50 episodes per task.

The standalone model endpoint can also be run in the model environment:

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/serve.py \
  --checkpoint artifacts/200k/checkpoints/model.pt --port 11093
```

## 6. Local validation and reference data

```bash
python scripts/verify_reference.py
python scripts/audit_repository.py
python -m pip install -r requirements/dev.txt
python -m pytest -q
```

The private model bundle can be checked without allocating GPU weights:

```bash
python scripts/validate_checkpoint.py \
  --checkpoint artifacts/200k/checkpoints/model.pt \
  --config artifacts/200k/config.full.yaml --base-vlm artifacts/200k/base_vlm
```

The history/model tests require model dependencies; the lightweight packaging
and reference tests run with pytest alone. Local validation performed while
preparing this snapshot is recorded in `validation/report.json`. It does not
represent a new 2,500-episode rollout from this portable snapshot.

## Provenance and private submission

The policy adapts a pretrained MR0 robot policy; RoboCasa adaptation used
Human300 (300 tasks × 100 human demonstrations), four-observation history,
640×320 observations and mode-balanced action loss. Both the VLM and action
head were fine-tuned; the effective training batch was 256 and this checkpoint
was saved at 200,000 steps. Preserve upstream model/data attribution in any
submission. The snapshot contains no training entrypoints.

See [NOTICE.md](NOTICE.md) for attribution and licenses. Sanitization removes
private infrastructure and personal run metadata while retaining third-party
copyright notices. Use a private GitHub repository for this code and a separate
private model repository for weights; grant the official reviewer access to
both. No repository, PR or artifact is automatically uploaded by these scripts.
