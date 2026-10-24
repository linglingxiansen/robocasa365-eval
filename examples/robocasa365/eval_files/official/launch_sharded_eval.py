#!/usr/bin/env python3
"""Launch task-sharded official RoboCasa365 evaluation on dedicated GPUs."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from .native12_policy import Native12Codec


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[3]
MODEL_PORT_BASE = 11093
OFFICIAL_PORT_BASE = 16000

@dataclass(frozen=True)
class ShardedEvaluationConfig:
    checkpoint: Path
    output_dir: Path
    task_set: tuple[str, ...]
    split: str
    n_episodes: int
    n_envs: int
    n_action_steps: int
    save_video: bool
    render_cuda_visible_devices: str
    model_cuda_visible_devices: tuple[str, ...]
    model_action_horizon: int


@dataclass(frozen=True)
class ShardPlan:
    shard_id: int
    model_cuda_visible_devices: str
    render_cuda_visible_devices: str
    model_port: int
    official_port: int
    env_names: tuple[str, ...]
    horizon_sum: int


_REQUIRED_KEYS = {
    "checkpoint",
    "output_dir",
    "task_set",
    "split",
    "n_episodes",
    "n_envs",
    "n_action_steps",
    "render_cuda_visible_devices",
    "model_cuda_visible_devices",
}
_OPTIONAL_KEYS = {"save_video"}


def _positive_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer, got {value!r}")
    return value


def _path(value: Any, key: str, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty path string")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def load_sharded_config(path: str | Path) -> ShardedEvaluationConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open() as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("evaluation config must be a YAML mapping")
    missing = sorted(_REQUIRED_KEYS - set(payload))
    unknown = sorted(set(payload) - _REQUIRED_KEYS - _OPTIONAL_KEYS)
    if missing:
        raise ValueError(f"evaluation config is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"evaluation config contains unknown keys: {unknown}")
    payload = {key: os.path.expandvars(value) if isinstance(value, str) else value for key, value in payload.items()}

    checkpoint = _path(payload["checkpoint"], "checkpoint", config_path.parent)
    output_dir = _path(payload["output_dir"], "output_dir", config_path.parent)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    task_sets = payload["task_set"]
    if (
        not isinstance(task_sets, list)
        or not task_sets
        or any(not isinstance(item, str) or item not in TASK_SET_REGISTRY for item in task_sets)
    ):
        raise ValueError("task_set must contain registered task-set names")
    if len(set(task_sets)) != len(task_sets):
        raise ValueError("task_set must not contain duplicates")
    split = payload["split"]
    if split not in {"pretrain", "target"}:
        raise ValueError("split must be 'pretrain' or 'target'")
    save_video = payload.get("save_video", False)
    if not isinstance(save_video, bool):
        raise ValueError("save_video must be true or false")
    render_devices = payload["render_cuda_visible_devices"]
    if not isinstance(render_devices, str) or not render_devices.strip():
        raise ValueError("render_cuda_visible_devices must be a non-empty string")
    model_devices = payload["model_cuda_visible_devices"]
    if (
        not isinstance(model_devices, list)
        or not model_devices
        or any(not isinstance(item, str) or not item.strip() for item in model_devices)
    ):
        raise ValueError("model_cuda_visible_devices must be a non-empty string list")
    model_devices = [item.strip() for item in model_devices]
    if len(set(model_devices)) != len(model_devices):
        raise ValueError("model_cuda_visible_devices must be unique")
    if render_devices.strip() in model_devices:
        raise ValueError("render GPU must not also host a model server")
    n_action_steps = _positive_int(payload, "n_action_steps")
    action_horizon = Native12Codec.from_checkpoint(checkpoint).action_horizon
    if n_action_steps > action_horizon:
        raise ValueError(
            f"n_action_steps cannot exceed checkpoint action horizon: "
            f"{n_action_steps} > {action_horizon}"
        )
    return ShardedEvaluationConfig(
        checkpoint=checkpoint,
        output_dir=output_dir,
        task_set=tuple(task_sets),
        split=split,
        n_episodes=_positive_int(payload, "n_episodes"),
        n_envs=_positive_int(payload, "n_envs"),
        n_action_steps=n_action_steps,
        save_video=save_video,
        render_cuda_visible_devices=render_devices.strip(),
        model_cuda_visible_devices=tuple(model_devices),
        model_action_horizon=action_horizon,
    )


def selected_tasks(task_sets: Sequence[str]) -> tuple[str, ...]:
    tasks: list[str] = []
    for task_set in task_sets:
        for task in TASK_SET_REGISTRY[task_set]:
            if task not in tasks:
                tasks.append(task)
    return tuple(tasks)


def build_shard_plan(config: ShardedEvaluationConfig) -> tuple[ShardPlan, ...]:
    worker_count = len(config.model_cuda_visible_devices)
    bins: list[list[str]] = [[] for _ in range(worker_count)]
    weights = [0] * worker_count
    indexed_tasks = list(enumerate(selected_tasks(config.task_set)))
    for _, task in sorted(
        indexed_tasks,
        key=lambda item: (-get_task_horizon(item[1]), item[0]),
    ):
        shard_id = min(range(worker_count), key=lambda index: (weights[index], index))
        bins[shard_id].append(task)
        weights[shard_id] += get_task_horizon(task)
    return tuple(
        ShardPlan(
            shard_id=shard_id,
            model_cuda_visible_devices=config.model_cuda_visible_devices[shard_id],
            render_cuda_visible_devices=config.render_cuda_visible_devices,
            model_port=MODEL_PORT_BASE + shard_id,
            official_port=OFFICIAL_PORT_BASE + shard_id,
            env_names=tuple(bins[shard_id]),
            horizon_sum=weights[shard_id],
        )
        for shard_id in range(worker_count)
    )


def _runtime_paths() -> tuple[Path, Path, Path]:
    keys = ("STARVLA_PYTHON", "ROBOCASA_PYTHON", "ROBOCASA_WEBSOCKET_SITE_PACKAGES")
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        raise ValueError(f"Missing runtime settings {missing}; configure the environment using scripts/run_eval.py")
    return tuple(Path(os.environ[key]) for key in keys)


def build_worker_specs(
    config: ShardedEvaluationConfig,
) -> tuple[dict[str, Any], ...]:
    model_python, simulation_python, websocket_site_packages = _runtime_paths()
    model_server = PROJECT / "deployment/model_server/server_policy.py"
    specs = []
    for shard in build_shard_plan(config):
        model_command = [
            str(model_python),
            str(model_server),
            "--ckpt_path",
            str(config.checkpoint),
            "--port",
            str(shard.model_port),
            "--host",
            "127.0.0.1",
            "--use_bf16",
            "--idle_timeout",
            "-1",
        ]
        eval_command = [
            str(simulation_python),
            "-m",
            "examples.robocasa365.eval_files.official.run_eval_shard",
            "--shard-id",
            str(shard.shard_id),
            "--env-names",
            *shard.env_names,
            "--model-path",
            str(config.checkpoint),
            "--output-dir",
            str(config.output_dir),
            "--split",
            config.split,
            "--n-episodes",
            str(config.n_episodes),
            "--n-envs",
            str(config.n_envs),
            "--n-action-steps",
            str(config.n_action_steps),
            "--port",
            str(shard.official_port),
            "--starvla-port",
            str(shard.model_port),
            "--websocket-site-packages",
            str(websocket_site_packages),
        ]
        if config.save_video:
            eval_command.append("--video")
        specs.append(
            {
                "shard": asdict(shard),
                "model_command": model_command,
                "model_environment": {
                    "CUDA_VISIBLE_DEVICES": shard.model_cuda_visible_devices
                },
                "eval_command": eval_command,
                "eval_environment": {
                    "CUDA_VISIBLE_DEVICES": shard.render_cuda_visible_devices,
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                },
            }
        )
    return tuple(specs)


def pending_worker_specs(
    config: ShardedEvaluationConfig,
) -> tuple[dict[str, Any], ...]:
    """Return only shards that still contain at least one unfinished task."""
    return tuple(
        spec
        for spec in build_worker_specs(config)
        if any(
            not (
                config.output_dir
                / "evals"
                / config.split
                / env_name
                / "stats.json"
            ).is_file()
            for env_name in spec["shard"]["env_names"]
        )
    )


def manifest(config: ShardedEvaluationConfig) -> dict[str, Any]:
    specs = build_worker_specs(config)
    return {
        "checkpoint": str(config.checkpoint),
        "output_dir": str(config.output_dir),
        "task_set": list(config.task_set),
        "task_count": len(selected_tasks(config.task_set)),
        "split": config.split,
        "n_episodes": config.n_episodes,
        "n_envs": config.n_envs,
        "n_action_steps": config.n_action_steps,
        "save_video": config.save_video,
        "render_cuda_visible_devices": config.render_cuda_visible_devices,
        "model_cuda_visible_devices": list(config.model_cuda_visible_devices),
        "model_action_horizon": config.model_action_horizon,
        "workers": [
            {
                **spec["shard"],
                "env_names": list(spec["shard"]["env_names"]),
                "model_command": shlex.join(spec["model_command"]),
                "model_environment": spec["model_environment"],
                "eval_command": shlex.join(spec["eval_command"]),
                "eval_environment": spec["eval_environment"],
            }
            for spec in specs
        ],
    }


def _validate_stats(path: Path, n_episodes: int) -> None:
    try:
        stats = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid evaluation statistics: {path}") from error
    if (
        not isinstance(stats, dict)
        or stats.get("num_episodes") != n_episodes
        or isinstance(stats.get("num_episodes"), bool)
        or not isinstance(stats.get("success_rate"), (int, float))
        or isinstance(stats.get("success_rate"), bool)
        or not 0 <= stats["success_rate"] <= 1
    ):
        raise ValueError(f"Evaluation statistics do not contain {n_episodes} valid episodes: {path}")


def _validate_existing_output(config: ShardedEvaluationConfig, current_manifest: dict[str, Any]) -> None:
    manifest_path = config.output_dir / "shard_manifest.json"
    stats_paths = list(config.output_dir.glob("evals/*/*/stats.json"))
    if not manifest_path.is_file():
        if stats_paths:
            raise ValueError(
                f"Existing evaluation results have no shard_manifest.json: {config.output_dir}"
            )
        return
    try:
        previous_manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid evaluation manifest: {manifest_path}") from error
    if previous_manifest != current_manifest:
        raise ValueError(
            f"Evaluation output belongs to a different checkpoint or protocol: {config.output_dir}"
        )
    expected = {
        config.output_dir / "evals" / config.split / name / "stats.json"
        for name in selected_tasks(config.task_set)
    }
    unexpected = set(stats_paths) - expected
    if unexpected:
        raise ValueError(f"Evaluation output contains unexpected task results: {sorted(unexpected)}")
    for path in stats_paths:
        _validate_stats(path, config.n_episodes)


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    python_paths = tuple(dict.fromkeys([str(PROJECT), *filter(None, environment.get("PYTHONPATH", "").split(os.pathsep))]))
    environment.update(
        PYTHONPATH=":".join(map(str, python_paths)),
        NO_ALBUMENTATIONS_UPDATE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    return environment


def _stop(processes: Sequence[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def _model_servers_ready(
    specs: Sequence[Mapping[str, Any]], log_dir: Path
) -> bool:
    markers = ("server running", "server listening")
    return all(
        any(
            marker
            in (log_dir / f"model_shard{spec['shard']['shard_id']}.log")
            .read_text()
            .lower()
            for marker in markers
        )
        for spec in specs
    )


def run(config: ShardedEvaluationConfig) -> None:
    current_manifest = manifest(config)
    _validate_existing_output(config, current_manifest)
    specs = pending_worker_specs(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = config.output_dir / "sharded_logs"
    log_dir.mkdir(exist_ok=True)
    (config.output_dir / "shard_manifest.json").write_text(
        json.dumps(current_manifest, indent=2)
    )
    environment = _base_environment()
    processes: list[subprocess.Popen] = []
    model_processes: list[subprocess.Popen] = []
    eval_processes: list[subprocess.Popen] = []
    handles = []
    try:
        for spec in specs:
            shard_id = spec["shard"]["shard_id"]
            handle = (log_dir / f"model_shard{shard_id}.log").open("w")
            handles.append(handle)
            process = subprocess.Popen(
                spec["model_command"],
                env={**environment, **spec["model_environment"]},
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=PROJECT,
            )
            processes.append(process)
            model_processes.append(process)
        deadline = time.monotonic() + 300
        while True:
            if _model_servers_ready(specs, log_dir):
                break
            if any(process.poll() is not None for process in model_processes):
                raise RuntimeError("A model server exited during startup")
            if time.monotonic() > deadline:
                raise TimeoutError("Model server startup timed out")
            time.sleep(2)
        for spec in specs:
            shard_id = spec["shard"]["shard_id"]
            handle = (log_dir / f"eval_shard{shard_id}.log").open("a")
            handles.append(handle)
            process = subprocess.Popen(
                spec["eval_command"],
                env={**environment, **spec["eval_environment"]},
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=PROJECT,
            )
            processes.append(process)
            eval_processes.append(process)
        failures = []
        for process in eval_processes:
            return_code = process.wait()
            if return_code:
                failures.append(return_code)
        if failures:
            raise RuntimeError(f"Evaluation shards failed: {failures}")
        stats_paths = [
            config.output_dir / "evals" / config.split / name / "stats.json"
            for name in selected_tasks(config.task_set)
        ]
        for path in stats_paths:
            if path.is_file():
                _validate_stats(path, config.n_episodes)
        summary = {
            "completed_tasks": sum(path.is_file() for path in stats_paths),
            "expected_tasks": len(stats_paths),
            "stats_paths": [str(path) for path in stats_paths if path.is_file()],
        }
        (config.output_dir / "sharded_summary.json").write_text(
            json.dumps(summary, indent=2)
        )
        if summary["completed_tasks"] != summary["expected_tasks"]:
            raise RuntimeError(f"Incomplete evaluation: {summary}")
    finally:
        _stop(processes)
        for handle in handles:
            handle.close()


def _signal_to_exit(signum, frame) -> None:
    del signum, frame
    raise SystemExit(128)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--background", action="store_true")
    mode.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_sharded_config(args.config)
    payload = manifest(config)
    print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run:
        return
    if args.background:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        runner_log = (config.output_dir / "sharded_runner.log").open("a")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "examples.robocasa365.eval_files.official.launch_sharded_eval",
                str(Path(args.config).resolve()),
                "--foreground",
            ],
            stdout=runner_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        (config.output_dir / "sharded_eval.pid").write_text(f"{process.pid}\n")
        print(f"Started sharded evaluation pid={process.pid}; log={runner_log.name}")
        runner_log.close()
        return
    signal.signal(signal.SIGTERM, _signal_to_exit)
    signal.signal(signal.SIGINT, _signal_to_exit)
    run(config)


if __name__ == "__main__":
    main()
