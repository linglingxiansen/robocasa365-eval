#!/usr/bin/env python3
"""Find a GPU that can sustain RoboCasa365 multi-process EGL rendering."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Sequence

DEFAULT_TASK = "OpenCabinet"
DEFAULT_N_ENVS = 5
DEFAULT_N_ACTION_STEPS = 16
QUICK_WORKERS = 1
DEFAULT_STRESS_WORKERS = 7


@dataclass(frozen=True)
class GpuStatus:
    index: str
    memory_used_mib: int
    utilization_percent: int


@dataclass(frozen=True)
class ProbeResult:
    gpu: str
    mode: str
    passed: bool
    workers: int
    n_envs: int
    total_envs: int
    duration_seconds: float
    reason: str
    worker_return_codes: tuple[int | None, ...]
    logs: tuple[str, ...]


def parse_gpu_selection(value: str, available: Sequence[str]) -> tuple[str, ...]:
    """Resolve ``all`` or a comma-separated physical GPU index list."""

    if value.strip().lower() == "all":
        if not available:
            raise ValueError("nvidia-smi did not report any GPUs")
        return tuple(available)
    selected = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selected:
        raise ValueError("--gpus must be 'all' or a comma-separated GPU index list")
    if len(set(selected)) != len(selected):
        raise ValueError(f"--gpus contains duplicates: {selected}")
    unknown = [gpu for gpu in selected if gpu not in available]
    if unknown:
        raise ValueError(f"GPU indices are not available: {unknown}; available={list(available)}")
    return selected


def recommended_model_gpus(render_gpu: str, available: Sequence[str]) -> tuple[str, ...]:
    """Return every physical GPU except the selected render GPU."""

    return tuple(gpu for gpu in available if gpu != render_gpu)


def _query_gpu_status(timeout_seconds: float = 5.0) -> tuple[GpuStatus, ...]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi is not installed or not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"nvidia-smi did not respond within {timeout_seconds:.1f}s; the NVIDIA driver may already be blocked"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"nvidia-smi failed: {detail}") from exc

    statuses = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError(f"unexpected nvidia-smi output: {line!r}")
        statuses.append(
            GpuStatus(
                index=fields[0],
                memory_used_mib=int(fields[1]),
                utilization_percent=int(fields[2]),
            )
        )
    return tuple(statuses)


def _worker_command(args: argparse.Namespace, worker_id: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "examples.robocasa365.eval_files.official.preflight.check_render_gpu",
        "--worker",
        "--worker-id",
        str(worker_id),
        "--task",
        args.task,
        "--split",
        args.split,
        "--n-envs",
        str(args.n_envs),
        "--n-action-steps",
        str(args.n_action_steps),
        "--steps",
        str(args.steps),
    ]


def _worker_environment(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=gpu,
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        PYTHONUNBUFFERED="1",
        NO_ALBUMENTATIONS_UPDATE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    # The production evaluator intentionally relies on one visible render GPU.
    environment.pop("MUJOCO_EGL_DEVICE_ID", None)
    return environment


def _terminate_process_groups(processes: Sequence[subprocess.Popen[Any]]) -> None:
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while any(process.poll() is None for process in alive) and time.monotonic() < deadline:
        time.sleep(0.2)
    for process in alive:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _tail(text: str, max_lines: int = 80) -> str:
    return "\n".join(text.splitlines()[-max_lines:])


def run_candidate(
    gpu: str,
    mode: str,
    workers: int,
    timeout_seconds: float,
    args: argparse.Namespace,
) -> ProbeResult:
    """Run independent official-vector-env workers concurrently on one GPU."""

    started = time.monotonic()
    processes: list[subprocess.Popen[Any]] = []
    handles = []
    try:
        for worker_id in range(workers):
            # Some shared filesystems cannot read an already-unlinked file.
            # Keep the log path alive until its contents have been collected.
            handle = tempfile.NamedTemporaryFile(mode="w+t")
            handles.append(handle)
            processes.append(
                subprocess.Popen(
                    _worker_command(args, worker_id),
                    env=_worker_environment(gpu),
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
            )

        deadline = started + timeout_seconds
        reason = ""
        while True:
            return_codes = [process.poll() for process in processes]
            nonzero = [code for code in return_codes if code not in (None, 0)]
            if nonzero:
                reason = f"render worker exited with return code {nonzero[0]}"
                _terminate_process_groups(processes)
                break
            if all(code == 0 for code in return_codes):
                break
            if time.monotonic() >= deadline:
                reason = (
                    f"timed out after {timeout_seconds:.1f}s; EGL workers may be blocked in the NVIDIA kernel driver"
                )
                _terminate_process_groups(processes)
                break
            time.sleep(0.25)

        return_codes = tuple(process.poll() for process in processes)
        logs = []
        for handle in handles:
            handle.flush()
            handle.seek(0)
            logs.append(_tail(handle.read()))
        passed = not reason and all(code == 0 for code in return_codes)
        if passed:
            try:
                _query_gpu_status(timeout_seconds=5.0)
            except RuntimeError as exc:
                passed = False
                reason = f"render completed but driver health query failed: {exc}"
        if passed:
            reason = "reset and step completed for every worker; nvidia-smi remained responsive"
        return ProbeResult(
            gpu=gpu,
            mode=mode,
            passed=passed,
            workers=workers,
            n_envs=args.n_envs,
            total_envs=workers * args.n_envs,
            duration_seconds=round(time.monotonic() - started, 2),
            reason=reason,
            worker_return_codes=return_codes,
            logs=tuple(logs if args.verbose or not passed else ()),
        )
    finally:
        _terminate_process_groups(processes)
        for handle in handles:
            handle.close()


def _zero_action(space: Any) -> Any:
    import gymnasium as gym
    import numpy as np

    if isinstance(space, gym.spaces.Dict):
        return {key: _zero_action(value) for key, value in space.items()}
    if isinstance(space, gym.spaces.Box):
        zeros = np.zeros(space.shape, dtype=space.dtype)
        return np.clip(zeros, space.low, space.high).astype(space.dtype, copy=False)
    return space.sample()


def _validate_video_observation(observation: Any, n_envs: int) -> dict[str, list[int]]:
    import numpy as np

    if not isinstance(observation, dict):
        raise RuntimeError(f"expected dict observation, got {type(observation).__name__}")
    video_shapes = {}
    for key, value in observation.items():
        if not key.startswith("video"):
            continue
        array = np.asarray(value)
        if array.shape[0] != n_envs:
            raise RuntimeError(f"{key} has invalid vector batch {array.shape}; expected first dimension {n_envs}")
        if array.size == 0 or not np.isfinite(array).all():
            raise RuntimeError(f"{key} contains an empty, NaN, or Inf render result")
        video_shapes[key] = list(array.shape)
    if not video_shapes:
        raise RuntimeError("environment observation did not contain rendered video fields")
    return video_shapes


def worker_main(args: argparse.Namespace) -> int:
    """Create the exact official vector environment without loading a VLA model."""

    import gymnasium as gym
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    from examples.robocasa365.eval_files.official.simulation import (
        MultiStepConfig,
        SimulationConfig,
        VideoConfig,
        _create_single_env,
    )

    config = SimulationConfig(
        env_name=f"robocasa/{args.task}",
        split=args.split,
        n_episodes=1,
        n_envs=args.n_envs,
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(
            n_action_steps=args.n_action_steps,
            max_episode_steps=get_task_horizon(args.task),
        ),
    )
    env_fns = [partial(_create_single_env, config=config, idx=index) for index in range(config.n_envs)]
    environment = None
    started = time.monotonic()
    try:
        if config.n_envs == 1:
            environment = gym.vector.SyncVectorEnv(env_fns)
        else:
            environment = gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )
        observation, _ = environment.reset(seed=args.worker_id)
        video_shapes = _validate_video_observation(observation, config.n_envs)
        for _ in range(args.steps):
            action = _zero_action(environment.action_space)
            observation, _, _, _, _ = environment.step(action)
            video_shapes = _validate_video_observation(observation, config.n_envs)
        print(
            "PREFLIGHT_RESULT="
            + json.dumps(
                {
                    "status": "pass",
                    "worker_id": args.worker_id,
                    "n_envs": args.n_envs,
                    "steps": args.steps,
                    "video_shapes": video_shapes,
                    "duration_seconds": round(time.monotonic() - started, 2),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        if environment is not None:
            environment.close()


def _print_result(result: ProbeResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(
        f"[{status}] GPU {result.gpu} {result.mode}: "
        f"{result.total_envs} envs, {result.duration_seconds:.2f}s — {result.reason}",
        flush=True,
    )
    if result.logs:
        for index, log in enumerate(result.logs):
            print(f"--- GPU {result.gpu} {result.mode} worker {index} log ---")
            print(log)


def _busy_reason(status: GpuStatus, args: argparse.Namespace) -> str | None:
    if args.allow_busy:
        return None
    if status.memory_used_mib > args.max_memory_used_mib:
        return f"GPU is busy: memory.used={status.memory_used_mib} MiB exceeds {args.max_memory_used_mib} MiB"
    if status.utilization_percent > args.max_utilization:
        return f"GPU is busy: utilization={status.utilization_percent}% exceeds {args.max_utilization}%"
    return None


def _skipped_result(gpu: str, mode: str, workers: int, args: argparse.Namespace, reason: str) -> ProbeResult:
    return ProbeResult(
        gpu=gpu,
        mode=mode,
        passed=False,
        workers=workers,
        n_envs=args.n_envs,
        total_envs=workers * args.n_envs,
        duration_seconds=0.0,
        reason=reason,
        worker_return_codes=(),
        logs=(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="all", help="all or comma-separated physical GPU indices")
    parser.add_argument("--mode", choices=("quick", "stress", "auto"), default="auto")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--split", choices=("pretrain", "target"), default="pretrain")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--n-action-steps", type=int, default=DEFAULT_N_ACTION_STEPS)
    parser.add_argument("--steps", type=int, default=1, help="vector steps after reset")
    parser.add_argument(
        "--stress-workers",
        type=int,
        default=DEFAULT_STRESS_WORKERS,
        help="concurrent vector workers in the stress phase (default: 7)",
    )
    parser.add_argument("--quick-timeout", type=float, default=180.0)
    parser.add_argument("--stress-timeout", type=float, default=300.0)
    parser.add_argument("--max-memory-used-mib", type=int, default=1024)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-id", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.n_envs < 1
        or args.n_action_steps < 1
        or args.steps < 1
        or args.stress_workers < 1
        or args.quick_timeout <= 0
        or args.stress_timeout <= 0
    ):
        parser.error("environment counts, steps, and timeouts must be positive")
    if args.max_memory_used_mib < 0 or not 0 <= args.max_utilization <= 100:
        parser.error("busy-GPU thresholds must be non-negative and utilization at most 100")
    if args.worker:
        return worker_main(args)

    try:
        initial_statuses = _query_gpu_status()
        available = tuple(status.index for status in initial_statuses)
        selected = parse_gpu_selection(args.gpus, available)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    status_by_gpu = {status.index: status for status in initial_statuses}
    quick_results: list[ProbeResult] = []
    stress_results: list[ProbeResult] = []

    if args.mode in {"quick", "auto"}:
        for gpu in selected:
            busy = _busy_reason(status_by_gpu[gpu], args)
            result = (
                _skipped_result(gpu, "quick", QUICK_WORKERS, args, busy)
                if busy
                else run_candidate(
                    gpu,
                    "quick",
                    QUICK_WORKERS,
                    args.quick_timeout,
                    args,
                )
            )
            quick_results.append(result)
            _print_result(result)

    if args.mode == "stress":
        stress_candidates = selected
    elif args.mode == "auto":
        stress_candidates = tuple(result.gpu for result in quick_results if result.passed)
    else:
        stress_candidates = ()

    recommended_gpu = None
    for gpu in stress_candidates:
        try:
            current = {status.index: status for status in _query_gpu_status()}[gpu]
            busy = _busy_reason(current, args)
        except (RuntimeError, KeyError) as exc:
            busy = str(exc)
        result = (
            _skipped_result(gpu, "stress", args.stress_workers, args, busy)
            if busy
            else run_candidate(
                gpu,
                "stress",
                args.stress_workers,
                args.stress_timeout,
                args,
            )
        )
        stress_results.append(result)
        _print_result(result)
        if result.passed:
            recommended_gpu = gpu
            break

    if args.mode == "quick":
        quick_passes = [result.gpu for result in quick_results if result.passed]
        recommended_gpu = quick_passes[0] if quick_passes else None

    result_payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "settings": {
            "gpus": list(selected),
            "mode": args.mode,
            "task": args.task,
            "split": args.split,
            "n_envs": args.n_envs,
            "n_action_steps": args.n_action_steps,
            "steps": args.steps,
            "stress_workers": args.stress_workers,
            "quick_timeout": args.quick_timeout,
            "stress_timeout": args.stress_timeout,
        },
        "initial_gpu_status": [asdict(status) for status in initial_statuses],
        "quick_results": [asdict(result) for result in quick_results],
        "stress_results": [asdict(result) for result in stress_results],
        "recommended_render_gpu": recommended_gpu,
        "recommended_model_gpus": (
            list(recommended_model_gpus(recommended_gpu, available)) if recommended_gpu is not None else []
        ),
    }
    if args.json_output:
        args.json_output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        args.json_output.expanduser().resolve().write_text(json.dumps(result_payload, indent=2) + "\n")

    if recommended_gpu is None:
        print("No render GPU passed the requested preflight mode.", file=sys.stderr)
        return 2
    models = ",".join(result_payload["recommended_model_gpus"])
    print(f"recommended_render_gpu={recommended_gpu}")
    print(f"recommended_model_gpus={models}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
