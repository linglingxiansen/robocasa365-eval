"""Native-12 RoboCasa365 checkpoint contract and StarVLA policy adapter.

The training loader represents RoboCasa observations as five named state
fields.  Two quaternion fields are converted to rotation-6D before the model
receives the resulting 20-D state.  Conversely, the model returns a
normalized H x 12 action chunk which has to be decoded into the named native
PandaOmron controller action before it is passed to the Gym wrapper.  ``H``
is read from the checkpoint; the official evaluator selects how many steps to
execute through ``--n_action_steps``.

This module keeps that contract explicit and deliberately does not depend on
the training data loader.  Evaluation therefore never opens or writes the
official dataset assets.
"""

from __future__ import annotations

import json
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EMBODIMENT_TAG = "new_embodiment"
ACTION_DIM = 12
RAW_STATE_DIM = 16
STATE_DIM = 20

VIDEO_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_KEYS = (
    "state.end_effector_position_relative",
    "state.end_effector_rotation_relative",
    "state.gripper_qpos",
    "state.base_position",
    "state.base_rotation",
)
ACTION_KEYS = (
    "action.end_effector_position",
    "action.end_effector_rotation",
    "action.gripper_close",
    "action.base_motion",
    "action.control_mode",
)

_RAW_STATE_SLICES = {
    "state.end_effector_position_relative": slice(0, 3),
    "state.end_effector_rotation_relative": slice(3, 7),
    "state.gripper_qpos": slice(7, 9),
    "state.base_position": slice(9, 12),
    "state.base_rotation": slice(12, 16),
}
_ACTION_SLICES = {
    "action.end_effector_position": slice(0, 3),
    "action.end_effector_rotation": slice(3, 6),
    "action.gripper_close": slice(6, 7),
    "action.base_motion": slice(7, 11),
    "action.control_mode": slice(11, 12),
}
_CONTINUOUS_ACTION_INDICES = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10])
_BINARY_ACTION_INDICES = np.array([6, 11])


class Native12ContractError(ValueError):
    """Raised when a checkpoint or online sample is not the MR0 ABI."""


def _as_last_frame(value: Any, dimension: int, key: str) -> np.ndarray:
    """Read a D-vector from either a Gym frame or a [T, D] wrapper output."""

    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1 and array.shape == (dimension,):
        frame = array
    elif array.ndim == 2 and array.shape[-1] == dimension and array.shape[0] >= 1:
        frame = array[-1]
    else:
        raise Native12ContractError(
            f"{key} must have shape ({dimension},) or [T, {dimension}], got {array.shape}"
        )
    if not np.isfinite(frame).all():
        raise Native12ContractError(f"{key} contains NaN or Inf")
    return frame


def _as_last_image(value: Any, key: str) -> np.ndarray:
    """Read an HWC RGB frame without changing the camera's spatial content."""

    array = np.asarray(value)
    if array.ndim == 4:
        array = array[-1]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise Native12ContractError(f"{key} must be HWC RGB or [T,H,W,C], got {array.shape}")
    if not np.isfinite(array).all():
        raise Native12ContractError(f"{key} contains NaN or Inf")
    return array


def _as_history_frames(value: Any, key: str, history_length: int) -> np.ndarray:
    """Return a fixed-length [T,H,W,C] history, left-padded at episode start."""

    array = np.asarray(value)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] < 1:
        raise Native12ContractError(
            f"{key} must be HWC RGB or [T,H,W,C], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise Native12ContractError(f"{key} contains NaN or Inf")
    if history_length < 1:
        raise Native12ContractError(f"history_length must be positive, got {history_length}")
    if array.shape[0] >= history_length:
        return np.asarray(array[-history_length:]).copy()
    padding = np.repeat(array[:1], history_length - array.shape[0], axis=0)
    return np.concatenate([padding, array], axis=0)


def _as_video_frames(value: Any, key: str, history_length: int) -> list[np.ndarray]:
    """Return the camera's obs-history as a frame list (oldest -> latest).

    The MultiStepWrapper already samples exactly ``history_length`` frames per
    observation (repeating the reset frame at episode start), so this only
    validates the stack and splits it into frames.
    """

    array = np.asarray(value)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise Native12ContractError(
            f"{key} must be [T,H,W,C] with T={history_length} for obs-history evaluation, "
            f"got {array.shape}"
        )
    if array.shape[0] != history_length:
        raise Native12ContractError(
            f"{key} must supply exactly {history_length} frames (wrapper video_delta_indices "
            f"must equal the checkpoint's obs_history_indices), got {array.shape[0]}"
        )
    if not np.isfinite(array).all():
        raise Native12ContractError(f"{key} contains NaN or Inf")
    return [array[index] for index in range(history_length)]


def quaternion_wxyz_to_rotation_6d(quaternion: np.ndarray) -> np.ndarray:
    """Match StarVLA's ``RotationTransform(quaternion -> rotation_6d)``.

    StarVLA uses PyTorch3D's scalar-first quaternion convention and flattens
    the first two rows of the rotation matrix.  Writing the small equivalent
    here avoids importing training-only dataset dependencies at evaluation.
    """

    q = np.asarray(quaternion, dtype=np.float32)
    if q.shape != (4,):
        raise Native12ContractError(f"quaternion must have shape (4,), got {q.shape}")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise Native12ContractError("quaternion must have a finite non-zero norm")
    w, x, y, z = q / norm
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return matrix[:2, :].reshape(6)


def _minmax_forward(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """StarVLA ``Normalizer(mode='min_max').forward`` semantics."""

    value = np.asarray(value, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    if value.shape[-1] != lower.shape[-1] or lower.shape != upper.shape:
        raise Native12ContractError(
            f"min-max shape mismatch: value={value.shape}, lower={lower.shape}, upper={upper.shape}"
        )
    result = np.zeros_like(value, dtype=np.float32)
    mask = lower != upper
    result[..., mask] = 2.0 * (value[..., mask] - lower[mask]) / (upper[mask] - lower[mask]) - 1.0
    return np.clip(result, -1.0, 1.0)


def _minmax_inverse(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """StarVLA ``Normalizer(mode='min_max').inverse`` semantics."""

    return (value + 1.0) * 0.5 * (upper - lower) + lower


def _require_statistics(block: Mapping[str, Any], name: str, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        lower = np.asarray(block["min"], dtype=np.float32)
        upper = np.asarray(block["max"], dtype=np.float32)
    except KeyError as exc:
        raise Native12ContractError(f"{name} statistics must contain min and max") from exc
    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise Native12ContractError(
            f"{name} statistics must be {dimension}D, got min={lower.shape}, max={upper.shape}"
        )
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise Native12ContractError(f"{name} statistics have invalid min/max values")
    return lower, upper


@dataclass(frozen=True)
class Native12CheckpointArtifacts:
    """The resolved checkpoint and its colocated model contract artifacts."""

    checkpoint_path: Path
    run_dir: Path
    model_config: Mapping[str, Any]
    normalization_statistics: Mapping[str, Any]


class Native12Codec:
    """Exact named-field codec for the MR0 20D-state / native-12 action ABI."""

    def __init__(
        self,
        model_config: Mapping[str, Any],
        normalization_statistics: Mapping[str, Any],
    ) -> None:
        self.model_config = model_config
        self.normalization_statistics = normalization_statistics
        self._validate_checkpoint_abi()
        try:
            block = normalization_statistics[EMBODIMENT_TAG]
            state_statistics = block["state"]
            action_statistics = block["action"]
        except (KeyError, TypeError) as exc:
            raise Native12ContractError(
                f"dataset_statistics.json must contain {EMBODIMENT_TAG!r} state/action statistics"
            ) from exc
        self.state_min, self.state_max = _require_statistics(
            state_statistics, "state", RAW_STATE_DIM
        )
        self.action_min, self.action_max = _require_statistics(
            action_statistics, "action", ACTION_DIM
        )

    @classmethod
    def load_artifacts(cls, checkpoint_path: str | Path) -> Native12CheckpointArtifacts:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
        if checkpoint.suffix not in {".pt", ".safetensors"}:
            raise Native12ContractError(f"checkpoint must be .pt or .safetensors, got {checkpoint.name}")
        run_dir = checkpoint.parents[2] if checkpoint.parent.name.startswith("steps_") else checkpoint.parents[1]
        config_path = run_dir / "config.full.yaml"
        stats_path = run_dir / "dataset_statistics.json"
        if not config_path.is_file() or not stats_path.is_file():
            raise FileNotFoundError(
                "checkpoint must live under <run>/final_model or <run>/checkpoints and be accompanied by "
                f"{config_path.name} and {stats_path.name} in {run_dir}"
            )
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("PyYAML is required to load StarVLA config.full.yaml") from exc
        with config_path.open() as handle:
            model_config = yaml.safe_load(handle)
        with stats_path.open() as handle:
            normalization_statistics = json.load(handle)
        if not isinstance(model_config, Mapping) or not isinstance(normalization_statistics, Mapping):
            raise Native12ContractError("checkpoint config/statistics must be mapping objects")
        return Native12CheckpointArtifacts(
            checkpoint_path=checkpoint,
            run_dir=run_dir,
            model_config=model_config,
            normalization_statistics=normalization_statistics,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
    ) -> "Native12Codec":
        artifacts = cls.load_artifacts(checkpoint_path)
        return cls(
            artifacts.model_config,
            artifacts.normalization_statistics,
        )

    def _validate_checkpoint_abi(self) -> None:
        try:
            action_model = self.model_config["framework"]["action_model"]
            data_config = self.model_config["datasets"]["vla_data"]
        except (KeyError, TypeError) as exc:
            raise Native12ContractError(
                "checkpoint config must contain framework.action_model and datasets.vla_data"
            ) from exc
        expected = {
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "state_placement": "condition",
        }
        for key, value in expected.items():
            if action_model.get(key) != value:
                raise Native12ContractError(
                    f"MR0 native12 evaluation requires action_model.{key}={value!r}, "
                    f"got {action_model.get(key)!r}"
                )
        checkpoint_horizon = action_model.get("action_horizon")
        if (
            isinstance(checkpoint_horizon, bool)
            or not isinstance(checkpoint_horizon, (int, np.integer))
            or int(checkpoint_horizon) < 1
        ):
            raise Native12ContractError(
                "framework.action_model.action_horizon must be a positive integer, "
                f"got {checkpoint_horizon!r}"
            )
        self.action_horizon = int(checkpoint_horizon)
        future_window = action_model.get("future_action_window_size")
        if future_window is not None and future_window != self.action_horizon - 1:
            raise Native12ContractError(
                "framework.action_model.future_action_window_size must equal "
                f"action_horizon - 1 ({self.action_horizon - 1}), got {future_window!r}"
            )
        data_horizon = data_config.get("action_horizon")
        if data_horizon is not None and data_horizon != self.action_horizon:
            raise Native12ContractError(
                "datasets.vla_data.action_horizon must match "
                f"framework.action_model.action_horizon={self.action_horizon}, got {data_horizon!r}"
            )
        expected_data = {"action_mode": "abs", "normalization_mode": "min_max"}
        for key, value in expected_data.items():
            if data_config.get(key) != value:
                raise Native12ContractError(
                    f"MR0 native12 evaluation requires datasets.vla_data.{key}={value!r}, "
                    f"got {data_config.get(key)!r}"
                )

        if self.model_config.get("framework", {}).get("cut3r", {}).get("enabled", False):
            raise Native12ContractError("CUT3R checkpoints are outside the supported MR0 recipes")

        framework_history = self.model_config.get("framework", {}).get("obs_history_indices")
        data_history = data_config.get("obs_history_indices")
        if framework_history is None and data_history is None:
            self.obs_history_indices = None
        else:
            if framework_history is None or data_history is None:
                raise Native12ContractError(
                    "obs_history_indices must be set in both framework and datasets.vla_data, "
                    f"got framework={framework_history!r}, datasets.vla_data={data_history!r}"
                )
            indices = [int(index) for index in framework_history]
            if [int(index) for index in data_history] != indices:
                raise Native12ContractError(
                    f"training obs-history mismatch: framework={indices}, datasets={list(data_history)}"
                )
            if not indices or indices[-1] != 0 or any(index > 0 for index in indices):
                raise Native12ContractError(
                    f"obs_history_indices must be non-empty, non-positive and end at 0, got {indices}"
                )
            self.obs_history_indices = tuple(indices)

        self.state_obs_history = bool(data_config.get("state_obs_history", False))
        if self.state_obs_history and self.obs_history_indices is None:
            raise Native12ContractError(
                "datasets.vla_data.state_obs_history requires obs_history_indices to be set"
            )

    def encode_state(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Convert online named state fields to normalized ``[1, 20]`` state."""

        try:
            fields = {
                key: _as_last_frame(observation[key], _RAW_STATE_SLICES[key].stop - _RAW_STATE_SLICES[key].start, key)
                for key in STATE_KEYS
            }
        except KeyError as exc:
            raise Native12ContractError(f"online observation is missing {exc.args[0]!r}") from exc

        encoded = self._encode_state_fields(fields)
        return encoded[None, :]

    def _encode_state_fields(self, fields: Mapping[str, np.ndarray]) -> np.ndarray:
        """Encode one frame's named raw fields into the normalized 20D state."""

        raw_state = np.concatenate([fields[key] for key in STATE_KEYS], axis=-1)
        if raw_state.shape != (RAW_STATE_DIM,):
            raise AssertionError(f"internal raw-state assembly mismatch: {raw_state.shape}")
        eef_rot6d = quaternion_wxyz_to_rotation_6d(fields["state.end_effector_rotation_relative"])
        base_rot6d = quaternion_wxyz_to_rotation_6d(fields["state.base_rotation"])

        encoded = np.concatenate(
            [
                _minmax_forward(
                    fields["state.end_effector_position_relative"],
                    self.state_min[_RAW_STATE_SLICES["state.end_effector_position_relative"]],
                    self.state_max[_RAW_STATE_SLICES["state.end_effector_position_relative"]],
                ),
                # Converted rotations use StarVLA's fixed min/max [-1, 1].
                np.clip(eef_rot6d, -1.0, 1.0),
                _minmax_forward(
                    fields["state.gripper_qpos"],
                    self.state_min[_RAW_STATE_SLICES["state.gripper_qpos"]],
                    self.state_max[_RAW_STATE_SLICES["state.gripper_qpos"]],
                ),
                _minmax_forward(
                    fields["state.base_position"],
                    self.state_min[_RAW_STATE_SLICES["state.base_position"]],
                    self.state_max[_RAW_STATE_SLICES["state.base_position"]],
                ),
                np.clip(base_rot6d, -1.0, 1.0),
            ],
            axis=-1,
        ).astype(np.float32, copy=False)
        if encoded.shape != (STATE_DIM,) or not np.isfinite(encoded).all():
            raise Native12ContractError(f"encoded state must be finite {STATE_DIM}D, got {encoded.shape}")
        return encoded

    def encode_state_history(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Encode the per-frame state window into ``[T, 20]`` (Arm B)."""

        history_length = len(self.obs_history_indices)
        try:
            raw_fields = {key: np.asarray(observation[key]) for key in STATE_KEYS}
        except KeyError as exc:
            raise Native12ContractError(f"online observation is missing {exc.args[0]!r}") from exc
        frames = []
        for index in range(history_length):
            fields = {}
            for key in STATE_KEYS:
                dimension = _RAW_STATE_SLICES[key].stop - _RAW_STATE_SLICES[key].start
                array = raw_fields[key]
                if array.ndim != 2 or array.shape != (history_length, dimension):
                    raise Native12ContractError(
                        f"{key} must be [T,{dimension}] with T={history_length} for state-history "
                        f"evaluation (wrapper state_delta_indices must equal obs_history_indices), "
                        f"got {array.shape}"
                    )
                if not np.isfinite(array[index]).all():
                    raise Native12ContractError(f"{key} contains NaN or Inf")
                fields[key] = array[index]
            frames.append(self._encode_state_fields(fields))
        return np.stack(frames, axis=0)

    def build_model_example(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Build the exact StarVLA prediction sample from a Gym observation."""

        try:
            if self.obs_history_indices is not None:
                images = [
                    _as_video_frames(observation[key], key, len(self.obs_history_indices))
                    for key in VIDEO_KEYS
                ]
            else:
                images = [_as_last_image(observation[key], key) for key in VIDEO_KEYS]
            language = observation["annotation.human.task_description"]
        except KeyError as exc:
            raise Native12ContractError(f"online observation is missing {exc.args[0]!r}") from exc
        if not isinstance(language, str):
            language = str(language)
        state = (
            self.encode_state_history(observation)
            if self.state_obs_history
            else self.encode_state(observation)
        )
        example = {"image": images, "lang": language, "state": state}
        return example

    def decode_actions(self, normalized_actions: Any) -> dict[str, np.ndarray]:
        """Decode StarVLA normalized output into named native12 controller actions."""

        actions = np.asarray(normalized_actions, dtype=np.float32)
        if actions.shape == (1, self.action_horizon, ACTION_DIM):
            actions = actions[0]
        if actions.shape != (self.action_horizon, ACTION_DIM):
            raise Native12ContractError(
                f"model output must have shape [1,{self.action_horizon},{ACTION_DIM}] or "
                f"[{self.action_horizon},{ACTION_DIM}], got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise Native12ContractError("model output contains NaN or Inf")

        decoded = np.empty_like(actions, dtype=np.float32)
        decoded[:, _CONTINUOUS_ACTION_INDICES] = _minmax_inverse(
            actions[:, _CONTINUOUS_ACTION_INDICES],
            self.action_min[_CONTINUOUS_ACTION_INDICES],
            self.action_max[_CONTINUOUS_ACTION_INDICES],
        )
        decoded[:, _BINARY_ACTION_INDICES] = (
            actions[:, _BINARY_ACTION_INDICES] > 0.5
        ).astype(np.float32)
        return {
            key: decoded[:, action_slice].copy()
            for key, action_slice in _ACTION_SLICES.items()
        }


def _decode_action_batch(
    codec: Native12Codec,
    normalized_actions: Any,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Decode a model ``[B, H, 12]`` result for a vectorized RoboCasa step."""

    actions = np.asarray(normalized_actions, dtype=np.float32)
    expected_shape = (batch_size, codec.action_horizon, ACTION_DIM)
    if actions.shape != expected_shape:
        raise Native12ContractError(
            f"batched model output must have shape {expected_shape}, got {actions.shape}"
        )
    decoded_per_env = [codec.decode_actions(actions[env_idx]) for env_idx in range(batch_size)]
    return {
        key: np.stack([decoded[key] for decoded in decoded_per_env], axis=0)
        for key in ACTION_KEYS
    }


class StarVLANative12Policy:
    """GPU policy that couples a validated MR0 checkpoint with ``Native12Codec``."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        precision: str = "bf16",
    ):
        artifacts = Native12Codec.load_artifacts(checkpoint_path)
        self.codec = Native12Codec(
            artifacts.model_config,
            artifacts.normalization_statistics,
        )
        self.checkpoint_path = artifacts.checkpoint_path
        try:
            import torch
            from starVLA.model.framework.base_framework import baseframework
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "StarVLA runtime dependencies are unavailable. Use the StarVLA training environment."
            ) from exc
        if precision not in {"bf16", "fp32"}:
            raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r}, but CUDA is unavailable")
        model = baseframework.from_pretrained(str(artifacts.checkpoint_path))
        if precision == "bf16":
            model = model.to(torch.bfloat16)
        self.model = model.to(device).eval()

    def predict_action_chunk(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Predict and decode one full native-12 action horizon from a Gym observation."""

        example = self.codec.build_model_example(observation)
        output = self.model.predict_action(examples=[example])
        try:
            normalized_actions = output["normalized_actions"]
        except KeyError as exc:
            raise Native12ContractError("StarVLA predict_action did not return normalized_actions") from exc
        return self.codec.decode_actions(normalized_actions)

    def predict_action_chunks(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        """Predict one native-12 chunk per environment in a vectorized rollout."""

        if not observations:
            raise ValueError("observations must not be empty")
        examples = [self.codec.build_model_example(observation) for observation in observations]
        output = self.model.predict_action(examples=examples)
        try:
            normalized_actions = output["normalized_actions"]
        except KeyError as exc:
            raise Native12ContractError("StarVLA predict_action did not return normalized_actions") from exc
        return _decode_action_batch(self.codec, normalized_actions, len(examples))

    def close(self) -> None:
        """Keep the local and remote policy interfaces interchangeable."""


def _load_package_from_site(package_name: str, site_packages: str | Path) -> None:
    """Load one package from a foreign environment without altering ``sys.path``.

    The simulation environment intentionally has no Qwen dependency.  On a
    machine where it lacks StarVLA's pure-Python websocket/msgpack packages,
    loading only those packages avoids shadowing its MuJoCo, NumPy, or
    Transformers installation with an entire second conda environment.
    """

    if package_name in sys.modules:
        return
    package_dir = Path(site_packages) / package_name
    init_path = package_dir / "__init__.py"
    if not init_path.is_file():
        raise ModuleNotFoundError(f"{package_name!r} was not found under {site_packages}")
    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not construct an import spec for {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


class RemoteStarVLANative12Policy:
    """Native12 codec plus a StarVLA WebSocket model server client.

    This follows Groot's server/client separation: QwenMMDiT stays in the
    StarVLA environment while RoboCasa/MuJoCo stays in the Groot environment.
    The wire payload is StarVLA's existing ``examples`` API and carries only
    normalized model inputs/outputs, never environment actions.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        host: str,
        port: int,
        websocket_site_packages: str | Path | None = None,
    ) -> None:
        artifacts = Native12Codec.load_artifacts(checkpoint_path)
        self.codec = Native12Codec(
            artifacts.model_config,
            artifacts.normalization_statistics,
        )
        self.checkpoint_path = artifacts.checkpoint_path
        try:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        except ModuleNotFoundError as exc:
            if exc.name not in {"websockets", "msgpack"} or websocket_site_packages is None:
                raise ModuleNotFoundError(
                    "The evaluation client needs StarVLA's websocket/msgpack packages. "
                    "Set --websocket-site-packages to the StarVLA environment's site-packages directory."
                ) from exc
            # ``msgpack_numpy`` imports msgpack, and the client imports websockets.
            _load_package_from_site("msgpack", websocket_site_packages)
            _load_package_from_site("websockets", websocket_site_packages)
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        self._client = WebsocketClientPolicy(host=host, port=port)

    def predict_action_chunk(self, observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
        example = self.codec.build_model_example(observation)
        response = self._client.predict_action({"examples": [example]})
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA model server returned an error: {response}")
        try:
            normalized_actions = response["data"]["normalized_actions"]
        except KeyError as exc:
            raise Native12ContractError(
                f"StarVLA model server response lacks normalized_actions: {response.keys()}"
            ) from exc
        return self.codec.decode_actions(normalized_actions)

    def predict_action_chunks(self, observations: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
        """Request a single model batch for all live RoboCasa vector environments."""

        if not observations:
            raise ValueError("observations must not be empty")
        examples = [self.codec.build_model_example(observation) for observation in observations]
        response = self._client.predict_action({"examples": examples})
        if not response.get("ok", False):
            raise RuntimeError(f"StarVLA model server returned an error: {response}")
        try:
            normalized_actions = response["data"]["normalized_actions"]
        except KeyError as exc:
            raise Native12ContractError(
                f"StarVLA model server response lacks normalized_actions: {response.keys()}"
            ) from exc
        return _decode_action_batch(self.codec, normalized_actions, len(examples))

    def close(self) -> None:
        self._client.close()


class RemoteStarVLAOfficialPolicy:
    """Expose StarVLA through the policy interface used by official GR00T eval."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        host: str,
        port: int,
        n_action_steps: int,
        websocket_site_packages: str | Path | None = None,
    ) -> None:
        self._policy = RemoteStarVLANative12Policy(
            checkpoint_path,
            host=host,
            port=port,
            websocket_site_packages=websocket_site_packages,
        )
        if n_action_steps < 1:
            raise ValueError("n_action_steps must be positive")
        if n_action_steps > self._policy.codec.action_horizon:
            raise ValueError(
                "n_action_steps cannot exceed the checkpoint action horizon: "
                f"{n_action_steps} > {self._policy.codec.action_horizon}"
            )
        self.n_action_steps = n_action_steps

    def get_action(self, observations: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Convert official batched observations to StarVLA inputs and actions."""

        batch_sizes = {
            np.asarray(value).shape[0]
            for key, value in observations.items()
            if key.startswith(("video.", "state.")) and np.asarray(value).ndim > 0
        }
        if len(batch_sizes) != 1:
            raise Native12ContractError(
                f"official observation batch has inconsistent sizes: {sorted(batch_sizes)}"
            )
        batch_size = batch_sizes.pop()
        per_environment = [
            {key: value[index] for key, value in observations.items()}
            for index in range(batch_size)
        ]
        action_chunks = self._policy.predict_action_chunks(per_environment)
        return {
            key: np.asarray(value)[:, : self.n_action_steps].copy()
            for key, value in action_chunks.items()
        }

    def get_modality_config(self) -> dict[str, tuple[str, ...]]:
        """Return the StarVLA native-12 modalities for the official info query."""

        return {
            "video": VIDEO_KEYS,
            "state": STATE_KEYS,
            "action": ACTION_KEYS,
        }

    def close(self) -> None:
        self._policy.close()
