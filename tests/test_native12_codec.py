import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.robocasa365.eval_files.official.native12_policy import (
    Native12Codec, Native12ContractError, VIDEO_KEYS,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def codec():
    return Native12Codec(yaml.safe_load((ROOT / "configs/checkpoint_200k.yaml").read_text()),
                         json.loads((ROOT / "configs/dataset_statistics.json").read_text()))


def observation():
    fields = {
        "state.end_effector_position_relative": [0.0, 0.0, 0.0],
        "state.end_effector_rotation_relative": [1.0, 0.0, 0.0, 0.0],
        "state.gripper_qpos": [0.0, 0.0],
        "state.base_position": [0.0, 0.0, 0.0],
        "state.base_rotation": [1.0, 0.0, 0.0, 0.0],
    }
    obs = {key: np.tile(value, (4, 1)).astype(np.float32) for key, value in fields.items()}
    obs.update({key: np.stack([np.full((8, 8, 3), camera*10+frame, dtype=np.uint8)
                              for frame in range(4)]) for camera, key in enumerate(VIDEO_KEYS)})
    obs["annotation.human.task_description"] = "Prepare coffee."
    return obs


def test_codec_preserves_camera_order_and_all_four_frames(codec):
    sample = codec.build_model_example(observation())
    assert sample["state"].shape == (4, 20)
    assert np.isfinite(sample["state"]).all()
    assert sample["lang"] == "Prepare coffee."
    assert len(sample["image"]) == 3
    for camera, images in enumerate(sample["image"]):
        assert len(images) == 4
        for frame, image in enumerate(images):
            assert np.asarray(image)[0, 0, 0] == camera*10+frame


def test_codec_rejects_partial_state_history(codec):
    obs = observation()
    obs["state.base_position"] = obs["state.base_position"][-1:]
    with pytest.raises(Native12ContractError):
        codec.build_model_example(obs)


def test_codec_rejects_nonfinite_action(codec):
    actions = np.zeros((50, 12), dtype=np.float32)
    actions[0, 0] = np.nan
    with pytest.raises(Native12ContractError):
        codec.decode_actions(actions)


def test_action_decode_has_named_native_fields_and_binary_controls(codec):
    low = codec.decode_actions(np.zeros((50, 12), dtype=np.float32))
    high = codec.decode_actions(np.ones((50, 12), dtype=np.float32))
    assert low["action.end_effector_position"].shape == (50, 3)
    assert high["action.base_motion"].shape == (50, 4)
    for key in ["action.gripper_close", "action.control_mode"]:
        assert set(np.unique(low[key])) <= {0, 1}
        assert set(np.unique(high[key])) <= {0, 1}
        assert np.all(high[key] == 1)
