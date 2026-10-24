"""CPU checks for exact state/RGB history and transport contracts."""
import numpy as np
import pytest
pytest.importorskip("gymnasium")
pytest.importorskip("zmq")
from examples.robocasa365.eval_files.runtime.contracts import ModalityConfig
from examples.robocasa365.eval_files.runtime.service import TorchSerializer
from examples.robocasa365.eval_files.official.wrappers.multistep_wrapper import MultiStepWrapper

def test_modality_contract_survives_service_serialization():
    config = ModalityConfig(delta_indices=[-6, -4, -2, 0], modality_keys=["state.base_position"])
    restored = TorchSerializer.from_bytes(TorchSerializer.to_bytes({"state": config}))
    assert restored["state"] == config


@pytest.mark.parametrize("n_action_steps", [1, 16])
def test_history_wrapper_repeats_first_frame_and_samples_stride_two(n_action_steps):
    import gymnasium as gym

    class ClockEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "state.clock": gym.spaces.Box(0, 100, (1,), dtype=np.float32),
            "video.clock": gym.spaces.Box(0, 255, (2, 2, 3), dtype=np.uint8),
        })
        action_space = gym.spaces.Dict({"action.clock": gym.spaces.Box(-1, 1, (1,), dtype=np.float32)})

        def reset(self, *, seed=None, options=None):
            self.tick = 0
            return self.observation(), {}

        def step(self, action):
            self.tick += 1
            return self.observation(), 0, False, False, {}

        def observation(self):
            return {"state.clock": np.array([self.tick], dtype=np.float32),
                    "video.clock": np.full((2, 2, 3), self.tick, dtype=np.uint8)}

    indices = np.array([-6, -4, -2, 0])
    env = MultiStepWrapper(ClockEnv(), indices, indices, n_action_steps)
    obs, _ = env.reset()
    np.testing.assert_array_equal(obs["state.clock"].ravel(), [0, 0, 0, 0])
    for _ in range(6):
        obs, *_ = env.step({"action.clock": np.zeros((n_action_steps, 1), dtype=np.float32)})
    expected = 6 * n_action_steps + indices
    np.testing.assert_array_equal(obs["state.clock"].ravel(), expected)
    np.testing.assert_array_equal(obs["video.clock"][:, 0, 0, 0], expected)
    env.close()
