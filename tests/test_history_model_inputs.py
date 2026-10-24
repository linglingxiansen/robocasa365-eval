"""CPU regression checks for video history and state-token conditioning."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from transformers import BatchFeature

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.modules.action_model.MMDiT_ActionHeader import (
    MMDiTFlowmatchingActionHead, ObservationProjection,
)
from starVLA.model.modules.vlm.QWen3_5 import _QWen3_5_VL_Interface


@pytest.mark.parametrize("history", [False, True])
def test_qwen_preserves_camera_frames_and_video_timestamps(history):
    calls = []

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append((messages, kwargs))
            return BatchFeature({"input_ids": torch.zeros((1, 3), dtype=torch.long)})

    interface = _QWen3_5_VL_Interface.__new__(_QWen3_5_VL_Interface)
    torch.nn.Module.__init__(interface)
    interface.model = SimpleNamespace(device=torch.device("cpu"))
    interface.processor = Processor()
    interface.config = OmegaConf.create({
        "framework": {"obs_history_indices": [-6, -4, -2, 0] if history else None,
                      "obs_history_source_fps": 20.0},
        "datasets": {"vla_data": {}},
    })
    cameras = [[np.full((8, 8, 3), 10 * c + t, dtype=np.uint8) for t in range(4)] for c in range(3)]
    images = to_pil_preserve(cameras if history else [camera[-1] for camera in cameras])
    result = interface.build_qwenvl_inputs([images], ["Open the drawer"])
    assert result["input_ids"].device.type == "cpu"
    messages, kwargs = calls[0]
    content = messages[0][0]["content"]
    assert content[-1] == {"type": "text", "text": "Open the drawer"}
    assert kwargs["enable_thinking"] is False
    if history:
        videos = [item["video"] for item in content if item["type"] == "video"]
        assert [[np.asarray(frame)[0, 0, 0] for frame in camera] for camera in videos] == [
            [0, 1, 2, 3], [10, 11, 12, 13], [20, 21, 22, 23],
        ]
        assert kwargs["do_sample_frames"] is False
        assert kwargs["video_metadata"] == [[{
            "total_num_frames": 7, "fps": 20.0, "frames_indices": [0, 2, 4, 6],
        }] * 3]
        assert [item["text"] for item in content[:-1] if item["type"] == "text"] == [
            "Left camera: ", "\nRight camera: ", "\nWrist camera: ",
        ]
    else:
        assert [item["type"] for item in content] == ["image", "image", "image", "text"]
        assert "video_metadata" not in kwargs


@pytest.mark.parametrize("state_tokens", [1, 4])
def test_state_tokens_extend_attention_mask_without_unmasking_padding(state_tokens):
    head = MMDiTFlowmatchingActionHead.__new__(MMDiTFlowmatchingActionHead)
    torch.nn.Module.__init__(head)
    projection = ObservationProjection(vl_feature_dim=8, output_dim=16, state_dim=20)
    state = torch.randn(2, state_tokens, 20)
    observed = projection(torch.randn(2, 3, 8), state)
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    result = head._build_obs_attention_mask(mask, observed, state)
    assert observed.shape == (2, 3 + state_tokens, 16)
    torch.testing.assert_close(result[:, :3], mask)
    torch.testing.assert_close(result[:, 3:], torch.ones((2, state_tokens), dtype=mask.dtype))
    with pytest.raises(ValueError, match="length must match"):
        head._build_obs_attention_mask(mask[:, :2], observed, state)
