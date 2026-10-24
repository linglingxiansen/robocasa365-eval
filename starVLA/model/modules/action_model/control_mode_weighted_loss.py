"""Native12 loss weighting from the clean, per-timestep control-mode target.

Layout: EEF pose [0:6], gripper [6], base/torso [7:11], mode [11].
Mode 1 is navigation; mode 0 is manipulation. The mode dimension always has
weight 1. Normalize by valid element count, not the sum of weights: changing
an inactive weight to 0.1 must reduce that element's gradient by exactly 10x.
This module has no model parameters and does not change inference.
"""
from __future__ import annotations

import math

import torch


def validate_mode_weighting(action_dim: int, inactive_weight: float) -> None:
    if action_dim != 12:
        raise ValueError("native12_control_mode_weighted requires action_dim=12")
    if not math.isfinite(inactive_weight) or not 0 <= inactive_weight <= 1:
        raise ValueError("inactive_action_weight must be finite and in [0, 1]")


def control_mode_weighted_velocity_mse(
    pred_velocity: torch.Tensor,
    velocity_target: torch.Tensor,
    clean_actions: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    *,
    inactive_weight: float = 0.1,
) -> torch.Tensor:
    """Weight velocity errors using ground-truth mode separately at every step.

    All action tensors must be [B,T,12]. clean_actions are normalized dataset
    targets, before adding noise; never use the noisy or predicted mode here.
    Optional bool masks may be [B,12] or [B,T,12]. Padded entries contribute
    neither to the numerator nor the denominator. Accumulate errors in fp32.
    """
    if pred_velocity.ndim != 3:
        raise ValueError("Expected action tensors with shape [B,T,12]")
    validate_mode_weighting(pred_velocity.shape[-1], inactive_weight)
    if velocity_target.shape != pred_velocity.shape or clean_actions.shape != pred_velocity.shape:
        raise ValueError("Prediction, velocity target and clean actions must have the same shape")
    if clean_actions.device != pred_velocity.device or velocity_target.device != pred_velocity.device:
        raise ValueError("Prediction and targets must be on the same device")

    navigation = clean_actions[..., 11:12].detach() > 0.5
    weights = torch.ones_like(pred_velocity, dtype=torch.float32)
    weights[..., :7] = torch.where(navigation, inactive_weight, 1.0)
    weights[..., 7:11] = torch.where(navigation, 1.0, inactive_weight)
    squared_error = (pred_velocity.float() - velocity_target.float()).square()
    if action_mask is None:
        return (squared_error * weights).mean()

    mask = action_mask.to(device=pred_velocity.device, dtype=torch.bool)
    if mask.shape == (pred_velocity.shape[0], 12):
        mask = mask[:, None, :].expand_as(pred_velocity)
    elif mask.shape != pred_velocity.shape:
        raise ValueError("action_mask must have shape [B,12] or [B,T,12]")
    return (squared_error * weights).masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)
