"""Native12 mode weighting with fixed gripper/mode weights and total weight 12."""
from __future__ import annotations
import torch
from .control_mode_weighted_loss import validate_mode_weighting


def control_mode_balanced_weights(clean_actions, inactive_weight=0.1):
    if clean_actions.ndim != 3:
        raise ValueError("Expected clean actions [B,T,12]")
    validate_mode_weighting(clean_actions.shape[-1], inactive_weight)
    navigation = clean_actions[..., 11:12].detach() > 0.5
    weights = torch.ones_like(clean_actions, dtype=torch.float32)
    # Gripper [6] and mode [11] retain 1 each. Allocate the other 10 units
    # between six arm dimensions and four base/torso dimensions.
    arm_active = (10.0 - 4.0 * inactive_weight) / 6.0
    base_active = (10.0 - 6.0 * inactive_weight) / 4.0
    weights[..., :6] = torch.where(navigation, inactive_weight, arm_active)
    weights[..., 7:11] = torch.where(navigation, base_active, inactive_weight)
    return weights


def control_mode_balanced_velocity_mse(pred_velocity, velocity_target, clean_actions,
                                      action_mask=None, *, inactive_weight=0.1):
    if pred_velocity.shape != velocity_target.shape or pred_velocity.shape != clean_actions.shape:
        raise ValueError("Prediction, velocity target and clean actions must have the same shape")
    if pred_velocity.device != velocity_target.device or pred_velocity.device != clean_actions.device:
        raise ValueError("Prediction and targets must have the same device")
    weights = control_mode_balanced_weights(clean_actions, inactive_weight)
    squared_error = (pred_velocity.float() - velocity_target.float()).square()
    if action_mask is None:
        return (weights * squared_error).mean()
    mask = action_mask.to(device=pred_velocity.device, dtype=torch.bool)
    if mask.shape == (pred_velocity.shape[0], 12):
        mask = mask[:, None, :].expand_as(pred_velocity)
    elif mask.shape != pred_velocity.shape:
        raise ValueError("action_mask must be [B,12] or [B,T,12]")
    return (weights * squared_error).masked_fill(~mask, 0.0).sum() / mask.sum().clamp_min(1)
