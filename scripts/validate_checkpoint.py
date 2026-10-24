"""Validate every checkpoint tensor against a metadata-only model; no GPU allocation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-vlm", required=True)
    args = parser.parse_args()
    import torch
    from accelerate import init_empty_weights
    from omegaconf import OmegaConf
    from starVLA.model.framework.base_framework import build_framework

    config = OmegaConf.load(args.config)
    config.framework.qwenvl.base_vlm = str(Path(args.base_vlm).resolve())
    config.framework.qwenvl.checkpoint_contains_vlm = True
    # Attention implementation does not affect parameter names or shapes.
    config.framework.qwenvl.attn_implementation = "sdpa"
    with init_empty_weights(include_buffers=True):
        model = build_framework(config)
    if Path(args.checkpoint).suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
            shapes = {key: tuple(handle.get_slice(key).get_shape()) for key in handle.keys()}
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
        if not all(torch.is_tensor(value) for value in state.values()):
            raise ValueError("Expected a tensor-only state dict")
        shapes = {key: tuple(value.shape) for key, value in state.items()}
    expected = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    report = {"expected_tensors": len(expected), "checkpoint_tensors": len(shapes),
              "missing": sorted(set(expected)-set(shapes)), "unexpected": sorted(set(shapes)-set(expected)),
              "shape_mismatches": sorted(key for key in set(expected)&set(shapes) if expected[key] != shapes[key])}
    report["passed"] = not any(report[key] for key in ["missing", "unexpected", "shape_mismatches"])
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
