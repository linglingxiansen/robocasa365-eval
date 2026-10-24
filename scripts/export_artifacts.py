"""Export portable inference metadata and optional weights into an ignored directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sanitize_json(value):
    """Drop cached location metadata, retaining tokenizer/model semantics."""
    if isinstance(value, dict):
        return {key: sanitize_json(item) for key, item in value.items()
                if key not in {"_name_or_path", "name_or_path", "tokenizer_file", "special_tokens_map_file"}}
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export(checkpoint, base_vlm, output, *, copy_weights=False):
    import yaml

    checkpoint, base_vlm, output = [Path(x).expanduser().resolve() for x in (checkpoint, base_vlm, output)]
    if not checkpoint.is_file() or checkpoint.suffix not in {".pt", ".safetensors"}:
        raise ValueError("Provide an existing .pt or .safetensors checkpoint")
    if not (base_vlm / "config.json").is_file() or not (base_vlm / "tokenizer.json").is_file():
        raise ValueError("Base VLM directory must contain config.json and tokenizer.json")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be absent or empty; existing artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    (output / "base_vlm").mkdir()
    (output / "checkpoints").mkdir()
    config = yaml.safe_load((ROOT / "configs/checkpoint_200k.yaml").read_text())
    # The packaged policy initializes its architecture from config and restores
    # all learned weights from the trained checkpoint, avoiding a second VLM copy.
    config["framework"]["qwenvl"]["checkpoint_contains_vlm"] = True
    (output / "config.full.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copyfile(ROOT / "configs/dataset_statistics.json", output / "dataset_statistics.json")
    allowed = {".json", ".txt", ".jinja", ".model", ".tiktoken"}
    for path in sorted(base_vlm.iterdir()):
        if path.is_file() and path.suffix in allowed:
            dest = output / "base_vlm" / path.name
            if path.suffix == ".json":
                dest.write_text(json.dumps(sanitize_json(json.loads(path.read_text())), ensure_ascii=False, indent=2)+"\n")
            else:
                shutil.copyfile(path, dest)
    dest = output / "checkpoints" / ("model" + checkpoint.suffix)
    if copy_weights:
        shutil.copyfile(checkpoint, dest)
    # No symlinks to source paths are created; the distribution never stores them.
    checksums = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()}
    if not copy_weights:
        checksums[str(dest.relative_to(output))] = sha256(checkpoint)
    (output / "SHA256SUMS.json").write_text(json.dumps(checksums, indent=2)+"\n")
    print("Exported portable metadata" + (" and checkpoint." if copy_weights else "; copy the original checkpoint to checkpoints/model" + checkpoint.suffix + " before evaluation."))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--output", default="artifacts/200k")
    parser.add_argument("--copy-weights", action="store_true", help="Copy large weights into the private artifact bundle")
    args = parser.parse_args()
    export(args.checkpoint, args.base_vlm, args.output, copy_weights=args.copy_weights)


if __name__ == "__main__":
    main()
