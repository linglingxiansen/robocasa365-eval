"""Run the trained policy without importing simulator dependencies."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-vlm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11093)
    args = parser.parse_args()
    if args.base_vlm:
        os.environ["EVAL_BASE_VLM"] = str(Path(args.base_vlm).expanduser().resolve())
    from deployment.model_server.server_policy import main as serve
    serve(argparse.Namespace(ckpt_path=args.checkpoint, host=args.host, port=args.port,
                             use_bf16=True, idle_timeout=-1))


if __name__ == "__main__":
    main()
