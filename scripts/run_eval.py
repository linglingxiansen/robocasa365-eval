"""Configure a portable two-environment sharded evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-python", required=True)
    parser.add_argument("--simulation-python", default=sys.executable)
    parser.add_argument("--config", default="configs/eval_8gpu.yaml")
    parser.add_argument("--output", default="outputs/eval_200k")
    parser.add_argument("--base-vlm", help="Optional override; defaults to the artifact bundle's base_vlm directory")
    parser.add_argument("--robocasa-root", help="Optional source checkout when RoboCasa is not installed editable")
    parser.add_argument("--robosuite-root", help="Optional source checkout when robosuite is not installed editable")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    model_python = Path(args.model_python).expanduser().resolve()
    simulation_python = Path(args.simulation_python).expanduser().resolve()
    for python in (model_python, simulation_python):
        if not python.is_file():
            parser.error("Both Python interpreters must exist")
    env = os.environ.copy()
    python_paths = [str(ROOT)]
    for value in (args.robocasa_root, args.robosuite_root):
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_dir():
                parser.error("Simulator source checkouts must be existing directories")
            python_paths.append(str(path))
    env.update(PYTHONPATH=os.pathsep.join(python_paths), STARVLA_PYTHON=str(model_python),
               ROBOCASA_PYTHON=str(simulation_python),
               EVAL_CHECKPOINT=str(Path(args.checkpoint).expanduser().resolve()),
               EVAL_OUTPUT=str(Path(args.output).expanduser().resolve()))
    if args.base_vlm:
        env["EVAL_BASE_VLM"] = str(Path(args.base_vlm).expanduser().resolve())
    probe = subprocess.run([str(model_python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                           check=True, capture_output=True, text=True)
    env["ROBOCASA_WEBSOCKET_SITE_PACKAGES"] = probe.stdout.strip()
    command = [str(simulation_python), "-m", "examples.robocasa365.eval_files.official.launch_sharded_eval",
               str(Path(args.config).expanduser().resolve())]
    if args.dry_run:
        command.append("--dry-run")
    elif args.background:
        command.append("--background")
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode or args.dry_run or args.background:
        raise SystemExit(result.returncode)
    raise SystemExit(subprocess.run([str(simulation_python), "-m", "examples.robocasa365.eval_files.compute_metrics",
                                    env["EVAL_OUTPUT"]], cwd=ROOT, env=env).returncode)


if __name__ == "__main__":
    main()
