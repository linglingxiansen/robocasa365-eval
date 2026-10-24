#!/usr/bin/env python3
"""Run a disjoint task shard through the vendored official evaluator."""

from __future__ import annotations

import argparse
import threading
import time

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

from .run_eval import run_client, run_server


def run_shard(args: argparse.Namespace) -> None:
    shard_name = f"starvla_shard_{args.shard_id}"
    TASK_SET_REGISTRY[shard_name] = list(args.env_names)
    # Checkpoints trained with VLM obs-history declare the frame window in
    # their config; everyone else keeps run_client's default window.
    from .native12_policy import Native12Codec

    model_config = Native12Codec.load_artifacts(args.model_path).model_config
    video_delta_indices = model_config.get("framework", {}).get("obs_history_indices")
    # Arm B: state joins the same history window.
    state_delta_indices = (
        video_delta_indices
        if video_delta_indices is not None
        and model_config.get("datasets", {}).get("vla_data", {}).get("state_obs_history", False)
        else None
    )
    server_thread = threading.Thread(
        target=run_server,
        kwargs={
            "data_config": args.data_config,
            "model_path": args.model_path,
            "embodiment_tag": args.embodiment_tag,
            "port": args.port,
            "starvla_host": args.starvla_host,
            "starvla_port": args.starvla_port,
            "websocket_site_packages": args.websocket_site_packages,
            "n_action_steps": args.n_action_steps,
        },
        daemon=True,
    )
    server_thread.start()
    time.sleep(2)
    run_client(
        host=args.host,
        port=args.port,
        task_set_list=[shard_name],
        video_dir=args.output_dir,
        split=args.split,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        n_action_steps=args.n_action_steps,
        save_video=args.video,
        video_delta_indices=video_delta_indices,
        state_delta_indices=state_delta_indices,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--env-names", nargs="+", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("pretrain", "target"), required=True)
    parser.add_argument("--n-episodes", type=int, required=True)
    parser.add_argument("--n-envs", type=int, required=True)
    parser.add_argument("--n-action-steps", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--starvla-host", default="127.0.0.1")
    parser.add_argument("--starvla-port", type=int, required=True)
    parser.add_argument("--websocket-site-packages", required=True)
    parser.add_argument("--data-config", default="panda_omron")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--video", action="store_true")
    return parser


def main() -> None:
    run_shard(build_parser().parse_args())


if __name__ == "__main__":
    main()
