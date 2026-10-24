# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import json
import numpy as np
import threading
import time
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

from examples.robocasa365.eval_files.runtime.robot import RobotInferenceServer
from .native12_policy import RemoteStarVLAOfficialPolicy
from .simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)


def run_server(
    data_config,
    model_path,
    embodiment_tag,
    port,
    starvla_host,
    starvla_port,
    websocket_site_packages,
    n_action_steps,
):
    # The only project-specific change: adapt StarVLA to the official policy API.
    policy = RemoteStarVLAOfficialPolicy(
        checkpoint_path=model_path,
        host=starvla_host,
        port=starvla_port,
        websocket_site_packages=websocket_site_packages,
        n_action_steps=n_action_steps,
    )

    # Start the server
    server = RobotInferenceServer(policy, host="127.0.0.1", port=port)
    server.run()


def run_client(
    host,
    port,
    task_set_list,
    video_dir,
    split,
    n_episodes,
    n_envs,
    n_action_steps,
    save_video,
    video_delta_indices=None,
    state_delta_indices=None,
):
    # Create a simulation client
    simulation_client = SimulationInferenceClient(host=host, port=port)

    print("Available modality configs:")
    modality_config = simulation_client.get_modality_config()
    print(modality_config.keys())

    all_env_names = []
    for task_set in task_set_list:
        all_env_names += TASK_SET_REGISTRY[task_set]
    # turn into unique list
    all_env_names = set(all_env_names)

    for env_name in all_env_names:
        this_video_dir = os.path.join(video_dir, "evals", split, env_name)

        stats_path = os.path.join(this_video_dir, "stats.json")
        if os.path.exists(stats_path):
            print(f"{env_name} stats already exsits. skipping.")
            continue
        horizon = get_task_horizon(env_name)
        # Create simulation configuration
        config = SimulationConfig(
            env_name=f"robocasa/{env_name}",
            split=split,
            n_episodes=n_episodes,
            n_envs=n_envs,
            video=VideoConfig(video_dir=this_video_dir if save_video else None),
            multistep=MultiStepConfig(
                video_delta_indices=(
                    np.array([-3, -2, -1, 0])
                    if video_delta_indices is None
                    else np.asarray(list(video_delta_indices), dtype=int)
                ),
                state_delta_indices=(
                    np.array([0])
                    if state_delta_indices is None
                    else np.asarray(list(state_delta_indices), dtype=int)
                ),
                n_action_steps=n_action_steps, max_episode_steps=horizon,
            ),
        )

        # Run the simulation
        print(f"Running simulation for {env_name}...")
        try:
            env_name, episode_successes = simulation_client.run_simulation(config)
        except Exception as e:
            print("Exception!", e)
            continue

        # Print results
        success_rate = np.mean(episode_successes)

        print(f"Results for {env_name}:")
        print(f"Success rate: {success_rate:.2f}")

        os.makedirs(this_video_dir, exist_ok=True)
        with open(stats_path, "w") as f:
            stats = {
                "num_episodes": len(episode_successes),
                "success_rate": success_rate,
            }
            json.dump(stats, f, indent=4)
        print(f"saved stats to {stats_path}")

        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="<PATH_TO_YOUR_MODEL>",  # change this to your model path
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        default="panda_omron",  # change this to your embodiment tag
    )
    parser.add_argument(
        "--task_set",
        type=str,
        nargs='+',
        help="Name of the task soup(s)",
        required=True,
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Split to evaluate on. Can be either pretrain or target.",
        choices=["pretrain", "target"],
        required=True,
    )
    parser.add_argument("--port", type=int, help="Port number for the server.", default=5555)
    parser.add_argument(
        "--host", type=str, help="Host address for the server.", default="localhost"
    )
    parser.add_argument("--starvla_host", type=str, default="127.0.0.1")
    parser.add_argument("--starvla_port", type=int, default=10093)
    parser.add_argument("--websocket_site_packages", type=str, default=None)
    parser.add_argument("--video_dir", type=str, help="Directory to save videos.", default=None)
    parser.add_argument("--video", action="store_true", help="Record evaluation videos.")
    parser.add_argument("--n_episodes", type=int, help="Number of episodes to run.", default=50)
    parser.add_argument("--n_envs", type=int, help="Number of parallel environments.", default=5)
    parser.add_argument(
        "--n_action_steps",
        type=int,
        help="Number of action steps per environment step.",
        default=16,
    )
    # server mode
    parser.add_argument("--server", action="store_true", help="Run the server.")
    # client mode
    parser.add_argument("--client", action="store_true", help="Run the client")
    args = parser.parse_args()

    if args.server:
        run_server(
            data_config=args.data_config,
            model_path=args.model_path,
            embodiment_tag=args.embodiment_tag,
            port=args.port,
            starvla_host=args.starvla_host,
            starvla_port=args.starvla_port,
            websocket_site_packages=args.websocket_site_packages,
            n_action_steps=args.n_action_steps,
        )
    elif args.client:
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir or args.model_path,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            save_video=args.video,
        )
    else:
        server_thread = threading.Thread(
            target=run_server,
            args=(
                args.data_config,
                args.model_path,
                args.embodiment_tag,
                args.port,
                args.starvla_host,
                args.starvla_port,
                args.websocket_site_packages,
                args.n_action_steps,
            ),
            daemon=True
        )
        server_thread.start()
        time.sleep(1)  # give server time to start
        run_client(
            host=args.host,
            port=args.port,
            task_set_list=args.task_set,
            video_dir=args.video_dir or args.model_path,
            split=args.split,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            save_video=args.video,
        )
