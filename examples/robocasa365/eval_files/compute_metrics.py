"""Compute and export success metrics for a complete sharded RoboCasa365 run."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .official.get_eval_stats import TASK_GROUP_MAPPING, validate_complete_results


def summarize(rows):
    episodes = sum(row["num_episodes"] for row in rows)
    successes = sum(row["num_successes"] for row in rows)
    return {
        "num_tasks": len(rows),
        "num_episodes": episodes,
        "num_successes": successes,
        "success_rate": successes / episodes,
        "mean_task_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
    }


def compute_metrics(output_dir):
    output_dir = Path(output_dir).expanduser().resolve()
    manifest = json.loads((output_dir / "shard_manifest.json").read_text())
    split = manifest["split"]
    if split not in {"pretrain", "target"}:
        raise ValueError(f"Unsupported split: {split!r}")
    groups = manifest["task_set"]
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("Manifest must contain nonempty, unique task groups")
    unknown = set(groups) - set(TASK_GROUP_MAPPING)
    if unknown:
        raise ValueError(f"Unknown task groups: {sorted(unknown)}")
    episodes = manifest["n_episodes"]
    if type(episodes) is not int or episodes < 1:
        raise ValueError("Manifest n_episodes must be a positive integer")
    expected = {task for group in groups for task in TASK_GROUP_MAPPING[group]}
    assigned = [task for worker in manifest["workers"] for task in worker["env_names"]]
    if (len(assigned) != len(set(assigned)) or set(assigned) != expected
            or manifest["task_count"] != len(expected)):
        raise ValueError("Manifest shard tasks do not match the selected task groups")
    actual = {p.parent.name for p in (output_dir / "evals" / split).glob("*/stats.json")}
    if actual != expected:
        raise ValueError(f"Incomplete or mixed results: missing={sorted(expected - actual)}, "
                         f"unexpected={sorted(actual - expected)}")
    validate_complete_results(output_dir, groups, split=split, episodes=episodes)
    rows = []
    for task in sorted(expected):
        stats = json.loads((output_dir / "evals" / split / task / "stats.json").read_text())
        success_count = stats["success_rate"] * episodes
        if not math.isclose(success_count, round(success_count), abs_tol=1e-6, rel_tol=0):
            raise ValueError(f"Success rate for {task} does not represent an integer episode count")
        if "episode_successes" in stats:
            flags = stats["episode_successes"]
            if len(flags) != episodes or sum(flags) != round(success_count):
                raise ValueError(f"Episode successes disagree with summary for {task}")
        rows.append({
            "task": task,
            "task_group": ",".join(group for group in groups if task in TASK_GROUP_MAPPING[group]),
            "split": split,
            "num_episodes": episodes,
            "num_successes": round(success_count),
            "success_rate": stats["success_rate"],
        })
    return {
        "checkpoint": manifest["checkpoint"],
        "output_dir": str(output_dir),
        "split": split,
        "complete": True,
        "overall": summarize(rows),
        "groups": {group: summarize([row for row in rows if row["task"] in TASK_GROUP_MAPPING[group]])
                   for group in groups},
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", help="Run directory containing shard_manifest.json and evals/")
    args = parser.parse_args()
    try:
        metrics = compute_metrics(args.output_dir)
        output_dir = Path(metrics["output_dir"])
        json_path = output_dir / "metrics.json"
        csv_path = output_dir / "metrics_per_task.csv"
        json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics["tasks"][0]))
            writer.writeheader()
            writer.writerows(metrics["tasks"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(f"Split: {metrics['split']} | Complete: yes")
    print(f"{'Group':<22} {'Tasks':>6} {'Successes/Episodes':>20} {'Success rate':>14}")
    for name, stats in [*metrics["groups"].items(), ("overall", metrics["overall"])]:
        counts = f"{stats['num_successes']}/{stats['num_episodes']}"
        print(f"{name:<22} {stats['num_tasks']:>6} {counts:>20} {stats['success_rate']:>13.2%}")
    print(f"Mean task success rate: {metrics['overall']['mean_task_success_rate']:.2%}")
    print("Overall is computed across all tasks/episodes, not by averaging the three group scores.")
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")


if __name__ == "__main__":
    main()
