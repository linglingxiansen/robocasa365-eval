"""Recompute the published local reference counts without GPU or simulator dependencies."""
import argparse
import json
import math
from pathlib import Path


def verify(root):
    root = Path(root)
    metrics = json.loads((root / "metrics.json").read_text())
    tasks = metrics["tasks"]
    if len(tasks) != 50 or len({r["task"] for r in tasks}) != 50:
        raise ValueError("Expected exactly 50 unique tasks")
    groups = {}
    for row in tasks:
        stats = json.loads((root / "evals/pretrain" / row["task"] / "stats.json").read_text())
        count, rate = stats["num_episodes"], stats["success_rate"]
        if count != 50 or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("Invalid episode count or success rate")
        successes = rate * count
        if not math.isclose(successes, round(successes), abs_tol=1e-6):
            raise ValueError("Success rate does not encode an integer count")
        if row["num_successes"] != round(successes) or row["num_episodes"] != count:
            raise ValueError("Per-task metrics disagree with raw stats")
        entry = groups.setdefault(row["task_group"], [0, 0, 0])
        entry[0] += 1
        entry[1] += round(successes)
        entry[2] += count
    expected = {"atomic_seen": [18, 711, 900], "composite_seen": [16, 432, 800],
                "composite_unseen": [16, 218, 800]}
    if groups != expected:
        raise ValueError("Reference group counts disagree")
    for group, (task_count, successes, count) in groups.items():
        summary = metrics["groups"][group]
        if (summary["num_tasks"], summary["num_successes"], summary["num_episodes"]) != (task_count, successes, count):
            raise ValueError("Group metrics disagree with task counts")
        if not math.isclose(summary["success_rate"], successes/count):
            raise ValueError("Group rate disagrees with counts")
    if (metrics["overall"]["num_successes"], metrics["overall"]["num_episodes"]) != (1361, 2500):
        raise ValueError("Overall metrics disagree")
    if not metrics["complete"] or not math.isclose(metrics["overall"]["success_rate"], 1361/2500):
        raise ValueError("Reference is incomplete or invalid")
    return {"complete": True, "tasks": 50, "successes": 1361, "episodes": 2500, "success_rate_pct": 54.44}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]/"results/reference_200k"))
    print(json.dumps(verify(parser.parse_args().root), indent=2))
