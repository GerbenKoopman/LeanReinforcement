#!/usr/bin/env python3
"""
Analyze average success rate and average successful proof depth
across alpha_zero training runs, contrasting hyperbolic vs euclidean.

Folder naming: checkpoints/mcts_{geometry}/alpha_zero-{iteration}-{job_id}/
  theorem_results_epoch_{n}.json contains per-theorem success/steps data.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

CHECKPOINTS_DIR = Path(__file__).parent / "checkpoints"
GEOMETRIES = {
    "euclidean": CHECKPOINTS_DIR / "mcts_euclidean",
    "hyperbolic": CHECKPOINTS_DIR / "mcts_hyperbolic",
}


def parse_iteration(folder_name: str) -> Optional[int]:
    """Extract the iteration number from folder names like alpha_zero-44-22710202."""
    m = re.match(r"alpha_zero-(\d+)", folder_name)
    return int(m.group(1)) if m else None


def load_epoch_stats(theorem_results_path: Path) -> Optional[dict]:
    """
    Load a theorem_results_epoch_N.json and return aggregate stats:
      - success_rate
      - avg_successful_depth: mean steps over successful theorems
      - n_proved, n_total
    Returns None if the file is missing or malformed.
    """
    try:
        with theorem_results_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    theorems = data.get("theorems", [])
    n_total = data.get("total", len(theorems))
    n_proved = data.get("proved", sum(1 for t in theorems if t.get("success")))
    success_rate = n_proved / n_total if n_total > 0 else 0.0

    successful_steps = [
        t["steps"] for t in theorems if t.get("success") and t.get("steps", 0) > 0
    ]
    avg_depth = float(np.mean(successful_steps)) if successful_steps else 0.0
    max_depth = float(max(successful_steps)) if successful_steps else 0.0

    return {
        "success_rate": success_rate,
        "avg_successful_depth": avg_depth,
        "max_successful_depth": max_depth,
        "n_proved": n_proved,
        "n_total": n_total,
    }


def collect_geometry_stats(geometry_dir: Path) -> dict[int, dict]:
    """
    For all alpha_zero-* folders in geometry_dir, collect per-iteration stats
    (averaging over all epochs and over multiple runs with the same iteration number).

    Returns: {iteration -> {"success_rate": float, "avg_depth": float, "n_runs": int}}
    """
    # iteration -> list of epoch-level stats
    iter_stats: dict[int, list[dict]] = defaultdict(list)

    for run_dir in sorted(geometry_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        iteration = parse_iteration(run_dir.name)
        if iteration is None:
            continue

        epoch_files = sorted(
            run_dir.glob("theorem_results_epoch_*.json"),
            key=lambda p: int(p.stem.split("epoch_")[1]),
        )
        for ep_file in epoch_files:
            stats = load_epoch_stats(ep_file)
            if stats is not None:
                iter_stats[iteration].append(stats)

    result = {}
    for iteration, stat_list in sorted(iter_stats.items()):
        result[iteration] = {
            "success_rate": float(np.mean([s["success_rate"] for s in stat_list])),
            "avg_depth": float(np.mean([s["avg_successful_depth"] for s in stat_list])),
            "max_depth": float(max(s["max_successful_depth"] for s in stat_list)),
            "n_epochs": len(stat_list),
        }
    return result


def main() -> None:
    all_stats = {}
    for geometry, geo_dir in GEOMETRIES.items():
        if not geo_dir.exists():
            print(f"Warning: {geo_dir} does not exist, skipping.")
            continue
        print(f"Loading {geometry}...")
        all_stats[geometry] = collect_geometry_stats(geo_dir)
        iterations = sorted(all_stats[geometry].keys())
        print(f"  Found iterations: {iterations}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13), sharex=False)

    colors = {"euclidean": "#1f77b4", "hyperbolic": "#d62728"}
    markers = {"euclidean": "o", "hyperbolic": "s"}

    for ax_idx, (metric_key, ylabel, title) in enumerate(
        [
            (
                "success_rate",
                "Average Success Rate",
                "Average Success Rate per Iteration",
            ),
            (
                "avg_depth",
                "Average Depth of Successful Proofs",
                "Average Successful Proof Depth per Iteration",
            ),
            (
                "max_depth",
                "Max Depth of Successful Proofs",
                "Max Successful Proof Depth per Iteration",
            ),
        ]
    ):
        ax = axes[ax_idx]
        for geometry, stats in all_stats.items():
            if not stats:
                continue
            iters = sorted(stats.keys())
            values = [stats[i][metric_key] for i in iters]
            ax.plot(
                iters,
                values,
                marker=markers[geometry],
                color=colors[geometry],
                label=geometry.capitalize(),
                linewidth=2,
                markersize=6,
                alpha=0.85,
            )
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(__file__).parent / "plots" / "training_depth_comparison.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.show()

    # ── Print summary table ───────────────────────────────────────────────────
    print(
        "\n{:<12} {:<10} {:<16} {:<22} {:<22} {:<10}".format(
            "Geometry",
            "Iteration",
            "Success Rate",
            "Avg Successful Depth",
            "Max Successful Depth",
            "N Epochs",
        )
    )
    print("-" * 90)
    for geometry, stats in all_stats.items():
        for iteration in sorted(stats.keys()):
            s = stats[iteration]
            print(
                "{:<12} {:<10} {:<16.4f} {:<22.4f} {:<22.4f} {:<10}".format(
                    geometry,
                    iteration,
                    s["success_rate"],
                    s["avg_depth"],
                    s["max_depth"],
                    s["n_epochs"],
                )
            )


if __name__ == "__main__":
    main()
