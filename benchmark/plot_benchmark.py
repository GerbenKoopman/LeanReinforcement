#!/usr/bin/env python3
"""
Plot benchmark results: proof success rate and validation loss per epoch.

Reads the theorem_results_epoch_*.json and training_data_epoch_*.json files
from each benchmark run directory.

Produces:
  - Proof success % per epoch (per algorithm, averaged over seeds)
  - Best validation loss per epoch (per algorithm, averaged over seeds)
  - Combined comparison plots across sizes
  - Test-set evaluation summary bar charts
"""

import matplotlib

matplotlib.use("Agg")  # Must be before pyplot import  # noqa: E402

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmark.run_benchmark import (  # noqa: E402
    ALGORITHMS,
    SEEDS,
    SIZES,
    NUM_EPOCHS,
    SIZE_CONFIGS,
    get_benchmark_dir,
    get_run_dir,
)


def collect_epoch_data(run_dir: Path) -> Dict[str, Any]:
    """Collect per-epoch metrics from a single run directory.

    Returns dict with:
      - epochs: list of epoch numbers
      - success_rates: list of success rates per epoch
      - proved_counts: list of proved counts per epoch
      - total_counts: list of total counts per epoch
      - best_val_losses: list of best val loss per epoch (from training_data files)
      - theorem_names_proved: dict of epoch -> list of proved theorem names
      - theorem_names_failed: dict of epoch -> list of failed theorem names
    """
    data: Dict[str, Any] = {
        "epochs": [],
        "success_rates": [],
        "proved_counts": [],
        "total_counts": [],
        "best_val_losses": [],
        "theorem_names_proved": {},
        "theorem_names_failed": {},
    }

    if not run_dir.exists():
        return data

    # Collect theorem results per epoch
    for epoch_file in sorted(run_dir.glob("theorem_results_epoch_*.json")):
        try:
            epoch_num = int(epoch_file.stem.split("_")[-1])
            with open(epoch_file) as fp:
                result = json.load(fp)

            data["epochs"].append(epoch_num)
            data["success_rates"].append(result.get("success_rate", 0.0))
            data["proved_counts"].append(result.get("proved", 0))
            data["total_counts"].append(result.get("total", 0))

            # Per-theorem details
            theorems = result.get("theorems", [])
            proved = [t["theorem_name"] for t in theorems if t.get("success")]
            failed = [t["theorem_name"] for t in theorems if not t.get("success")]
            data["theorem_names_proved"][epoch_num] = proved
            data["theorem_names_failed"][epoch_num] = failed

        except (ValueError, json.JSONDecodeError, KeyError):
            continue

    # Collect validation losses from checkpoint metadata or wandb logs
    # Try to extract from training_metadata.json or training_data files
    metadata_file = run_dir / "training_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file) as fp:
                json.load(fp)
        except Exception:
            pass

    # Try to extract best val loss from wandb logs in the run directory
    # or from the benchmark_config
    _collect_val_losses(run_dir, data)

    return data


def _collect_val_losses(run_dir: Path, data: Dict) -> None:
    """Try to extract best validation loss per epoch.

    Strategy:
      1. Look for val_loss_epoch_*.json files (saved by trainer)
      2. Look for wandb summary files
      3. Parse log files for val loss lines
    """
    val_losses = {}

    # Strategy 1: Read val_loss_epoch_*.json files (most reliable)
    for loss_file in sorted(run_dir.glob("val_loss_epoch_*.json")):
        try:
            epoch_num = int(loss_file.stem.split("_")[-1])
            with open(loss_file) as fp:
                loss_data = json.load(fp)
            val_losses[epoch_num] = loss_data.get("best_val_loss")
        except (ValueError, json.JSONDecodeError, KeyError):  # noqa: E722
            continue

    # Strategy 2: scan log files for "Val Loss:" pattern (fallback)
    if not val_losses:
        log_files = list(sorted(run_dir.glob("*.log")))
        if Path("logs").exists():
            log_files.extend(list(Path("logs").glob("*.out")))
        for log_file in log_files:
            try:
                with open(log_file) as f:
                    current_epoch = None
                    best_val_for_epoch = float("inf")
                    for line in f:
                        # Match epoch markers from trainer
                        epoch_match = re.search(r"Starting Epoch (\d+)/", line)
                        if epoch_match:
                            if (
                                current_epoch is not None
                                and best_val_for_epoch < float("inf")
                            ):
                                val_losses[current_epoch] = best_val_for_epoch
                            current_epoch = int(epoch_match.group(1))
                            best_val_for_epoch = float("inf")

                        # Match val loss from value head training
                        val_match = re.search(r"Val Loss: ([\d.]+)", line)
                        if val_match and current_epoch is not None:
                            val_loss = float(val_match.group(1))
                            best_val_for_epoch = min(best_val_for_epoch, val_loss)

                        # Match "New best validation loss"
                        best_match = re.search(
                            r"New best validation loss: ([\d.]+)", line
                        )
                        if best_match and current_epoch is not None:
                            val_loss = float(best_match.group(1))
                            best_val_for_epoch = min(best_val_for_epoch, val_loss)

                    # Don't forget the last epoch
                    if current_epoch is not None and best_val_for_epoch < float("inf"):
                        val_losses[current_epoch] = best_val_for_epoch

            except Exception:
                continue

    # Also try wandb local files
    wandb_dir = run_dir / "wandb"
    if wandb_dir.exists():
        for event_file in wandb_dir.rglob("*.jsonl"):
            try:
                with open(event_file) as fp:
                    for line in fp:
                        entry = json.loads(line)
                        if "value_head/val_loss" in entry:
                            epoch = entry.get("value_head/epoch", 0)
                            val_loss = entry["value_head/val_loss"]
                            if epoch not in val_losses or val_loss < val_losses[epoch]:
                                val_losses[epoch] = val_loss
            except Exception:
                continue

    # Align val losses with epochs
    if val_losses:
        for epoch in data["epochs"]:
            if epoch in val_losses:
                data["best_val_losses"].append(val_losses[epoch])
            else:
                data["best_val_losses"].append(None)
    else:
        data["best_val_losses"] = [None] * len(data["epochs"])


def plot_success_rate(benchmark_dir: Path, output_dir: Path) -> None:
    """Plot proof success rate per epoch, grouped by algorithm and size."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for size_idx, size in enumerate(SIZES):
        ax = axes[size_idx]
        ax.set_title(f"Proof Success Rate ({size.capitalize()})", fontsize=14)
        ax.set_xlabel("Epoch")
        if size_idx == 0:
            ax.set_ylabel("Success Rate (%)")
        ax.grid(True, alpha=0.3)

        for algo in ALGORITHMS:
            all_rates = []
            max_epochs = 0

            for seed in SEEDS:
                run_dir = get_run_dir(benchmark_dir, algo, seed, size)
                epoch_data = collect_epoch_data(run_dir)

                if epoch_data["epochs"]:
                    rates = [r * 100 for r in epoch_data["success_rates"]]
                    all_rates.append((epoch_data["epochs"], rates))
                    max_epochs = max(max_epochs, max(epoch_data["epochs"]))

            if not all_rates:
                continue

            # Interpolate to common epoch grid
            epochs_grid = np.arange(1, max_epochs + 1)
            rate_matrix = np.full((len(all_rates), len(epochs_grid)), np.nan)

            for i, (epochs, rates) in enumerate(all_rates):
                for e, r in zip(epochs, rates):
                    if 1 <= e <= max_epochs:
                        rate_matrix[i, e - 1] = r

            mean_rates = np.nanmean(rate_matrix, axis=0)
            std_rates = np.nanstd(rate_matrix, axis=0)

            # Plot mean ± std
            label = algo.replace("_", " ").title()
            color = "tab:blue" if algo == "alpha_zero" else "tab:orange"

            ax.plot(
                epochs_grid,
                mean_rates,
                "-o",
                label=label,
                color=color,
                markersize=3,
                linewidth=1.5,
            )
            ax.fill_between(
                epochs_grid,
                mean_rates - std_rates,
                mean_rates + std_rates,
                alpha=0.2,
                color=color,
            )

            # Plot individual seeds as thin lines
            for i, (epochs, rates) in enumerate(all_rates):
                ax.plot(epochs, rates, "--", alpha=0.3, color=color, linewidth=0.8)

        ax.legend(loc="lower right")
        ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "success_rate_per_epoch.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "success_rate_per_epoch.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved success rate plot to {output_dir / 'success_rate_per_epoch.png'}")


def plot_validation_loss(benchmark_dir: Path, output_dir: Path) -> None:
    """Plot best validation loss per epoch, grouped by algorithm and size."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for size_idx, size in enumerate(SIZES):
        ax = axes[size_idx]
        ax.set_title(f"Best Validation Loss ({size.capitalize()})", fontsize=14)
        ax.set_xlabel("Epoch")
        if size_idx == 0:
            ax.set_ylabel("Best Val Loss (MSE)")
        ax.grid(True, alpha=0.3)

        for algo in ALGORITHMS:
            all_losses = []
            max_epochs = 0

            for seed in SEEDS:
                run_dir = get_run_dir(benchmark_dir, algo, seed, size)
                epoch_data = collect_epoch_data(run_dir)

                if epoch_data["epochs"] and any(
                    v is not None for v in epoch_data["best_val_losses"]
                ):
                    valid_epochs = []
                    valid_losses = []
                    for e, v in zip(
                        epoch_data["epochs"], epoch_data["best_val_losses"]
                    ):
                        if v is not None:
                            valid_epochs.append(e)
                            valid_losses.append(v)
                    if valid_epochs:
                        all_losses.append((valid_epochs, valid_losses))
                        max_epochs = max(max_epochs, max(valid_epochs))

            if not all_losses:
                continue

            epochs_grid = np.arange(1, max_epochs + 1)
            loss_matrix = np.full((len(all_losses), len(epochs_grid)), np.nan)

            for i, (epochs, losses) in enumerate(all_losses):
                for e, l in zip(epochs, losses):
                    if 1 <= e <= max_epochs:
                        loss_matrix[i, e - 1] = l

            mean_losses = np.nanmean(loss_matrix, axis=0)
            std_losses = np.nanstd(loss_matrix, axis=0)

            label = algo.replace("_", " ").title()
            color = "tab:blue" if algo == "alpha_zero" else "tab:orange"

            valid_mask = ~np.isnan(mean_losses)
            ax.plot(
                epochs_grid[valid_mask],
                mean_losses[valid_mask],
                "-o",
                label=label,
                color=color,
                markersize=3,
                linewidth=1.5,
            )
            ax.fill_between(
                epochs_grid[valid_mask],
                (mean_losses - std_losses)[valid_mask],
                (mean_losses + std_losses)[valid_mask],
                alpha=0.2,
                color=color,
            )

        ax.legend(loc="upper right")
        ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    plt.tight_layout()
    plt.savefig(
        output_dir / "validation_loss_per_epoch.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(output_dir / "validation_loss_per_epoch.pdf", bbox_inches="tight")
    plt.close()
    print(
        f"Saved validation loss plot to {output_dir / 'validation_loss_per_epoch.png'}"
    )


def plot_test_evaluation(
    benchmark_dir: Path, output_dir: Path, split: str = "test"
) -> None:
    """Plot test-set evaluation results as a grouped bar chart."""
    eval_file = benchmark_dir / f"eval_summary_{split}.json"
    if not eval_file.exists():
        print(
            f"No evaluation summary found at {eval_file}. Run evaluate_benchmark.py first."
        )
        return

    with open(eval_file) as f:
        eval_data = json.load(f)

    results = eval_data.get("results", [])
    completed = [r for r in results if r.get("status") == "completed"]

    if not completed:
        print("No completed evaluation results to plot.")
        return

    # Group by size
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for size_idx, size in enumerate(SIZES):
        ax = axes[size_idx]
        ax.set_title(f"Test Success Rate ({size.capitalize()})", fontsize=14)
        if size_idx == 0:
            ax.set_ylabel("Success Rate (%)")

        size_results = [r for r in completed if r.get("size") == size]

        algo_data = {}
        for algo in ALGORITHMS:
            algo_results = [r for r in size_results if r.get("algorithm") == algo]
            rates = [r["success_rate"] * 100 for r in algo_results]
            algo_data[algo] = rates

        x = np.arange(len(SEEDS))
        width = 0.35

        for i, algo in enumerate(ALGORITHMS):
            rates = algo_data.get(algo, [])
            label = algo.replace("_", " ").title()
            color = "tab:blue" if algo == "alpha_zero" else "tab:orange"

            if rates:
                bars = ax.bar(
                    x[: len(rates)] + i * width - width / 2,
                    rates,
                    width,
                    label=label,
                    color=color,
                    alpha=0.8,
                )
                # Add value labels on bars
                for bar, rate in zip(bars, rates):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{rate:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in SEEDS])
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"{split}_evaluation_results.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(output_dir / f"{split}_evaluation_results.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved evaluation plot to {output_dir / f'{split}_evaluation_results.png'}")


def plot_combined_comparison(benchmark_dir: Path, output_dir: Path) -> None:
    """Plot a combined comparison: success rate across sizes for each algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for algo_idx, algo in enumerate(ALGORITHMS):
        ax = axes[algo_idx]
        label = algo.replace("_", " ").title()
        ax.set_title(f"{label} — Success Rate by Size", fontsize=14)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Success Rate (%)")
        ax.grid(True, alpha=0.3)

        colors = {"light": "tab:green", "medium": "tab:blue", "heavy": "tab:red"}

        for size in SIZES:
            all_rates = []
            max_epochs = 0

            for seed in SEEDS:
                run_dir = get_run_dir(benchmark_dir, algo, seed, size)
                epoch_data = collect_epoch_data(run_dir)

                if epoch_data["epochs"]:
                    rates = [r * 100 for r in epoch_data["success_rates"]]
                    all_rates.append((epoch_data["epochs"], rates))
                    max_epochs = max(max_epochs, max(epoch_data["epochs"]))

            if not all_rates:
                continue

            epochs_grid = np.arange(1, max_epochs + 1)
            rate_matrix = np.full((len(all_rates), len(epochs_grid)), np.nan)

            for i, (epochs, rates) in enumerate(all_rates):
                for e, r in zip(epochs, rates):
                    if 1 <= e <= max_epochs:
                        rate_matrix[i, e - 1] = r

            mean_rates = np.nanmean(rate_matrix, axis=0)
            std_rates = np.nanstd(rate_matrix, axis=0)

            ax.plot(
                epochs_grid,
                mean_rates,
                "-o",
                label=f"{size.capitalize()} ({SIZE_CONFIGS[size]['num_iterations']} iter)",
                color=colors[size],
                markersize=3,
                linewidth=1.5,
            )
            ax.fill_between(
                epochs_grid,
                mean_rates - std_rates,
                mean_rates + std_rates,
                alpha=0.15,
                color=colors[size],
            )

        ax.legend(loc="lower right")
        ax.set_xlim(0.5, NUM_EPOCHS + 0.5)

    plt.tight_layout()
    plt.savefig(
        output_dir / "combined_size_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.savefig(output_dir / "combined_size_comparison.pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved combined comparison to {output_dir / 'combined_size_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/benchmark",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split for evaluation plots.",
    )
    parser.add_argument(
        "--skip-eval-plots",
        action="store_true",
        help="Skip test evaluation bar charts.",
    )

    args = parser.parse_args()
    benchmark_dir = (
        Path(args.benchmark_dir) if args.benchmark_dir else get_benchmark_dir()
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading benchmark data from: {benchmark_dir}")
    print(f"Saving plots to: {output_dir}")

    # Generate all plots
    plot_success_rate(benchmark_dir, output_dir)
    plot_validation_loss(benchmark_dir, output_dir)
    plot_combined_comparison(benchmark_dir, output_dir)

    if not args.skip_eval_plots:
        plot_test_evaluation(benchmark_dir, output_dir, split=args.split)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
