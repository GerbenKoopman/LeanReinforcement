#!/usr/bin/env python3
"""
Generate training plots from benchmark cached data.

Creates separate plots per algorithm with color-coded sizes and shaded regions
between seeds (42 and 43). Extracts success rates and proved counts from
theorem_results_epoch_*.json files in the benchmark cache.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Configuration
BENCHMARK_CACHE_DIR = (
    Path.home() / ".cache" / "lean-reinforcement" / "checkpoints" / "benchmark"
)
PLOT_OUTPUT_DIR = Path(__file__).parent / "plots"
ALGORITHMS = ["alpha_zero", "guided_rollout"]
SEEDS = [42, 43]
SIZES = ["light", "medium", "heavy"]

# Color mapping for sizes
SIZE_COLORS = {
    "light": "#1f77b4",  # blue
    "medium": "#ff7f0e",  # orange
    "heavy": "#2ca02c",  # green
}

SIZE_LABELS = {
    "light": "Light (1x)",
    "medium": "Medium (1.5x)",
    "heavy": "Heavy (2x)",
}


def load_epoch_data(run_dir: Path) -> Dict[int, Tuple[int, int, float]]:
    """
    Load success metrics from theorem_results_epoch_*.json files.

    Returns dict: epoch -> (proved, total, success_rate)
    """
    data: Dict[int, Tuple[int, int, float]] = {}

    if not run_dir.exists():
        return data

    for epoch_file in sorted(run_dir.glob("theorem_results_epoch_*.json")):
        try:
            with open(epoch_file) as f:
                result = json.load(f)

            epoch = result.get("epoch", -1)
            if epoch > 0:
                proved = result.get("proved", 0)
                total = result.get("total", 0)
                success_rate = result.get("success_rate", 0.0)
                data[epoch] = (proved, total, success_rate)
        except (ValueError, json.JSONDecodeError, KeyError):
            continue

    return data


def load_validation_losses(run_dir: Path) -> Dict[int, float]:
    """
    Load validation losses from val_loss_epoch_*.json files.

    Returns dict: epoch -> loss
    """
    data: Dict[int, float] = {}

    if not run_dir.exists():
        return data

    for loss_file in sorted(run_dir.glob("val_loss_epoch_*.json")):
        try:
            with open(loss_file) as f:
                loss_data = json.load(f)

            epoch = loss_file.stem.split("_")[-1]
            epoch_num = int(epoch)
            if epoch_num > 0:
                loss = loss_data.get("best_val_loss", None)
                if loss is not None:
                    data[epoch_num] = loss
        except (ValueError, json.JSONDecodeError, KeyError):
            continue

    return data


def collect_benchmark_data() -> Dict:
    """
    Collect all benchmark data organized as:
    algorithm -> size -> seed -> {
        'success_rates': epoch -> (proved, total, success_rate),
        'losses': epoch -> loss
    }
    """
    all_data: Dict = {}

    for algorithm in ALGORITHMS:
        all_data[algorithm] = {}

        for size in SIZES:
            all_data[algorithm][size] = {}

            for seed in SEEDS:
                run_name = f"{algorithm}_{seed}_{size}"
                run_dir = BENCHMARK_CACHE_DIR / run_name

                epoch_data = load_epoch_data(run_dir)
                loss_data = load_validation_losses(run_dir)

                all_data[algorithm][size][seed] = {
                    "success_rates": epoch_data,
                    "losses": loss_data,
                }

    return all_data


def plot_algorithm_success_rate(algorithm: str, all_data: Dict) -> None:
    """
    Create side-by-side plots for success rate for a specific algorithm.

    Left plot: zoomed in on data (0 to max if max < 0.5, else min to 1)
    Right plot: full y-range [0, 1]
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Collect all data to determine zoom range for left plot
    all_rates = []
    for size in SIZES:
        for seed in SEEDS:
            seed_data = all_data[algorithm][size][seed]["success_rates"]
            if seed_data:
                all_rates.extend([seed_data[e][2] for e in seed_data.keys()])

    if all_rates:
        max_rate = max(all_rates)
        min_rate = min(all_rates)
        if max_rate < 0.5:
            zoom_ymin, zoom_ymax = 0, max_rate + 0.05
        else:
            zoom_ymin, zoom_ymax = max(0, min_rate - 0.05), 1
    else:
        zoom_ymin, zoom_ymax = 0, 1

    for ax_idx, ax in enumerate(axs):
        for size in SIZES:
            color = SIZE_COLORS[size]
            label = SIZE_LABELS[size]

            # Collect data for both seeds
            seed_42_data = all_data[algorithm][size][42]["success_rates"]
            seed_43_data = all_data[algorithm][size][43]["success_rates"]

            if not seed_42_data or not seed_43_data:
                continue

            # Get common epochs
            epochs = sorted(set(seed_42_data.keys()) & set(seed_43_data.keys()))
            if not epochs:
                continue

            # Extract success rates
            success_rates_42 = np.array([seed_42_data[e][2] for e in epochs])
            success_rates_43 = np.array([seed_43_data[e][2] for e in epochs])

            # Plot both seeds
            ax.plot(
                epochs,
                success_rates_42,
                color=color,
                linestyle="-",
                linewidth=2.5,
                label=f"{label} (Seed 42)" if ax_idx == 0 else "",
            )
            ax.plot(
                epochs,
                success_rates_43,
                color=color,
                linestyle="--",
                linewidth=2.5,
                label=f"{label} (Seed 43)" if ax_idx == 0 else "",
            )

            # Shade area between seeds
            # ax.fill_between(
            #     epochs, success_rates_42, success_rates_43, color=color, alpha=0.15
            # )

        ax.set_xlabel("Epoch", fontsize=18, fontweight="bold")
        ax.set_ylabel("Success Rate", fontsize=18, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.grid(True, alpha=0.3)

        # Set x-limits
        max_epoch = (
            max(
                [
                    max(all_data[algorithm][size][seed]["success_rates"].keys())
                    for size in SIZES
                    for seed in SEEDS
                    if all_data[algorithm][size][seed]["success_rates"]
                ]
            )
            if any(
                all_data[algorithm][size][seed]["success_rates"]
                for size in SIZES
                for seed in SEEDS
            )
            else 1
        )
        ax.set_xlim(0.5, max_epoch + 0.5)

        if ax_idx == 0:
            ax.set_title(
                "Success Rate vs Epochs (Zoomed)", fontsize=18, fontweight="bold"
            )
            ax.set_ylim(zoom_ymin, zoom_ymax)
        else:
            ax.set_title(
                "Success Rate vs Epochs (Full [0, 1])",
                fontsize=18,
                fontweight="bold",
            )
            ax.set_ylim(0, 1)

    # Only show legend on left plot to avoid duplication
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=16,
    )

    fig.suptitle(
        f"Success Rate: {algorithm.replace('_', ' ').title()}",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    output_path = PLOT_OUTPUT_DIR / f"{algorithm}_success_rate_sidebyside.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def plot_combined_loss(all_data: Dict) -> None:
    """
    Create side-by-side loss plots for both algorithms.
    Guided rollout on the left, alpha zero on the right.
    Plots both seeds with shaded region between them.

    Y-axis: 0 to max_loss * 1.1
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Order: guided_rollout on left, alpha_zero on right
    algorithms_ordered = ["guided_rollout", "alpha_zero"]

    # Collect all data to determine y-range across both algorithms
    all_losses = []
    for algorithm in algorithms_ordered:
        for size in SIZES:
            for seed in SEEDS:
                seed_data = all_data[algorithm][size][seed]["losses"]
                if seed_data:
                    all_losses.extend([seed_data[e] for e in seed_data.keys()])

    if all_losses:
        max_loss = max(all_losses)
        ymax = max_loss * 1.1
    else:
        ymax = 1.0

    for ax_idx, algorithm in enumerate(algorithms_ordered):
        ax = axs[ax_idx]

        for size in SIZES:
            color = SIZE_COLORS[size]
            label = SIZE_LABELS[size]

            # Collect data for both seeds
            seed_42_losses = all_data[algorithm][size][42]["losses"]
            seed_43_losses = all_data[algorithm][size][43]["losses"]

            if not seed_42_losses or not seed_43_losses:
                continue

            # Get common epochs
            epochs = sorted(set(seed_42_losses.keys()) & set(seed_43_losses.keys()))
            if not epochs:
                continue

            # Extract losses
            losses_42 = np.array([seed_42_losses[e] for e in epochs])
            losses_43 = np.array([seed_43_losses[e] for e in epochs])

            # Plot both seeds
            ax.plot(
                epochs,
                losses_42,
                color=color,
                linestyle="-",
                linewidth=2.5,
                label=f"{label} (Seed 42)" if ax_idx == 0 else "",
            )
            ax.plot(
                epochs,
                losses_43,
                color=color,
                linestyle="--",
                linewidth=2.5,
                label=f"{label} (Seed 43)" if ax_idx == 0 else "",
            )

            # Shade area between seeds
            # ax.fill_between(epochs, losses_42, losses_43, color=color, alpha=0.15)

        ax.set_xlabel("Epoch", fontsize=18, fontweight="bold")
        ax.set_ylabel("Validation Loss", fontsize=18, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_title(
            f"{algorithm.replace('_', ' ').title()}",
            fontsize=18,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)

        # Set x-limits
        max_epoch = (
            max(
                [
                    max(all_data[algorithm][size][seed]["losses"].keys())
                    for size in SIZES
                    for seed in SEEDS
                    if all_data[algorithm][size][seed]["losses"]
                ]
            )
            if any(
                all_data[algorithm][size][seed]["losses"]
                for size in SIZES
                for seed in SEEDS
            )
            else 1
        )
        ax.set_xlim(0.5, max_epoch + 0.5)
        ax.set_ylim(0, ymax)

    # Only show legend below plots
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=16,
    )

    fig.suptitle(
        "Validation Loss",
        fontsize=20,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout()
    output_path = PLOT_OUTPUT_DIR / "combined_loss.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def main():
    """Main entry point."""
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark data from {BENCHMARK_CACHE_DIR}...")
    all_data = collect_benchmark_data()

    # Generate side-by-side plots per algorithm for success rate
    for algorithm in ALGORITHMS:
        print(f"Generating success rate plots for {algorithm}...")
        plot_algorithm_success_rate(algorithm, all_data)

    # Generate combined loss plot with both algorithms side by side
    print("Generating combined loss plot...")
    plot_combined_loss(all_data)

    print("Done!")


if __name__ == "__main__":
    main()
