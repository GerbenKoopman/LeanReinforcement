#!/usr/bin/env python

import subprocess
import matplotlib.pyplot as plt
import numpy as np


def store_data():
    for algorithm in mixed_runs.keys():
        runs = mixed_runs[algorithm]

        for run_id in [
            key for key in runs.keys() if key not in ("avg_loss", "success_rate")
        ]:
            folder_name = subprocess.getoutput(f"ls {wandb_folder} | grep {run_id}")
            runs[run_id] = subprocess.getoutput(
                f"wandb sync --view --verbose {wandb_folder}{folder_name}/run-{run_id}.wandb"
            )

        # Create plot folder if it doesn't exist
        subprocess.run(["mkdir", "-p", plot_folder])

        # Convert all values to strings before joining
        runs_str_values = [str(value) for value in runs.values()]

        # Store concatenated strings to a text file
        with open(f"{plot_folder}{algorithm}.txt", "w") as f:
            f.write("\n".join(runs_str_values))


def plot_data():
    for algorithm in mixed_runs.keys():
        runs = mixed_runs[algorithm]

        with open(f"{plot_folder}{algorithm}.txt", "r") as f:
            found = False
            avg_loss = []

            for line in f:
                if found:
                    avg_loss.append(float(line.split(" ")[-1].strip('"\n')))

                if "value_head/avg_loss" in line:
                    found = True
                else:
                    found = False

        # Filter data
        runs["avg_loss"] = avg_loss[::2]

    min_num_epochs = min(
        len(mixed_runs["guided_rollout"]["avg_loss"]),  # type: ignore[arg-type]
        len(mixed_runs["alpha_zero"]["avg_loss"]),  # type: ignore[arg-type]
    )

    for algorithm in mixed_runs.keys():
        avg_loss = mixed_runs[algorithm]["avg_loss"][:min_num_epochs]  # type: ignore[index]

        # Plot both figures side by side
        x = np.arange(len(avg_loss)) / 3.0
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))

        # Left: full range
        axs[0].plot(x, avg_loss, label="Avg Loss")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("Average Loss")
        axs[0].set_title("Average Loss vs Epochs")
        axs[0].set_xlim(0, x[-1] + 1 if len(x) else 0)
        axs[0].legend(loc="lower left")
        axs[0].grid(True)

        # Right: limited y-range
        axs[1].plot(x, avg_loss, label="Avg Loss")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("Average Loss")
        axs[1].set_title("Average Loss vs Epochs")
        axs[1].set_xlim(0, x[-1] + 1 if len(x) else 0)
        axs[1].set_ylim(0, 1)
        axs[1].legend(loc="lower left")
        axs[1].grid(True)

        fig.tight_layout()
        fig.savefig(f"{plot_folder}{algorithm}_side_by_side.png")


def count_successes():
    for algorithm in mixed_runs.keys():
        runs = mixed_runs[algorithm]

        running_successes = []

        with open(f"{plot_folder}{algorithm}.txt", "r") as f:
            found = False
            proof_total = 0
            successes = 0

            for line in f:
                if found and "1" in line:
                    successes += 1
                    proof_total += 1
                elif found:
                    proof_total += 1

                found = False

                if "training_data/positive_samples" in line:
                    found = True

                running_successes.append(
                    successes / proof_total if proof_total > 0 else 0
                )

        runs["success_rate"] = successes / proof_total if proof_total > 0 else 0
        print(f"Algorithm: {algorithm}, Success Rate: {runs['success_rate']:.2%}")

        # Plot running successes
        x = np.arange(len(running_successes))
        plt.figure(figsize=(10, 6))
        plt.plot(x, running_successes, label="Success Rate")
        plt.xlabel("Number of Proofs Attempted")
        plt.ylabel("Success Rate")
        plt.title(f"Running Success Rate vs Proofs Attempted ({algorithm})")
        plt.xlim(0, x[-1] + 1 if len(x) else 0)
        # plt.ylim(0, 1)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f"{plot_folder}{algorithm}_success_rate.png")


def main():
    store_data()
    plot_data()
    count_successes()


if __name__ == "__main__":
    plot_folder = "plots/"
    wandb_folder = "wandb/"
    mixed_runs = {
        "guided_rollout": {
            "mtixou6j": "",
            "z7biwqn6": "",
            "atsj4y2v": "",
            "avg_loss": [],
            "success_rate": 0.0,
        },
        "alpha_zero": {
            "8ytd9339": "",
            "4txgmct4": "",
            "uqcxidxl": "",
            "avg_loss": [],
            "success_rate": 0.0,
        },
    }

    main()
