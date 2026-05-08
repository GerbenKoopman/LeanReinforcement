import argparse
import json
from pathlib import Path
from typing import Optional

from benchmark.evaluate_benchmark import TestEvaluator, build_eval_config
from lean_reinforcement.utilities.config import TrainingConfig


def find_latest_run(base_dir: Path):
    if not base_dir.exists():
        return None
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and "-" in d.name]
    if not dirs:
        return base_dir  # Assume it IS the run dir
    # Sort by the iteration number after the hyphen
    latest_dir = sorted(
        dirs,
        key=lambda d: (
            int(d.name.split("-")[-1]) if d.name.split("-")[-1].isdigit() else 0
        ),
    )[-1]
    return latest_dir


def load_training_config(run_dir: Path) -> Optional[TrainingConfig]:
    config_path = run_dir / "training_config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            raw = json.load(f)
        config = TrainingConfig(**raw)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(f"Warning: failed to load training config from {config_path}: {exc}")
        return None

    # Ensure evaluation writes under the specific run directory.
    config.checkpoint_dir = str(run_dir)
    return config


def build_eval_from_training(
    config: TrainingConfig, num_theorems: int
) -> TrainingConfig:
    # Keep search hyperparameters identical, but adjust training-only settings.
    config.num_epochs = 1
    config.num_theorems = num_theorems
    config.save_checkpoints = False
    config.save_training_data = False
    config.use_wandb = False
    config.resume = False
    config.train_value_head = False
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory containing checkpoint subdirectories (used when --run-dir is not provided)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Exact run directory to evaluate (preferred for parallel runs)",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--use-hyperbolic",
        action="store_true",
        help="Use hyperbolic value head for evaluation",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=None,
        help="Optional hyperbolic curvature. If omitted, config default is used.",
    )
    parser.add_argument("--num-theorems", type=int, default=250)
    args = parser.parse_args()

    run_dir = None
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
    elif args.base_dir is not None:
        base_dir = Path(args.base_dir)
        run_dir = find_latest_run(base_dir)

    if run_dir is None:
        parser.error("One of --run-dir or --base-dir must be provided")
        return

    if not run_dir.exists():
        print(f"Could not find run directory: {run_dir}")
        return

    print(f"Evaluating run in {run_dir}")

    training_config = load_training_config(run_dir)
    if training_config is not None:
        eval_config = build_eval_from_training(training_config, args.num_theorems)
    else:
        eval_config = build_eval_config(
            algorithm="alpha_zero",
            seed=42,
            size="light",
            run_dir=run_dir,
            num_theorems=args.num_theorems,
        )

    if args.curvature is not None:
        eval_config.curvature = args.curvature

    use_hyperbolic = eval_config.use_hyperbolic
    if training_config is None:
        use_hyperbolic = args.use_hyperbolic
        eval_config.use_hyperbolic = args.use_hyperbolic
    elif args.use_hyperbolic and not eval_config.use_hyperbolic:
        print(
            "Warning: --use-hyperbolic ignored because training_config.json "
            "indicates euclidean evaluation."
        )

    evaluator = TestEvaluator(
        config=eval_config,
        run_dir=run_dir,
        dataset_split=args.split,
        use_hyperbolic=use_hyperbolic,
    )
    evaluator.train()  # This runs evaluation


if __name__ == "__main__":
    main()
