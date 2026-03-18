import argparse
from pathlib import Path
from benchmark.evaluate_benchmark import TestEvaluator, build_eval_config


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Base directory containing the model checkpoint subdirectories",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--use-hyperbolic",
        action="store_true",
        help="Use hyperbolic value head for evaluation",
    )
    parser.add_argument("--num-theorems", type=int, default=250)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    run_dir = find_latest_run(base_dir)

    if not run_dir or not run_dir.exists():
        print(f"Could not find run directory in {base_dir}")
        return

    print(f"Evaluating run in {run_dir}")

    eval_config = build_eval_config(
        algorithm="alpha_zero",
        seed=42,
        size="light",
        run_dir=run_dir,
        num_theorems=args.num_theorems,
    )

    evaluator = TestEvaluator(
        config=eval_config,
        run_dir=run_dir,
        dataset_split=args.split,
        use_hyperbolic=args.use_hyperbolic,
    )
    evaluator.train()  # This runs evaluation


if __name__ == "__main__":
    main()
