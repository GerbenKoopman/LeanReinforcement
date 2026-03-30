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

    eval_config = build_eval_config(
        algorithm="alpha_zero",
        seed=42,
        size="light",
        run_dir=run_dir,
        num_theorems=args.num_theorems,
    )
    if args.curvature is not None:
        eval_config.curvature = args.curvature

    evaluator = TestEvaluator(
        config=eval_config,
        run_dir=run_dir,
        dataset_split=args.split,
        use_hyperbolic=args.use_hyperbolic,
    )
    evaluator.train()  # This runs evaluation


if __name__ == "__main__":
    main()
