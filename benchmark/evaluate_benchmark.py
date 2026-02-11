#!/usr/bin/env python3
"""
Evaluate all benchmark checkpoints on the test set.

For each completed benchmark run, loads the final checkpoint and evaluates
on the test split using the same hyperparameters used during training.

Produces a JSON results file and per-run theorem-level results.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from dotenv import load_dotenv
from loguru import logger

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.training.trainer import Trainer
from lean_reinforcement.utilities.checkpoint import load_checkpoint
from lean_reinforcement.utilities.config import TrainingConfig

from benchmark.run_benchmark import (
    BASE_PARAMS,
    SIZE_CONFIGS,
    ALGORITHMS,
    SEEDS,
    SIZES,
    NUM_EPOCHS,
    get_benchmark_dir,
    get_run_dir,
    is_run_complete,
    get_all_runs,
)

load_dotenv()


class TestEvaluator(Trainer):
    """Trainer subclass for test-set evaluation of benchmark checkpoints."""

    def __init__(
        self, config: TrainingConfig, run_dir: Path, dataset_split: str = "test"
    ):
        self.config = config
        self.dataset_split = dataset_split

        # Set seeds for reproducibility during evaluation
        if config.seed is not None:
            self._set_seeds(config.seed)

        # Use the run directory as checkpoint dir
        self.checkpoint_dir = run_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # No wandb during evaluation
        self._setup_models_for_eval(run_dir)
        self._setup_data()
        self._setup_multiprocessing()

    def _setup_models_for_eval(self, run_dir: Path) -> None:
        """Set up models and load the best checkpoint."""

        logger.info(f"Loading models for evaluation from {run_dir}")
        self.transformer = Transformer(model_name=self.config.model_name)
        self.value_head = None
        self.start_epoch = 0

        if self.config.mcts_type == "alpha_zero" or self.config.train_value_head:
            self.value_head = ValueHead(
                self.transformer, hidden_dims=self.config.value_head_hidden_dims
            )
            prefix = f"value_head_{self.config.mcts_type}"
            loaded_epoch = load_checkpoint(self.value_head, run_dir, prefix=prefix)
            logger.info(f"Loaded checkpoint from epoch {loaded_epoch}")

        self._log_gpu_memory("After model initialization - ")

    def _run_epoch(self, epoch, inference_server):
        """Override to use test/val split and not shuffle."""
        self._drain_theorem_queue()
        self._start_workers()

        # Use the specified split
        if self.dataset_split == "val":
            split_data = self.dataloader.val_data
        elif self.dataset_split == "test":
            split_data = self.dataloader.test_data
        else:
            split_data = self.dataloader.train_data

        logger.info(
            f"Evaluating on {self.dataset_split} split "
            f"({len(split_data[:self.config.num_theorems])} theorems)"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Don't shuffle for evaluation
        theorems_to_process = split_data[: self.config.num_theorems]

        for thm in theorems_to_process:
            self.theorem_queue.put(thm)

        training_data_buffer, epoch_metrics = self._collect_data(
            theorems_to_process, inference_server, epoch
        )

        self._stop_workers()
        self._drain_queues()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return epoch_metrics


def build_eval_config(
    algorithm: str, seed: int, size: str, run_dir: Path, num_theorems: int = 500
) -> TrainingConfig:
    """Build evaluation config with same hyperparameters as training."""
    size_params = SIZE_CONFIGS[size]

    # Extract values with type assertions
    data_type_val: str = BASE_PARAMS["data_type"]  # type: ignore[assignment]
    max_steps_val: int = BASE_PARAMS["max_steps"]  # type: ignore[assignment]
    num_workers_val: int = BASE_PARAMS["num_workers"]  # type: ignore[assignment]
    indexed_corpus_val = BASE_PARAMS["indexed_corpus_path"]
    value_head_hidden_dims_val: list = BASE_PARAMS["value_head_hidden_dims"]  # type: ignore[assignment]
    use_final_reward_val: bool = BASE_PARAMS["use_final_reward"]  # type: ignore[assignment]
    inference_timeout_val: float = BASE_PARAMS["inference_timeout"]  # type: ignore[assignment]
    model_name_val: str = BASE_PARAMS["model_name"]  # type: ignore[assignment]
    num_tactics_val: int = BASE_PARAMS["num_tactics_to_expand"]  # type: ignore[assignment]
    max_rollout_val: int = BASE_PARAMS["max_rollout_depth"]  # type: ignore[assignment]
    num_iterations_val: int = size_params["num_iterations"]  # type: ignore[assignment]
    max_time_val: float = size_params["max_time"]  # type: ignore[assignment]
    env_timeout_val: int = size_params["env_timeout"]  # type: ignore[assignment]
    proof_timeout_val: float = size_params["proof_timeout"]  # type: ignore[assignment]

    return TrainingConfig(
        data_type=data_type_val,
        num_epochs=1,  # Single evaluation pass
        num_theorems=num_theorems,
        num_iterations=num_iterations_val,
        max_steps=max_steps_val,
        batch_size=8,  # Reduced for evaluation memory efficiency
        num_workers=num_workers_val,
        mcts_type=algorithm,
        indexed_corpus_path=indexed_corpus_val,  # type: ignore[arg-type]
        train_epochs=0,  # No training during evaluation
        value_head_batch_size=2,
        value_head_hidden_dims=value_head_hidden_dims_val,  # type: ignore[arg-type]
        train_value_head=False,
        use_final_reward=use_final_reward_val,
        save_training_data=False,
        use_caching=True,
        seed=seed,
        save_checkpoints=False,
        resume=False,
        use_test_value_head=False,
        checkpoint_dir=str(run_dir),
        use_wandb=False,
        inference_timeout=inference_timeout_val,
        model_name=model_name_val,
        num_tactics_to_expand=num_tactics_val,
        max_rollout_depth=max_rollout_val,
        max_time=max_time_val,
        env_timeout=env_timeout_val,
        proof_timeout=proof_timeout_val,
    )


def evaluate_single_run(
    algorithm: str,
    seed: int,
    size: str,
    benchmark_dir: Path,
    num_theorems: int = 500,
    split: str = "test",
) -> Dict[str, Any]:
    """Evaluate a single benchmark run on the test set."""
    run_dir = get_run_dir(benchmark_dir, algorithm, seed, size)
    run_name = f"{algorithm}_{seed}_{size}"

    # Check if evaluation was already done
    eval_results_file = run_dir / f"eval_results_{split}.json"
    if eval_results_file.exists():
        logger.info(f"[SKIP] {run_name} — evaluation already exists")
        with open(eval_results_file) as f:
            result: Dict[str, Any] = json.load(f)
            return result

    # Check if training run is complete
    if not is_run_complete(run_dir, NUM_EPOCHS):
        logger.warning(f"[SKIP] {run_name} — training not complete")
        return {"status": "training_incomplete", "run_name": run_name}

    logger.info(f"[EVAL] {run_name} on {split} split ({num_theorems} theorems)")

    config = build_eval_config(algorithm, seed, size, run_dir, num_theorems)

    start_time = time.time()
    try:
        evaluator = TestEvaluator(config, run_dir, dataset_split=split)
        results = evaluator.train()
        elapsed = time.time() - start_time

        # Count successes from theorem results
        theorem_results_file = run_dir / "theorem_results_epoch_1.json"
        successful = 0
        total = 0
        theorem_details = []

        if theorem_results_file.exists():
            with open(theorem_results_file) as f:
                epoch_results = json.load(f)
                successful = epoch_results.get("proved", 0)
                total = epoch_results.get("total", 0)
                theorem_details = epoch_results.get("theorems", [])
        else:
            # Fall back to counting from metrics
            for metric in results:
                if isinstance(metric, dict) and "proof_search/success" in metric:
                    total += 1
                    if metric["proof_search/success"]:
                        successful += 1

        success_rate = successful / total if total > 0 else 0.0

        eval_result = {
            "status": "completed",
            "run_name": run_name,
            "algorithm": algorithm,
            "seed": seed,
            "size": size,
            "split": split,
            "num_theorems": total,
            "successful": successful,
            "success_rate": success_rate,
            "time_seconds": elapsed,
            "size_params": SIZE_CONFIGS[size],
            "theorem_details": theorem_details,
        }

        with open(eval_results_file, "w") as f:
            json.dump(eval_result, f, indent=2)

        logger.success(
            f"[DONE] {run_name}: {successful}/{total} = {success_rate:.1%} "
            f"in {elapsed/60:.1f}min"
        )
        return eval_result

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[FAILED] {run_name}: {e}")
        return {"status": "failed", "run_name": run_name, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark checkpoints on test set"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=SIZES)
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=500,
        help="Number of theorems to evaluate on.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
    )
    parser.add_argument("--benchmark-dir", type=str, default=None)
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Path to save aggregated evaluation results.",
    )

    args = parser.parse_args()
    benchmark_dir = (
        Path(args.benchmark_dir) if args.benchmark_dir else get_benchmark_dir()
    )

    runs = [
        r
        for r in get_all_runs()
        if r["algorithm"] in args.algorithms
        and r["seed"] in args.seeds
        and r["size"] in args.sizes
    ]

    logger.info(f"Evaluating {len(runs)} runs on {args.split} set")

    all_results = []
    for i, run in enumerate(runs, 1):
        logger.info(f"\nEvaluation {i}/{len(runs)}")
        result = evaluate_single_run(
            run["algorithm"],
            run["seed"],
            run["size"],
            benchmark_dir,
            args.num_theorems,
            args.split,
        )
        all_results.append(result)

    # Print summary table
    print(f"\n{'='*90}")
    print(f"EVALUATION SUMMARY ({args.split} set)")
    print(f"{'='*90}")
    print(f"{'Run':<35} {'Status':<12} {'Success':<12} {'Rate':<10} {'Time'}")
    print(f"{'-'*90}")

    for r in all_results:
        name = r.get("run_name", "?")
        status = r.get("status", "?")
        if status == "completed":
            succ = f"{r['successful']}/{r['num_theorems']}"
            rate = f"{r['success_rate']:.1%}"
            t = f"{r['time_seconds']/60:.1f}min"
        else:
            succ = rate = t = "—"
        print(f"{name:<35} {status:<12} {succ:<12} {rate:<10} {t}")
    print(f"{'='*90}\n")

    # Save aggregated results
    output_file = args.output_file or str(
        benchmark_dir / f"eval_summary_{args.split}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "split": args.split,
                "num_theorems": args.num_theorems,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
