#!/usr/bin/env python3
"""
Benchmark runner for AlphaZero and Guided Rollout MCTS algorithms.

Runs 18 configurations total:
  - 2 algorithms (alpha_zero, guided_rollout)
  - 3 seeds (42, 43, 44)
  - 3 sizes (light=1x, medium=1.5x, heavy=2x)

Each run uses 20 epochs. Checkpoints are stored as:
  <cache_dir>/benchmark/<algorithm>_<seed>_<size>/

The benchmark is resumable: if a run's checkpoint directory already contains
a completed marker or enough epoch checkpoints, it is skipped.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List

from loguru import logger
from dotenv import load_dotenv

from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.training.trainer import Trainer

load_dotenv()


# ── Base hyperparameters (from training job files) ──────────────────────────

BASE_PARAMS = {
    "data_type": "novel_premises",
    "indexed_corpus_path": (
        os.environ.get("CORPUS_DIR", "") + "/indexed_corpus.pkl"
        if os.environ.get("CORPUS_DIR")
        else None
    ),
    "value_head_hidden_dims": [1024, 512, 256, 128, 64],
    "num_theorems": 100,
    "batch_size": 16,
    "num_tactics_to_expand": 16,
    "num_workers": 12,
    "train_epochs": 50,
    "train_value_head": True,
    "use_final_reward": True,
    "save_training_data": True,
    "save_checkpoints": True,
    "model_name": "kaiyuy/leandojo-lean4-tacgen-byt5-small",
    "max_steps": 10,
    "max_rollout_depth": 30,
    "value_head_batch_size": 4,
    "use_caching": False,
    "inference_timeout": 600.0,
}

# Size-specific overrides: scaling num_iterations and time parameters
SIZE_CONFIGS = {
    "light": {
        "num_iterations": 200,
        "max_time": 40.0,
        "env_timeout": 72,
        "proof_timeout": 300.0,
    },
    "medium": {
        "num_iterations": 300,  # 200 * 1.5
        "max_time": 60.0,  # 40 * 1.5
        "env_timeout": 108,  # 72 * 1.5
        "proof_timeout": 450.0,  # 300 * 1.5
    },
    "heavy": {
        "num_iterations": 400,  # 200 * 2
        "max_time": 80.0,  # 40 * 2
        "env_timeout": 144,  # 72 * 2
        "proof_timeout": 600.0,  # 300 * 2
    },
}

ALGORITHMS = ["alpha_zero", "guided_rollout"]
SEEDS = [42, 43]
SIZES = ["light", "medium", "heavy"]
NUM_EPOCHS = 15


def get_benchmark_dir() -> Path:
    """Get or create the benchmark cache directory."""
    checkpoint_dir = os.getenv("CHECKPOINT_DIR")
    if checkpoint_dir:
        base = Path(checkpoint_dir) / "benchmark"
    else:
        base = Path("checkpoints") / "benchmark"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_run_dir(benchmark_dir: Path, algorithm: str, seed: int, size: str) -> Path:
    """Get the checkpoint directory for a specific run."""
    return benchmark_dir / f"{algorithm}_{seed}_{size}"


def is_run_complete(run_dir: Path, num_epochs: int) -> bool:
    """Check if a run is already complete by looking for a completion marker or
    the final epoch checkpoint."""
    marker = run_dir / "benchmark_complete.json"
    if marker.exists():
        return True
    # Also check if the final epoch checkpoint exists
    for prefix in ["value_head_alpha_zero", "value_head_guided_rollout"]:
        final_ckpt = run_dir / f"{prefix}_epoch_{num_epochs}.pth"
        if final_ckpt.exists():
            return True
    return False


def get_completed_epochs(run_dir: Path) -> int:
    """Determine how many epochs have been completed for a run."""
    if not run_dir.exists():
        return 0
    # Look for theorem_results_epoch_N.json files (written every epoch)
    max_epoch = 0
    for f in run_dir.glob("theorem_results_epoch_*.json"):
        try:
            epoch_num = int(f.stem.split("_")[-1])
            max_epoch = max(max_epoch, epoch_num)
        except ValueError:
            continue
    # Also check value head checkpoints
    for f in run_dir.glob("value_head_*_epoch_*.pth"):
        try:
            epoch_num = int(f.stem.split("_")[-1])
            max_epoch = max(max_epoch, epoch_num)
        except ValueError:
            continue
    return max_epoch


def build_config(
    algorithm: str, seed: int, size: str, run_dir: Path, resume: bool = False
) -> TrainingConfig:
    """Build a TrainingConfig for a specific benchmark run."""
    size_params = SIZE_CONFIGS[size]

    # Determine start epoch if resuming
    start_from = 0
    if resume and run_dir.exists():
        start_from = get_completed_epochs(run_dir)

    remaining_epochs = NUM_EPOCHS - start_from
    if remaining_epochs <= 0:
        remaining_epochs = 0

    # Extract values with type assertions
    data_type_val: str = BASE_PARAMS["data_type"]  # type: ignore[assignment]
    num_theorems_val: int = BASE_PARAMS["num_theorems"]  # type: ignore[assignment]
    max_steps_val: int = BASE_PARAMS["max_steps"]  # type: ignore[assignment]
    batch_size_val: int = BASE_PARAMS["batch_size"]  # type: ignore[assignment]
    num_workers_val: int = BASE_PARAMS["num_workers"]  # type: ignore[assignment]
    indexed_corpus_val = BASE_PARAMS["indexed_corpus_path"]
    train_epochs_val: int = BASE_PARAMS["train_epochs"]  # type: ignore[assignment]
    value_head_batch_size_val: int = BASE_PARAMS["value_head_batch_size"]  # type: ignore[assignment]
    value_head_hidden_dims_val: list = BASE_PARAMS["value_head_hidden_dims"]  # type: ignore[assignment]
    train_value_head_val: bool = BASE_PARAMS["train_value_head"]  # type: ignore[assignment]
    use_final_reward_val: bool = BASE_PARAMS["use_final_reward"]  # type: ignore[assignment]
    save_training_data_val: bool = BASE_PARAMS["save_training_data"]  # type: ignore[assignment]
    use_caching_val: bool = BASE_PARAMS["use_caching"]  # type: ignore[assignment]
    inference_timeout_val: float = BASE_PARAMS["inference_timeout"]  # type: ignore[assignment]
    model_name_val: str = BASE_PARAMS["model_name"]  # type: ignore[assignment]
    num_tactics_val: int = BASE_PARAMS["num_tactics_to_expand"]  # type: ignore[assignment]
    max_rollout_val: int = BASE_PARAMS["max_rollout_depth"]  # type: ignore[assignment]
    num_iterations_val: int = size_params["num_iterations"]  # type: ignore[assignment]
    max_time_val: float = size_params["max_time"]  # type: ignore[assignment]
    env_timeout_val: int = size_params["env_timeout"]  # type: ignore[assignment]
    proof_timeout_val: float = size_params["proof_timeout"]  # type: ignore[assignment]

    config = TrainingConfig(
        # Data and MCTS
        data_type=data_type_val,
        num_epochs=remaining_epochs,
        num_theorems=num_theorems_val,
        num_iterations=num_iterations_val,
        max_steps=max_steps_val,
        batch_size=batch_size_val,
        num_workers=num_workers_val,
        mcts_type=algorithm,
        indexed_corpus_path=indexed_corpus_val,  # type: ignore[arg-type]
        # Training
        train_epochs=train_epochs_val,
        value_head_batch_size=value_head_batch_size_val,
        value_head_hidden_dims=value_head_hidden_dims_val,  # type: ignore[arg-type]
        train_value_head=train_value_head_val,
        use_final_reward=use_final_reward_val,
        save_training_data=save_training_data_val,
        use_caching=use_caching_val,
        # Reproducibility
        seed=seed,
        # Checkpoints - use the run directory directly
        save_checkpoints=True,
        resume=resume and start_from > 0,
        use_test_value_head=False,
        checkpoint_dir=str(run_dir),
        use_wandb=True,
        # Inference
        inference_timeout=inference_timeout_val,
        # Model
        model_name=model_name_val,
        num_tactics_to_expand=num_tactics_val,
        max_rollout_depth=max_rollout_val,
        # Timeouts
        max_time=max_time_val,
        env_timeout=env_timeout_val,
        proof_timeout=proof_timeout_val,
    )
    return config


def mark_run_complete(
    run_dir: Path, algorithm: str, seed: int, size: str, elapsed_time: float
) -> None:
    """Write a completion marker to the run directory."""
    marker = run_dir / "benchmark_complete.json"
    with open(marker, "w") as f:
        json.dump(
            {
                "algorithm": algorithm,
                "seed": seed,
                "size": size,
                "num_epochs": NUM_EPOCHS,
                "elapsed_time_seconds": elapsed_time,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )


def run_single_benchmark(
    algorithm: str, seed: int, size: str, benchmark_dir: Path
) -> Dict[str, Any]:
    """Run a single benchmark configuration. Returns summary dict."""
    run_dir = get_run_dir(benchmark_dir, algorithm, seed, size)
    run_name = f"{algorithm}_{seed}_{size}"

    # Check if already complete
    if is_run_complete(run_dir, NUM_EPOCHS):
        logger.info(f"[SKIP] {run_name} — already complete")
        return {"status": "skipped", "run_name": run_name}

    # Check for partial progress (resume)
    completed = get_completed_epochs(run_dir)
    resume = completed > 0

    if resume:
        logger.info(
            f"[RESUME] {run_name} — resuming from epoch {completed}/{NUM_EPOCHS}"
        )
    else:
        logger.info(f"[START] {run_name} — starting fresh")

    run_dir.mkdir(parents=True, exist_ok=True)

    # Build config
    config = build_config(algorithm, seed, size, run_dir, resume=resume)

    if config.num_epochs <= 0:
        logger.info(f"[SKIP] {run_name} — all epochs already completed")
        mark_run_complete(run_dir, algorithm, seed, size, 0.0)
        return {"status": "skipped", "run_name": run_name}

    # Save benchmark config for reference
    config_file = run_dir / "benchmark_config.json"
    with open(config_file, "w") as f:
        json.dump(
            {
                "algorithm": algorithm,
                "seed": seed,
                "size": size,
                "size_params": SIZE_CONFIGS[size],
                "num_epochs": NUM_EPOCHS,
                "resumed_from_epoch": completed,
                "config": asdict(config),
            },
            f,
            indent=2,
        )

    # Run training
    start_time = time.time()
    try:
        trainer = BenchmarkTrainer(config, run_dir, start_epoch_override=completed)
        trainer.train()
        elapsed = time.time() - start_time
        mark_run_complete(run_dir, algorithm, seed, size, elapsed)
        logger.success(f"[DONE] {run_name} completed in {elapsed/3600:.1f}h")
        return {"status": "completed", "run_name": run_name, "time": elapsed}
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        logger.warning(f"[INTERRUPTED] {run_name} after {elapsed/3600:.1f}h")
        return {"status": "interrupted", "run_name": run_name, "time": elapsed}
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[FAILED] {run_name} after {elapsed/3600:.1f}h: {e}")
        return {
            "status": "failed",
            "run_name": run_name,
            "error": str(e),
            "time": elapsed,
        }


class BenchmarkTrainer(Trainer):
    """Trainer subclass that uses a fixed checkpoint directory (no iteration subdirs)."""

    def __init__(
        self, config: TrainingConfig, run_dir: Path, start_epoch_override: int = 0
    ):
        self.config = config
        self._start_epoch_override = start_epoch_override

        # Set global random seeds for reproducibility
        if config.seed is not None:
            self._set_seeds(config.seed)
            logger.info(f"Global random seed set to {config.seed}")

        # Use the run directory directly as checkpoint dir (no iteration subdirs)
        self.checkpoint_dir = run_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Setup wandb with a descriptive run name
        if self.config.use_wandb:
            import wandb
            from dataclasses import asdict

            wandb.init(
                entity="gerbennkoopman-university-of-amsterdam",
                project="lean-reinforcement-benchmark",
                name=run_dir.name,
                config=asdict(self.config),
                resume="allow",
            )

        self._setup_models()
        self._setup_data()
        self._setup_multiprocessing()

    def _setup_models(self) -> None:
        """Override to handle resume with the benchmark's flat directory structure."""
        logger.info(f"Using checkpoint directory: {self.checkpoint_dir}")

        from lean_reinforcement.agent.transformer import Transformer
        from lean_reinforcement.agent.value_head import ValueHead
        from lean_reinforcement.utilities.checkpoint import load_checkpoint

        self.transformer = Transformer(model_name=self.config.model_name)

        self.value_head = None
        self.start_epoch = self._start_epoch_override

        if self.config.mcts_type == "alpha_zero" or self.config.train_value_head:
            self.value_head = ValueHead(
                self.transformer, hidden_dims=self.config.value_head_hidden_dims
            )

            if self.config.resume:
                prefix = f"value_head_{self.config.mcts_type}"
                loaded_epoch = load_checkpoint(
                    self.value_head, self.checkpoint_dir, prefix=prefix
                )
                if loaded_epoch > 0:
                    self.start_epoch = loaded_epoch
                    logger.info(f"Resuming from epoch {self.start_epoch}")
                else:
                    logger.info("No checkpoint found, starting from scratch")
                    self.start_epoch = 0

        self._log_gpu_memory("After model initialization - ")

    def train(self) -> List[Dict[str, Any]]:
        """Override train to use absolute epoch numbers."""
        all_metrics = []
        try:
            from lean_reinforcement.training.inference_server import InferenceServer

            inference_server = InferenceServer(
                self.transformer,
                self.value_head,
                self.request_queue,
                self.response_queues,
                self.config.batch_size,
            )

            for epoch in range(
                self.start_epoch, self.start_epoch + self.config.num_epochs
            ):
                metrics = self._run_epoch(epoch, inference_server)
                all_metrics.extend(metrics)

            return all_metrics

        except KeyboardInterrupt:
            logger.info("Training interrupted by user.")
            return all_metrics
        except Exception as e:
            logger.error(f"Training crashed: {e}")
            raise e
        finally:
            self._cleanup_workers()
            if self.config.use_wandb:
                import wandb

                wandb.finish()


def get_all_runs() -> List[Dict[str, Any]]:
    """Generate all 18 run configurations."""
    runs = []
    for algorithm in ALGORITHMS:
        for seed in SEEDS:
            for size in SIZES:
                runs.append(
                    {
                        "algorithm": algorithm,
                        "seed": seed,
                        "size": size,
                    }
                )
    return runs


def print_benchmark_status(benchmark_dir: Path) -> None:
    """Print the status of all benchmark runs."""
    runs = get_all_runs()
    print(f"\n{'='*80}")
    print(f"BENCHMARK STATUS — {benchmark_dir}")
    print(f"{'='*80}")
    print(f"{'Run Name':<35} {'Status':<12} {'Epochs':<10} {'Complete'}")
    print(f"{'-'*80}")

    completed_count = 0
    for run in runs:
        run_dir = get_run_dir(benchmark_dir, run["algorithm"], run["seed"], run["size"])
        run_name = f"{run['algorithm']}_{run['seed']}_{run['size']}"

        if is_run_complete(run_dir, NUM_EPOCHS):
            status = "✓ DONE"
            completed_count += 1
            epochs = f"{NUM_EPOCHS}/{NUM_EPOCHS}"
            complete = "Yes"
        elif run_dir.exists():
            completed_epochs = get_completed_epochs(run_dir)
            if completed_epochs > 0:
                status = "⟳ PARTIAL"
                epochs = f"{completed_epochs}/{NUM_EPOCHS}"
                complete = "No"
            else:
                status = "○ STARTED"
                epochs = f"0/{NUM_EPOCHS}"
                complete = "No"
        else:
            status = "— PENDING"
            epochs = f"0/{NUM_EPOCHS}"
            complete = "No"

        print(f"{run_name:<35} {status:<12} {epochs:<10} {complete}")

    print(f"{'-'*80}")
    print(f"Total: {completed_count}/{len(runs)} complete")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark suite for AlphaZero and Guided Rollout"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the status of all benchmark runs and exit.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
        help="Which algorithms to run (default: both).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Which seeds to run (default: 42 43 44).",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=SIZES,
        default=SIZES,
        help="Which sizes to run (default: all).",
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default=None,
        help="Override benchmark cache directory.",
    )

    args = parser.parse_args()

    benchmark_dir = (
        Path(args.benchmark_dir) if args.benchmark_dir else get_benchmark_dir()
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_benchmark_status(benchmark_dir)
        return

    # Filter runs based on CLI args
    runs = [
        r
        for r in get_all_runs()
        if r["algorithm"] in args.algorithms
        and r["seed"] in args.seeds
        and r["size"] in args.sizes
    ]

    logger.info(f"Benchmark directory: {benchmark_dir}")
    logger.info(f"Total runs to process: {len(runs)}")

    print_benchmark_status(benchmark_dir)

    results = []
    for i, run in enumerate(runs, 1):
        logger.info(
            f"\n{'#'*60}\n"
            f"# RUN {i}/{len(runs)}: {run['algorithm']}_{run['seed']}_{run['size']}\n"
            f"{'#'*60}"
        )
        result = run_single_benchmark(
            run["algorithm"], run["seed"], run["size"], benchmark_dir
        )
        results.append(result)

        # Save incremental progress
        progress_file = benchmark_dir / "benchmark_progress.json"
        with open(progress_file, "w") as f:
            json.dump(
                {
                    "total_runs": len(runs),
                    "completed_runs": sum(
                        1 for r in results if r["status"] in ("completed", "skipped")
                    ),
                    "results": results,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                f,
                indent=2,
            )

        # If interrupted, stop gracefully
        if result["status"] == "interrupted":
            logger.warning("Benchmark interrupted. Progress saved. Re-run to resume.")
            break

    print_benchmark_status(benchmark_dir)

    # Final summary
    completed = sum(1 for r in results if r["status"] == "completed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    logger.info(
        f"Benchmark summary: {completed} completed, {skipped} skipped, {failed} failed"
    )


if __name__ == "__main__":
    main()
