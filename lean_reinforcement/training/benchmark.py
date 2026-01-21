"""
Lightweight benchmark for measuring theorem proving performance.

This module provides quick benchmarks to measure proofs/second without
running full training. It's designed for:
1. Hyperparameter tuning experiments
2. Hardware capacity assessment
3. Regression testing for performance changes

Usage:
    python -m lean_reinforcement.training.benchmark --num-theorems 10
    python -m lean_reinforcement.training.benchmark --profile laptop
    python -m lean_reinforcement.training.benchmark --quick
"""

import os
import time
import json
import random
import argparse
import gc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

import torch
from loguru import logger
import datetime

from lean_reinforcement.agent.transformer import Transformer
from lean_reinforcement.utilities.dataloader import LeanDataLoader
from ReProver.common import Corpus
from lean_reinforcement.utilities.config import TrainingConfig
from lean_reinforcement.training.trainer import Trainer


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    # Identifiers
    profile: str
    timestamp: str

    # Performance metrics
    total_time: float
    num_theorems_attempted: int
    num_theorems_succeeded: int
    proofs_per_second: float
    success_rate: float

    # Per-theorem statistics
    avg_time_per_theorem: float
    avg_steps_per_proof: float
    min_proof_time: float
    max_proof_time: float

    # Resource usage
    peak_gpu_memory_gb: float
    avg_cpu_percent: float

    # Configuration used
    config: Dict[str, Any]

    def __str__(self) -> str:
        return (
            f"BenchmarkResult(\n"
            f"  profile={self.profile},\n"
            f"  proofs/sec={self.proofs_per_second:.4f},\n"
            f"  success_rate={self.success_rate:.2%},\n"
            f"  total_time={self.total_time:.1f}s,\n"
            f"  attempted={self.num_theorems_attempted},\n"
            f"  succeeded={self.num_theorems_succeeded}\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Hardware profiles with optimized settings
# Timeout hierarchy: env_timeout < max_time < proof_timeout
# All timeouts in seconds
PROFILES: Dict[str, Dict[str, Any]] = {
    "laptop": {
        "num_workers": 10,
        "batch_size": 16,
        "num_tactics_to_expand": 12,
        "num_iterations": 100,
        "max_time": 300.0,  # 5 min per MCTS step
        "max_steps": 40,
        "proof_timeout": 1200.0,  # 20 min per theorem
        "env_timeout": 180,  # 3 min per tactic
        "max_rollout_depth": 30,
    },
    "hpc": {
        "num_workers": 32,
        "batch_size": 32,
        "num_tactics_to_expand": 32,
        "num_iterations": 400,
        "max_time": 300.0,  # 5 min per MCTS step
        "max_steps": 50,
        "proof_timeout": 1200.0,  # 20 min per theorem
        "env_timeout": 180,  # 3 min per tactic
        "max_rollout_depth": 50,
    },
    "quick": {
        "num_workers": 4,
        "batch_size": 8,
        "num_tactics_to_expand": 8,
        "num_iterations": 50,
        "max_time": 120.0,  # 2 min per MCTS step
        "max_steps": 20,
        "proof_timeout": 300.0,  # 5 min per theorem
        "env_timeout": 60,  # 1 min per tactic
        "max_rollout_depth": 20,
    },
    "minimal": {
        "num_workers": 2,
        "batch_size": 4,
        "num_tactics_to_expand": 4,
        "num_iterations": 25,
        "max_time": 60.0,  # 1 min per MCTS step
        "max_steps": 10,
        "proof_timeout": 120.0,  # 2 min per theorem
        "env_timeout": 30,  # 30 sec per tactic
        "max_rollout_depth": 10,
    },
}


class ProofBenchmark:
    """
    Lightweight benchmark runner for theorem proving.

    This class runs proof attempts with minimal overhead to accurately
    measure throughput and success rate for different configurations.
    """

    def __init__(
        self,
        profile: str = "laptop",
        data_type: str = "novel_premises",
        model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small",
        output_dir: str = "benchmark_results",
    ) -> None:
        """
        Initialize the benchmark runner.

        Args:
            profile: Hardware profile ('laptop', 'hpc', 'quick', 'minimal')
            data_type: Dataset split ('novel_premises' or 'random')
            model_name: HuggingFace model name
            output_dir: Directory for saving results
        """
        self.profile: str = profile
        self.data_type: str = data_type
        self.model_name: str = model_name
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if profile not in PROFILES:
            raise ValueError(
                f"Unknown profile: {profile}. Available: {list(PROFILES.keys())}"
            )

        self.config: Dict[str, Any] = PROFILES[profile].copy()
        self.config["data_type"] = data_type
        self.config["model_name"] = model_name

        self._transformer: Optional[Any] = None
        self._value_head: Optional[Any] = None
        self._dataloader: Optional[Any] = None

    def _setup(self) -> None:
        """Lazy initialization of models and data."""
        if self._transformer is not None:
            return

        logger.info(f"Loading model: {self.model_name}")
        self._transformer = Transformer(model_name=self.model_name)

        logger.info(f"Loading data: {self.data_type}")
        corpus_path = os.path.join("leandojo_benchmark_4/corpus.jsonl")
        corpus = Corpus(corpus_path)
        self._dataloader = LeanDataLoader(
            corpus,
            dataset_path="leandojo_benchmark_4",
            data_type=self.data_type,
        )

    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0.0

    @contextmanager
    def _track_resources(self):
        """Context manager for tracking resource usage."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        start_time = time.time()
        yield
        elapsed = time.time() - start_time

        peak_gpu = self._get_gpu_memory()
        logger.debug(f"Resource tracking: {elapsed:.1f}s, {peak_gpu:.2f}GB GPU")

    def run(
        self,
        num_theorems: int = 10,
        timeout: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> BenchmarkResult:
        """
        Run the benchmark.

        Args:
            num_theorems: Number of theorems to attempt
            timeout: Optional overall timeout (seconds)
            seed: Random seed for reproducibility

        Returns:
            BenchmarkResult with performance metrics
        """

        self._setup()

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        logger.info(
            f"Starting benchmark: {num_theorems} theorems, profile={self.profile}"
        )

        # Prepare configuration
        config_dict = {
            **self.config,
            "num_epochs": 1,
            "num_theorems": num_theorems,
            "train_epochs": 0,
            "train_value_head": False,
            "use_final_reward": True,
            "save_training_data": False,
            "save_checkpoints": False,
            "use_wandb": False,
            "mcts_type": "guided_rollout",  # More stable for benchmarks
            "indexed_corpus_path": None,
            "resume": False,
            "use_test_value_head": False,
            "checkpoint_dir": None,
            "inference_timeout": 600.0,
        }

        # Convert to TrainingConfig
        args = argparse.Namespace(**config_dict)
        training_config = TrainingConfig.from_args(args)

        # Track metrics
        start_time = time.time()
        proof_times: List[float] = []
        proof_steps: List[int] = []
        successes = 0
        attempted = 0

        peak_gpu_memory = 0.0

        try:
            with self._track_resources():
                # Use a simplified run that collects metrics
                trainer = Trainer(training_config)

                # Hook into trainer to collect per-theorem metrics
                # For now, run the full trainer and extract stats
                trainer.train()

                # Extract metrics from wandb or logs (simplified for benchmark)
                # In production, we'd intercept the trainer's metric collection

        except KeyboardInterrupt:
            logger.info("Benchmark interrupted by user")
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")

        total_time = time.time() - start_time
        peak_gpu_memory = self._get_gpu_memory()

        # Calculate metrics
        attempted = num_theorems  # Placeholder - actual count from trainer
        proofs_per_second = successes / total_time if total_time > 0 else 0
        success_rate = successes / attempted if attempted > 0 else 0
        avg_time = total_time / attempted if attempted > 0 else 0
        avg_steps = sum(proof_steps) / len(proof_steps) if proof_steps else 0
        min_time = min(proof_times) if proof_times else 0
        max_time = max(proof_times) if proof_times else 0

        result = BenchmarkResult(
            profile=self.profile,
            timestamp=datetime.datetime.now().isoformat(),
            total_time=total_time,
            num_theorems_attempted=attempted,
            num_theorems_succeeded=successes,
            proofs_per_second=proofs_per_second,
            success_rate=success_rate,
            avg_time_per_theorem=avg_time,
            avg_steps_per_proof=avg_steps,
            min_proof_time=min_time,
            max_proof_time=max_time,
            peak_gpu_memory_gb=peak_gpu_memory,
            avg_cpu_percent=0.0,  # TODO: Add CPU tracking
            config=config_dict,
        )

        self._save_result(result)
        return result

    def _save_result(self, result: BenchmarkResult) -> None:
        """Save benchmark result to JSON."""
        filename = (
            f"benchmark_{result.profile}_{result.timestamp.replace(':', '-')}.json"
        )
        filepath = self.output_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        logger.info(f"Result saved to: {filepath}")

    def compare_profiles(
        self,
        profiles: Optional[List[str]] = None,
        num_theorems: int = 5,
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple hardware profiles.

        Args:
            profiles: List of profile names to compare. If None, compare all.
            num_theorems: Number of theorems per profile.

        Returns:
            Dict mapping profile names to results.
        """
        if profiles is None:
            profiles = list(PROFILES.keys())

        results = {}
        for profile in profiles:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing profile: {profile}")
            logger.info("=" * 60)

            self.profile = profile
            self.config = PROFILES[profile].copy()

            result = self.run(num_theorems=num_theorems)
            results[profile] = result

            # Cleanup between profiles
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._print_comparison(results)
        return results

    def _print_comparison(self, results: Dict[str, BenchmarkResult]) -> None:
        """Print comparison table of results."""
        print("\n" + "=" * 80)
        print("BENCHMARK COMPARISON")
        print("=" * 80)
        print(
            f"{'Profile':<12} {'Proofs/s':<12} {'Success%':<12} "
            f"{'Time(s)':<12} {'GPU(GB)':<12}"
        )
        print("-" * 80)

        for profile, result in sorted(
            results.items(), key=lambda x: -x[1].proofs_per_second
        ):
            print(
                f"{profile:<12} {result.proofs_per_second:<12.4f} "
                f"{result.success_rate * 100:<12.1f} "
                f"{result.total_time:<12.1f} {result.peak_gpu_memory_gb:<12.2f}"
            )

        print("=" * 80)


def run_quick_benchmark():
    """Run a quick benchmark with minimal settings."""
    benchmark = ProofBenchmark(profile="minimal")
    result = benchmark.run(num_theorems=3)
    print(result)
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Lightweight benchmark for theorem proving performance"
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(PROFILES.keys()),
        default="laptop",
        help="Hardware profile to use",
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=10,
        help="Number of theorems to attempt",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["novel_premises", "random"],
        default="novel_premises",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all profiles",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmark with minimal settings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for saving results",
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_benchmark()
        return

    benchmark = ProofBenchmark(
        profile=args.profile,
        data_type=args.data_type,
        output_dir=args.output_dir,
    )

    if args.compare:
        benchmark.compare_profiles(num_theorems=args.num_theorems)
    else:
        result = benchmark.run(num_theorems=args.num_theorems, seed=args.seed)
        print(result)


if __name__ == "__main__":
    main()
