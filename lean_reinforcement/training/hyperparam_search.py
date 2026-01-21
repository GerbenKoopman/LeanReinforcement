"""
Hyperparameter search module for optimizing proofs/second on different hardware.

This module provides grid search and binary search capabilities to find optimal
hyperparameters for theorem proving. Results are designed to be transferable
from local testing (laptop) to HPC clusters (like Snellius).

Key metrics optimized:
- Proofs per second (throughput)
- Success rate (accuracy)
- GPU/CPU utilization efficiency

Hardware profiles:
- laptop: Constrained VRAM/RAM (e.g., RTX 4060 with 8GB VRAM)
- hpc: High-end hardware (e.g., A100 with 80GB VRAM)
"""

import time
import json
import random
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

# Try to import wandb but make it optional
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None  # type: ignore[assignment]


@dataclass
class HyperparameterConfig:
    """Configuration for a single hyperparameter search trial."""

    # Core search parameters (most impactful)
    num_workers: int = 10
    batch_size: int = 16
    num_tactics_to_expand: int = 12
    num_iterations: int = 100

    # Timeout parameters (all in seconds)
    # Hierarchy: env_timeout < max_time < proof_timeout
    max_time: float = 300.0  # Max time per MCTS search step
    max_steps: int = 40  # Max proof depth (not a timeout)
    proof_timeout: float = 1200.0  # Max time for entire proof
    env_timeout: int = 180  # Max time per tactic execution

    # Search behavior
    max_rollout_depth: int = 30
    mcts_type: str = "guided_rollout"

    # Fixed parameters (rarely tuned)
    model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
    data_type: str = "novel_premises"

    # Training parameters (for full training runs)
    num_epochs: int = 1
    num_theorems: int = 50
    train_epochs: int = 1
    train_value_head: bool = False
    use_final_reward: bool = True

    # Evaluation mode
    save_training_data: bool = False
    save_checkpoints: bool = False
    use_wandb: bool = False

    def to_args_dict(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for TrainingConfig."""
        return {
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "num_tactics_to_expand": self.num_tactics_to_expand,
            "num_iterations": self.num_iterations,
            "max_time": self.max_time,
            "max_steps": self.max_steps,
            "proof_timeout": self.proof_timeout,
            "env_timeout": self.env_timeout,
            "max_rollout_depth": self.max_rollout_depth,
            "mcts_type": self.mcts_type,
            "model_name": self.model_name,
            "data_type": self.data_type,
            "num_epochs": self.num_epochs,
            "num_theorems": self.num_theorems,
            "train_epochs": self.train_epochs,
            "train_value_head": self.train_value_head,
            "use_final_reward": self.use_final_reward,
            "save_training_data": self.save_training_data,
            "save_checkpoints": self.save_checkpoints,
            "use_wandb": self.use_wandb,
            "indexed_corpus_path": None,
            "resume": False,
            "use_test_value_head": False,
            "checkpoint_dir": None,
            "inference_timeout": 600.0,
        }


@dataclass
class TrialResult:
    """Results from a single hyperparameter trial."""

    config: HyperparameterConfig
    total_time: float
    num_proofs_attempted: int
    num_proofs_succeeded: int
    proofs_per_second: float
    success_rate: float
    avg_proof_time: float
    avg_steps_per_proof: float
    error_message: Optional[str] = None

    @property
    def score(self) -> float:
        """
        Combined score for ranking trials.

        Balances throughput (proofs/second) with success rate.
        Higher is better.
        """
        # Weight success rate heavily but also reward throughput
        return self.proofs_per_second * (self.success_rate**2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "total_time": self.total_time,
            "num_proofs_attempted": self.num_proofs_attempted,
            "num_proofs_succeeded": self.num_proofs_succeeded,
            "proofs_per_second": self.proofs_per_second,
            "success_rate": self.success_rate,
            "avg_proof_time": self.avg_proof_time,
            "avg_steps_per_proof": self.avg_steps_per_proof,
            "score": self.score,
            "error_message": self.error_message,
        }


# Hardware-specific default configurations
LAPTOP_DEFAULTS = HyperparameterConfig(
    num_workers=10,  # Avoid thermal throttling
    batch_size=16,  # Saturate RTX 4060
    num_tactics_to_expand=12,  # Reduce Lean executions
    num_iterations=100,  # Minimum viable for AlphaZero
    max_time=300.0,  # 5 minutes per MCTS step
    max_steps=40,  # Reasonable depth
    proof_timeout=1200.0,  # 20 minutes per theorem
    env_timeout=180,  # 3 minutes per tactic
    max_rollout_depth=30,
)

HPC_DEFAULTS = HyperparameterConfig(
    num_workers=32,  # Utilize full node
    batch_size=32,  # Larger batches for A100
    num_tactics_to_expand=32,  # Full expansion
    num_iterations=400,  # Deeper search
    max_time=300.0,  # 5 minutes per MCTS step
    max_steps=50,  # Allow longer proofs
    proof_timeout=1200.0,  # 20 minutes per theorem
    env_timeout=180,  # 3 minutes per tactic (same as laptop)
    max_rollout_depth=50,
)


# Search spaces for grid search
LAPTOP_SEARCH_SPACE = {
    "num_workers": [6, 8, 10, 12],
    "batch_size": [8, 16, 24],
    "num_tactics_to_expand": [8, 12, 16],
    "num_iterations": [200, 300, 400],
}

HPC_SEARCH_SPACE = {
    "num_workers": [16, 24, 32, 48],
    "batch_size": [16, 32, 48],
    "num_tactics_to_expand": [16, 24, 32],
    "num_iterations": [200, 300, 400],
}


class HyperparameterSearcher:
    """
    Orchestrates hyperparameter search to optimize proofs/second.

    Supports:
    - Grid search: Exhaustive search over parameter combinations
    - Binary search: Efficient search for optimal value of single parameter
    - Adaptive search: Iteratively refines the search space
    """

    def __init__(
        self,
        hardware_profile: str = "laptop",
        output_dir: str = "hyperparam_results",
        wandb_project: str = "lean-hyperparam-search",
        use_wandb: bool = False,
    ):
        """
        Initialize the hyperparameter searcher.

        Args:
            hardware_profile: 'laptop' or 'hpc'
            output_dir: Directory to save results
            wandb_project: WandB project name for logging
            use_wandb: Whether to log to WandB
        """
        self.hardware_profile = hardware_profile
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project

        # Set defaults based on hardware profile
        if hardware_profile == "laptop":
            self.default_config = LAPTOP_DEFAULTS
            self.search_space = LAPTOP_SEARCH_SPACE
        else:
            self.default_config = HPC_DEFAULTS
            self.search_space = HPC_SEARCH_SPACE

        self.results: List[TrialResult] = []

    def _run_single_trial(
        self,
        config: HyperparameterConfig,
        num_theorems: int = 50,
        timeout_per_theorem: float = 600.0,
    ) -> TrialResult:
        """
        Run a single hyperparameter trial.

        This is a lightweight benchmark that:
        1. Loads a subset of theorems
        2. Attempts to prove them with given config
        3. Measures throughput and success rate
        """
        from lean_reinforcement.utilities.config import TrainingConfig
        from lean_reinforcement.training.trainer import Trainer
        import argparse

        logger.info(f"Starting trial with config: {asdict(config)}")

        # Override config for benchmark mode
        config.num_theorems = num_theorems
        config.num_epochs = 1
        config.save_training_data = False
        config.save_checkpoints = False
        config.train_value_head = False

        # Convert to TrainingConfig
        args_dict = config.to_args_dict()
        args = argparse.Namespace(**args_dict)
        training_config = TrainingConfig.from_args(args)

        start_time = time.time()
        total_steps = 0
        num_succeeded = 0
        error_msg = None

        try:
            trainer = Trainer(training_config)

            # Run for 1 epoch with limited theorems
            # We intercept the metrics instead of full training
            training_config.num_epochs = 1

            # We'll collect metrics from the trainer
            trainer.train()

            # Parse results from trainer (we need to add metric collection)
            # For now, estimate from logs or wandb

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Trial failed: {e}")

        total_time = time.time() - start_time
        num_attempted = num_theorems

        # Calculate metrics (placeholder - actual implementation needs trainer integration)
        proofs_per_second = num_succeeded / total_time if total_time > 0 else 0
        success_rate = num_succeeded / num_attempted if num_attempted > 0 else 0
        avg_time = total_time / num_attempted if num_attempted > 0 else 0
        avg_steps = total_steps / num_attempted if num_attempted > 0 else 0

        result = TrialResult(
            config=config,
            total_time=total_time,
            num_proofs_attempted=num_attempted,
            num_proofs_succeeded=num_succeeded,
            proofs_per_second=proofs_per_second,
            success_rate=success_rate,
            avg_proof_time=avg_time,
            avg_steps_per_proof=avg_steps,
            error_message=error_msg,
        )

        self.results.append(result)
        return result

    def grid_search(
        self,
        search_space: Optional[Dict[str, List[Any]]] = None,
        num_theorems: int = 50,
        max_trials: Optional[int] = None,
    ) -> List[TrialResult]:
        """
        Perform grid search over hyperparameter space.

        Args:
            search_space: Dict mapping param names to lists of values to try.
                         If None, uses default for hardware profile.
            num_theorems: Number of theorems per trial.
            max_trials: Maximum number of trials to run (for large search spaces).

        Returns:
            List of TrialResults sorted by score (best first).
        """
        if search_space is None:
            search_space = self.search_space

        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = [search_space[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        if max_trials and len(all_combinations) > max_trials:
            logger.info(
                f"Sampling {max_trials} trials from {len(all_combinations)} combinations"
            )
            random.shuffle(all_combinations)
            all_combinations = all_combinations[:max_trials]

        logger.info(f"Running grid search with {len(all_combinations)} trials")

        if self.use_wandb and WANDB_AVAILABLE and wandb is not None:
            try:
                wandb.init(
                    project=self.wandb_project,
                    config={
                        "search_type": "grid",
                        "hardware_profile": self.hardware_profile,
                        "search_space": search_space,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize WandB: {e}. Continuing without WandB logging."
                )
                self.use_wandb = False

        results = []
        for i, values in enumerate(all_combinations):
            config = HyperparameterConfig(**asdict(self.default_config))

            # Override with trial values
            for name, value in zip(param_names, values):
                setattr(config, name, value)

            logger.info(f"Trial {i + 1}/{len(all_combinations)}")
            result = self._run_single_trial(config, num_theorems)
            results.append(result)

            # Log to wandb
            if self.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "trial": i + 1,
                        **{
                            f"param/{name}": value
                            for name, value in zip(param_names, values)
                        },
                        "proofs_per_second": result.proofs_per_second,
                        "success_rate": result.success_rate,
                        "score": result.score,
                    }
                )

            # Save intermediate results
            self._save_results(results, "grid_search_intermediate.json")

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        self._save_results(results, "grid_search_final.json")

        if self.use_wandb and wandb is not None:
            wandb.finish()

        return results

    def binary_search_parameter(
        self,
        param_name: str,
        min_val: float,
        max_val: float,
        num_theorems: int = 50,
        tolerance: float = 0.1,
        max_iterations: int = 10,
    ) -> Tuple[float, TrialResult]:
        """
        Binary search for optimal value of a single parameter.

        This is efficient for parameters where:
        - Increasing the value improves some metric up to a point
        - Beyond that point, performance degrades (e.g., due to resource limits)

        Good candidates:
        - num_workers: Too many causes thrashing
        - batch_size: Too large causes OOM
        - num_iterations: Diminishing returns

        Args:
            param_name: Name of parameter to search.
            min_val: Minimum value to try.
            max_val: Maximum value to try.
            num_theorems: Theorems per trial.
            tolerance: Stop when range is smaller than this fraction.
            max_iterations: Maximum search iterations.

        Returns:
            Tuple of (optimal_value, best_result).
        """
        logger.info(f"Binary search for {param_name} in range [{min_val}, {max_val}]")

        if self.use_wandb and WANDB_AVAILABLE and wandb is not None:
            try:
                wandb.init(
                    project=self.wandb_project,
                    config={
                        "search_type": "binary",
                        "param_name": param_name,
                        "min_val": min_val,
                        "max_val": max_val,
                    },
                )
            except Exception as e:
                logger.warning(
                    f"Failed to initialize WandB: {e}. Continuing without WandB logging."
                )
                self.use_wandb = False

        low, high = min_val, max_val
        best_val = (low + high) / 2
        best_result = None

        for iteration in range(max_iterations):
            # Test three points: low, mid, high
            mid = (low + high) / 2

            # Create configs for mid point
            config = HyperparameterConfig(**asdict(self.default_config))
            setattr(config, param_name, int(mid) if isinstance(min_val, int) else mid)

            result = self._run_single_trial(config, num_theorems)

            if best_result is None or result.score > best_result.score:
                best_val = mid
                best_result = result

            logger.info(
                f"Iteration {iteration + 1}: {param_name}={mid:.2f}, "
                f"score={result.score:.4f}, best={best_val:.2f}"
            )

            if self.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "iteration": iteration + 1,
                        f"param/{param_name}": mid,
                        "score": result.score,
                        "best_score": best_result.score,
                    }
                )

            # Narrow search range based on score gradient
            # This is a simplified version - could use more sophisticated optimization
            range_size = high - low
            if range_size / max_val < tolerance:
                logger.info(f"Converged at {param_name}={best_val:.2f}")
                break

            # Test slightly above and below mid
            test_low = (low + mid) / 2
            test_high = (mid + high) / 2

            config_low = HyperparameterConfig(**asdict(self.default_config))
            setattr(
                config_low,
                param_name,
                int(test_low) if isinstance(min_val, int) else test_low,
            )
            result_low = self._run_single_trial(config_low, num_theorems)

            config_high = HyperparameterConfig(**asdict(self.default_config))
            setattr(
                config_high,
                param_name,
                int(test_high) if isinstance(min_val, int) else test_high,
            )
            result_high = self._run_single_trial(config_high, num_theorems)

            # Move toward better region
            if result_low.score > result_high.score:
                high = mid
            else:
                low = mid

        if self.use_wandb and wandb is not None:
            wandb.finish()

        if best_result is None:
            raise RuntimeError("Binary search failed to find a valid configuration")

        return best_val, best_result

    def quick_benchmark(
        self,
        config: Optional[HyperparameterConfig] = None,
        num_theorems: int = 5,
    ) -> TrialResult:
        """
        Run a quick benchmark with given or default config.

        Useful for:
        - Quick sanity checks
        - Establishing baseline performance
        - Testing configuration changes
        """
        if config is None:
            config = self.default_config

        logger.info("Running quick benchmark...")
        return self._run_single_trial(config, num_theorems)

    def _save_results(self, results: List[TrialResult], filename: str) -> None:
        """Save results to JSON file."""
        filepath = self.output_dir / filename
        data = [r.to_dict() for r in results]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {filepath}")

    def load_results(self, filename: str) -> List[TrialResult]:
        """Load results from JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "r") as f:
            data = json.load(f)

        results = []
        for item in data:
            config = HyperparameterConfig(**item["config"])
            result = TrialResult(
                config=config,
                total_time=item["total_time"],
                num_proofs_attempted=item["num_proofs_attempted"],
                num_proofs_succeeded=item["num_proofs_succeeded"],
                proofs_per_second=item["proofs_per_second"],
                success_rate=item["success_rate"],
                avg_proof_time=item["avg_proof_time"],
                avg_steps_per_proof=item["avg_steps_per_proof"],
                error_message=item.get("error_message"),
            )
            results.append(result)

        return results

    def print_summary(self, results: Optional[List[TrialResult]] = None) -> None:
        """Print summary of search results."""
        if results is None:
            results = self.results

        if not results:
            logger.warning("No results to summarize")
            return

        print("\n" + "=" * 80)
        print("HYPERPARAMETER SEARCH SUMMARY")
        print("=" * 80)
        print(f"Total trials: {len(results)}")
        print(f"Hardware profile: {self.hardware_profile}")
        print()

        # Best overall
        best = max(results, key=lambda r: r.score)
        print("BEST CONFIGURATION:")
        print(f"  Score: {best.score:.4f}")
        print(f"  Proofs/second: {best.proofs_per_second:.4f}")
        print(f"  Success rate: {best.success_rate:.2%}")
        print(f"  Avg proof time: {best.avg_proof_time:.1f}s")
        print()
        print("  Parameters:")
        for key, value in asdict(best.config).items():
            if key not in [
                "model_name",
                "data_type",
                "save_training_data",
                "save_checkpoints",
                "use_wandb",
                "train_value_head",
                "use_final_reward",
            ]:
                print(f"    {key}: {value}")

        print()

        # Top 5
        print("TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(
            sorted(results, key=lambda r: r.score, reverse=True)[:5]
        ):
            print(
                f"  {i + 1}. score={result.score:.4f}, "
                f"p/s={result.proofs_per_second:.4f}, "
                f"rate={result.success_rate:.2%}, "
                f"workers={result.config.num_workers}, "
                f"batch={result.config.batch_size}"
            )

        print("=" * 80)

    def generate_config_for_hpc(
        self, best_config: HyperparameterConfig
    ) -> Dict[str, Any]:
        """
        Generate HPC-translated configuration from laptop benchmark results.

        Applies scaling factors based on known hardware differences:
        - More workers (higher core count)
        - Larger batches (more VRAM)
        - Higher iteration counts (more compute)
        """
        hpc_config = HyperparameterConfig(**asdict(best_config))

        # Scaling factors for A100 vs RTX 4060 laptop
        worker_scale = 3.0  # ~3x more workers feasible
        batch_scale = 2.0  # ~2x larger batches (80GB vs 8GB)
        iter_scale = 2.0  # ~2x more iterations

        hpc_config.num_workers = int(best_config.num_workers * worker_scale)
        hpc_config.batch_size = int(best_config.batch_size * batch_scale)
        hpc_config.num_iterations = int(best_config.num_iterations * iter_scale)
        hpc_config.num_tactics_to_expand = min(
            32, int(best_config.num_tactics_to_expand * 1.5)
        )

        return asdict(hpc_config)


def run_laptop_benchmark():
    """Quick benchmark for laptop hardware."""
    searcher = HyperparameterSearcher(hardware_profile="laptop")
    result = searcher.quick_benchmark(num_theorems=3)
    searcher.print_summary([result])
    return result


def run_grid_search(hardware_profile: str = "laptop", num_theorems: int = 50):
    """Run full grid search."""
    searcher = HyperparameterSearcher(hardware_profile=hardware_profile)

    # # Reduced search space for faster iteration
    # reduced_space = {
    #     "num_workers": [8, 10, 12],
    #     "batch_size": [8, 16],
    #     "num_tactics_to_expand": [8, 12],
    #     "num_iterations": [200, 300],
    # }

    results = searcher.grid_search(
        search_space=LAPTOP_SEARCH_SPACE,
        num_theorems=num_theorems,
        max_trials=100,
    )

    searcher.print_summary(results)
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter search for theorem proving"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["benchmark", "grid", "binary"],
        default="benchmark",
        help="Search mode",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=["laptop", "hpc"],
        default="laptop",
        help="Hardware profile",
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=50,
        help="Number of theorems per trial",
    )
    parser.add_argument(
        "--param",
        type=str,
        default="num_workers",
        help="Parameter for binary search",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Log to WandB",
    )

    args = parser.parse_args()

    searcher = HyperparameterSearcher(
        hardware_profile=args.hardware,
        use_wandb=args.use_wandb,
    )

    if args.mode == "benchmark":
        result = searcher.quick_benchmark(num_theorems=args.num_theorems)
        searcher.print_summary([result])
    elif args.mode == "grid":
        results = searcher.grid_search(num_theorems=args.num_theorems)
        searcher.print_summary(results)
    elif args.mode == "binary":
        # Define ranges for common parameters
        param_ranges = {
            "num_workers": (4, 16),
            "batch_size": (4, 32),
            "num_iterations": (50, 300),
            "num_tactics_to_expand": (4, 24),
        }
        if args.param in param_ranges:
            min_val, max_val = param_ranges[args.param]
            best_val, best_result = searcher.binary_search_parameter(
                args.param,
                min_val,
                max_val,
                num_theorems=args.num_theorems,
            )
            print(f"\nOptimal {args.param}: {best_val}")
            searcher.print_summary([best_result])
        else:
            print(f"Unknown parameter: {args.param}")
            print(f"Available: {list(param_ranges.keys())}")
