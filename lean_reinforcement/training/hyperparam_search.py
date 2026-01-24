"""
Hyperparameter search module for optimizing proofs/hour on different hardware.

This module provides grid search and binary search capabilities to find optimal
hyperparameters for theorem proving. Results are designed to be transferable
from local testing (laptop) to HPC clusters (like Snellius).

Key metrics optimized:
- Proofs per hour (throughput) - PRIMARY METRIC
- Success rate (accuracy)
- GPU/CPU utilization efficiency

Hardware profiles:
- laptop: Constrained VRAM/RAM (e.g., RTX 4060 with 8GB VRAM)
- hpc: High-end hardware (e.g., A100 with 80GB VRAM)

Search strategies:
- Grid search: Exhaustive search (expensive)
- Binary search: Per-dimension optimization assuming independence (efficient)
- Coordinate descent: Iterative per-dimension binary search
"""

import time
import json
import random
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict, field
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
    def proofs_per_hour(self) -> float:
        """Proofs per hour - PRIMARY METRIC for optimization."""
        return self.proofs_per_second * 3600.0

    @property
    def score(self) -> float:
        """
        Combined score for ranking trials.

        Uses proofs per hour as the primary metric.
        Higher is better.
        """
        # Proofs per hour is our primary metric
        return self.proofs_per_hour

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "total_time": self.total_time,
            "num_proofs_attempted": self.num_proofs_attempted,
            "num_proofs_succeeded": self.num_proofs_succeeded,
            "proofs_per_second": self.proofs_per_second,
            "proofs_per_hour": self.proofs_per_hour,
            "success_rate": self.success_rate,
            "avg_proof_time": self.avg_proof_time,
            "avg_steps_per_proof": self.avg_steps_per_proof,
            "score": self.score,
            "error_message": self.error_message,
        }


# Hardware-specific default configurations
# NOTE: For hyperparameter search, we use shorter timeouts than production
# to allow faster iteration. Production runs can use longer timeouts.
LAPTOP_DEFAULTS = HyperparameterConfig(
    num_workers=10,  # Avoid thermal throttling
    batch_size=16,  # Saturate RTX 4060
    num_tactics_to_expand=12,  # Reduce Lean executions
    num_iterations=100,  # Minimum viable for AlphaZero
    max_time=120.0,  # 2 minutes per MCTS step (reduced for search)
    max_steps=40,  # Reasonable depth
    proof_timeout=300.0,  # 5 minutes per theorem (reduced for search)
    env_timeout=60,  # 1 minute per tactic (reduced for search)
    max_rollout_depth=30,
)

HPC_DEFAULTS = HyperparameterConfig(
    num_workers=32,  # Utilize full node
    batch_size=32,  # Larger batches for A100
    num_tactics_to_expand=32,  # Full expansion
    num_iterations=400,  # Deeper search
    max_time=180.0,  # 3 minutes per MCTS step (reduced for search)
    max_steps=50,  # Allow longer proofs
    proof_timeout=600.0,  # 10 minutes per theorem (reduced for search)
    env_timeout=120,  # 2 minutes per tactic (reduced for search)
    max_rollout_depth=50,
)


# Search spaces for grid search
LAPTOP_SEARCH_SPACE: Dict[str, List[Any]] = {
    # "num_workers": [6, 8, 10, 12],
    "batch_size": [8, 16, 24],
    "num_tactics_to_expand": [8, 12, 16],
    "num_iterations": [200, 300, 400],
    "max_time": [60.0, 120.0, 180.0],
    "max_steps": [20, 40, 60],
    "proof_timeout": [300.0, 600.0, 900.0],
    "env_timeout": [30, 60, 120],
}

HPC_SEARCH_SPACE: Dict[str, List[Any]] = {
    "num_workers": [16, 24, 32, 48],
    "batch_size": [16, 32, 48],
    "num_tactics_to_expand": [16, 24, 32],
    "num_iterations": [200, 300, 400],
}

# =============================================================================
# BINARY SEARCH PARAMETER DEFINITIONS
# =============================================================================
# Parameters ordered by logical dependencies:
# 1. Resource parameters (affect capacity): num_workers, batch_size
# 2. Search behavior (depend on resource capacity): num_tactics_to_expand, num_iterations
# 3. Timeout parameters (depend on search behavior): env_timeout, max_time, proof_timeout
# 4. Search depth parameters: max_steps, max_rollout_depth


@dataclass
class ParameterRange:
    """Definition of a parameter's search range and properties."""

    name: str
    min_val: float
    max_val: float
    is_integer: bool = True
    description: str = ""
    # Dependencies: parameters that should be optimized before this one
    depends_on: List[str] = field(default_factory=list)


# Laptop parameter ranges - ordered by dependency
LAPTOP_PARAMETER_RANGES: List[ParameterRange] = [
    # === TIER 1: Resource/Capacity Parameters (no dependencies) ===
    ParameterRange(
        name="num_workers",
        min_val=4,
        max_val=16,
        is_integer=True,
        description="Number of parallel Lean workers",
        depends_on=[],
    ),
    ParameterRange(
        name="batch_size",
        min_val=4,
        max_val=32,
        is_integer=True,
        description="Inference batch size for GPU",
        depends_on=[],
    ),
    # === TIER 2: Search Behavior (depend on resources) ===
    ParameterRange(
        name="num_tactics_to_expand",
        min_val=4,
        max_val=24,
        is_integer=True,
        description="Tactics expanded per MCTS node",
        depends_on=["batch_size"],  # Should be <= batch_size for efficiency
    ),
    ParameterRange(
        name="num_iterations",
        min_val=50,
        max_val=500,
        is_integer=True,
        description="MCTS iterations per search step",
        depends_on=["num_workers"],  # More workers can support more iterations
    ),
    # === TIER 3: Timeout Parameters (form a hierarchy) ===
    # NOTE: For hyperparameter search, we use shorter ranges to avoid long waits
    ParameterRange(
        name="env_timeout",
        min_val=30,
        max_val=120,
        is_integer=True,
        description="Max seconds per tactic execution",
        depends_on=[],  # Base timeout, no dependencies
    ),
    ParameterRange(
        name="max_time",
        min_val=60.0,
        max_val=300.0,
        is_integer=False,
        description="Max seconds per MCTS search step",
        depends_on=["env_timeout"],  # Should be > env_timeout
    ),
    ParameterRange(
        name="proof_timeout",
        min_val=120.0,
        max_val=600.0,
        is_integer=False,
        description="Max seconds for entire proof search",
        depends_on=["max_time"],  # Should be > max_time * max_steps
    ),
    # === TIER 4: Search Depth Parameters ===
    ParameterRange(
        name="max_steps",
        min_val=10,
        max_val=60,
        is_integer=True,
        description="Maximum proof depth (steps)",
        depends_on=["num_iterations"],
    ),
    ParameterRange(
        name="max_rollout_depth",
        min_val=10,
        max_val=50,
        is_integer=True,
        description="Max depth for MCTS rollouts",
        depends_on=["max_steps"],
    ),
]

# HPC parameter ranges - scaled for more resources
HPC_PARAMETER_RANGES: List[ParameterRange] = [
    # === TIER 1: Resource/Capacity Parameters ===
    ParameterRange(
        name="num_workers",
        min_val=16,
        max_val=64,
        is_integer=True,
        description="Number of parallel Lean workers",
        depends_on=[],
    ),
    ParameterRange(
        name="batch_size",
        min_val=16,
        max_val=64,
        is_integer=True,
        description="Inference batch size for GPU",
        depends_on=[],
    ),
    # === TIER 2: Search Behavior ===
    ParameterRange(
        name="num_tactics_to_expand",
        min_val=8,
        max_val=48,
        is_integer=True,
        description="Tactics expanded per MCTS node",
        depends_on=["batch_size"],
    ),
    ParameterRange(
        name="num_iterations",
        min_val=100,
        max_val=800,
        is_integer=True,
        description="MCTS iterations per search step",
        depends_on=["num_workers"],
    ),
    # === TIER 3: Timeout Parameters ===
    # NOTE: For hyperparameter search, we use shorter ranges to avoid long waits
    ParameterRange(
        name="env_timeout",
        min_val=30,
        max_val=180,
        is_integer=True,
        description="Max seconds per tactic execution",
        depends_on=[],
    ),
    ParameterRange(
        name="max_time",
        min_val=60.0,
        max_val=360.0,
        is_integer=False,
        description="Max seconds per MCTS search step",
        depends_on=["env_timeout"],
    ),
    ParameterRange(
        name="proof_timeout",
        min_val=180.0,
        max_val=900.0,
        is_integer=False,
        description="Max seconds for entire proof search",
        depends_on=["max_time"],
    ),
    # === TIER 4: Search Depth Parameters ===
    ParameterRange(
        name="max_steps",
        min_val=20,
        max_val=80,
        is_integer=True,
        description="Maximum proof depth (steps)",
        depends_on=["num_iterations"],
    ),
    ParameterRange(
        name="max_rollout_depth",
        min_val=20,
        max_val=80,
        is_integer=True,
        description="Max depth for MCTS rollouts",
        depends_on=["max_steps"],
    ),
]


def topological_sort_parameters(params: List[ParameterRange]) -> List[ParameterRange]:
    """Sort parameters so dependencies come before dependents."""
    # Build dependency graph
    param_dict = {p.name: p for p in params}
    visited = set()
    result = []

    def visit(name: str):
        if name in visited:
            return
        visited.add(name)
        param = param_dict.get(name)
        if param:
            for dep in param.depends_on:
                if dep in param_dict:
                    visit(dep)
            result.append(param)

    for p in params:
        visit(p.name)

    return result


class HyperparameterSearcher:
    """
    Orchestrates hyperparameter search to optimize proofs/hour.

    Supports:
    - Grid search: Exhaustive search over parameter combinations
    - Binary search: Efficient search for optimal value of single parameter
    - Coordinate descent: Sequential binary search per dimension (assumes independence)

    The coordinate descent method is most efficient for large search spaces,
    requiring O(k * log(n)) trials instead of O(n^k) for grid search.
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

        # Typed attributes for static checkers
        self.default_config: HyperparameterConfig
        self.search_space: Dict[str, List[Any]]
        self.parameter_ranges: List[ParameterRange]

        # Set defaults based on hardware profile
        if hardware_profile == "laptop":
            self.default_config = LAPTOP_DEFAULTS
            self.search_space = LAPTOP_SEARCH_SPACE
            self.parameter_ranges = topological_sort_parameters(LAPTOP_PARAMETER_RANGES)
        else:
            self.default_config = HPC_DEFAULTS
            self.search_space = HPC_SEARCH_SPACE
            self.parameter_ranges = topological_sort_parameters(HPC_PARAMETER_RANGES)

        self.results: List[TrialResult] = []

        # Cache for coordinate descent - stores best values found so far
        self.best_config_cache: Dict[str, Any] = {}

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

        Returns metrics with proofs_per_hour as the primary optimization target.
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
            training_config.num_epochs = 1

            # Get metrics from trainer (now returns List[Dict])
            metrics_list = trainer.train()

            # Aggregate metrics from all workers
            # Metrics keys are: proof_search/success, proof_search/steps, proof_search/time
            if metrics_list:
                for m in metrics_list:
                    if m.get("proof_search/success", False):
                        num_succeeded += 1
                    total_steps += m.get("proof_search/steps", 0)

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Trial failed: {e}")

        total_time = time.time() - start_time
        num_attempted = num_theorems

        # Calculate metrics - proofs_per_hour is derived from proofs_per_second
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

        logger.info(
            f"Trial complete: {num_succeeded}/{num_attempted} proofs "
            f"({result.proofs_per_hour:.1f} proofs/hour, "
            f"{success_rate:.1%} success rate)"
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
        # Ensure we have a concrete search space for static checkers
        space: Dict[str, List[Any]] = (
            search_space if search_space is not None else self.search_space
        )

        # Generate all combinations
        param_names = list(space.keys())
        param_values = [space[name] for name in param_names]
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
                        "search_space": space,
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

    def coordinate_descent_search(
        self,
        num_theorems: int = 50,
        max_iterations_per_param: int = 8,
        tolerance: float = 0.1,
        params_to_search: Optional[List[str]] = None,
        num_rounds: int = 1,
    ) -> Tuple[HyperparameterConfig, List[TrialResult]]:
        """
        Coordinate descent optimization: binary search each dimension sequentially.

        This method assumes hyperparameters are approximately independent.
        It optimizes each parameter in turn using binary search, carrying
        forward the best values found for previous parameters.

        Parameters are searched in dependency order (dependencies first).

        Complexity: O(k * log(n) * num_rounds) trials instead of O(n^k) for grid search.

        Args:
            num_theorems: Number of theorems per trial.
            max_iterations_per_param: Max binary search iterations per parameter.
            tolerance: Convergence tolerance for binary search.
            params_to_search: List of parameter names to optimize (None = all).
            num_rounds: Number of full passes over all parameters.
                       More rounds can refine results if independence assumption
                       is imperfect.

        Returns:
            Tuple of (best_config, all_results).
        """
        logger.info("=" * 60)
        logger.info("COORDINATE DESCENT HYPERPARAMETER SEARCH")
        logger.info("Metric: proofs per hour")
        logger.info(f"Hardware profile: {self.hardware_profile}")
        logger.info(f"Theorems per trial: {num_theorems}")
        logger.info(f"Max iterations per param: {max_iterations_per_param}")
        logger.info(f"Number of rounds: {num_rounds}")
        logger.info("=" * 60)

        if self.use_wandb and WANDB_AVAILABLE and wandb is not None:
            try:
                wandb.init(
                    project=self.wandb_project,
                    config={
                        "search_type": "coordinate_descent",
                        "hardware_profile": self.hardware_profile,
                        "num_theorems": num_theorems,
                        "max_iterations_per_param": max_iterations_per_param,
                        "num_rounds": num_rounds,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
                self.use_wandb = False

        # Start with default config
        current_config = HyperparameterConfig(**asdict(self.default_config))
        all_results: List[TrialResult] = []

        # Filter parameters if specified
        if params_to_search is not None:
            param_ranges = [
                p for p in self.parameter_ranges if p.name in params_to_search
            ]
        else:
            param_ranges = self.parameter_ranges

        logger.info(f"Optimizing {len(param_ranges)} parameters in order:")
        for i, p in enumerate(param_ranges):
            deps = f" (depends on: {p.depends_on})" if p.depends_on else ""
            logger.info(f"  {i+1}. {p.name}: [{p.min_val}, {p.max_val}]{deps}")

        # Run baseline with default config
        logger.info("\n--- Running baseline with default config ---")
        baseline_result = self._run_single_trial(
            HyperparameterConfig(**asdict(current_config)), num_theorems
        )
        all_results.append(baseline_result)
        best_score = baseline_result.score
        logger.info(f"Baseline: {baseline_result.proofs_per_hour:.1f} proofs/hour")

        # Multiple rounds of coordinate descent
        for round_num in range(num_rounds):
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num + 1}/{num_rounds}")
            logger.info(f"{'='*60}")

            round_improvements = 0

            for param in param_ranges:
                logger.info(f"\n--- Optimizing {param.name} ---")
                logger.info(f"Range: [{param.min_val}, {param.max_val}]")
                logger.info(f"Current value: {getattr(current_config, param.name)}")

                # Binary search for this parameter
                best_val, param_results = self._binary_search_single_param(
                    param=param,
                    base_config=current_config,
                    num_theorems=num_theorems,
                    max_iterations=max_iterations_per_param,
                    tolerance=tolerance,
                )

                all_results.extend(param_results)

                # Update config with best value
                old_val = getattr(current_config, param.name)
                if param.is_integer:
                    best_val = int(round(best_val))
                setattr(current_config, param.name, best_val)

                # Check improvement
                best_param_result = max(param_results, key=lambda r: r.score)
                if best_param_result.score > best_score:
                    improvement = best_param_result.score - best_score
                    best_score = best_param_result.score
                    round_improvements += 1
                    logger.info(
                        f"✓ {param.name}: {old_val} → {best_val} "
                        f"(+{improvement:.1f} proofs/hour)"
                    )
                else:
                    logger.info(f"  {param.name}: kept at {best_val} (no improvement)")

                # Save intermediate results
                self._save_results(all_results, "coordinate_descent_intermediate.json")

                # Log to WandB
                if self.use_wandb and wandb is not None:
                    wandb.log(
                        {
                            "round": round_num + 1,
                            "param": param.name,
                            f"param/{param.name}": best_val,
                            "best_score": best_score,
                            "proofs_per_hour": best_param_result.proofs_per_hour,
                        }
                    )

            logger.info(
                f"\nRound {round_num + 1} complete: {round_improvements} parameters improved"
            )
            logger.info(f"Current best: {best_score:.1f} proofs/hour")

            # Early stopping if no improvements in this round
            if round_improvements == 0 and round_num > 0:
                logger.info("No improvements in this round. Stopping early.")
                break

        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("COORDINATE DESCENT COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total trials: {len(all_results)}")
        logger.info(f"Best score: {best_score:.1f} proofs/hour")
        logger.info("Optimal configuration:")
        for param in param_ranges:
            logger.info(f"  {param.name}: {getattr(current_config, param.name)}")

        # Save final results
        self._save_results(all_results, "coordinate_descent_final.json")
        self._save_config(current_config, "optimal_config.json")

        if self.use_wandb and wandb is not None:
            wandb.finish()

        return current_config, all_results

    def _binary_search_single_param(
        self,
        param: ParameterRange,
        base_config: HyperparameterConfig,
        num_theorems: int,
        max_iterations: int,
        tolerance: float,
    ) -> Tuple[float, List[TrialResult]]:
        """
        Binary search for optimal value of a single parameter.

        Uses golden section search for unimodal optimization.

        Returns:
            Tuple of (best_value, list_of_results).
        """
        results: List[TrialResult] = []

        low = param.min_val
        high = param.max_val

        # Golden ratio for golden section search
        phi = (1 + 5**0.5) / 2

        # Initial test points using golden section
        x1 = high - (high - low) / phi
        x2 = low + (high - low) / phi

        # Evaluate initial points
        def evaluate(val: float) -> TrialResult:
            config = HyperparameterConfig(**asdict(base_config))
            actual_val = int(round(val)) if param.is_integer else val
            setattr(config, param.name, actual_val)
            result = self._run_single_trial(config, num_theorems)
            results.append(result)
            return result

        result1 = evaluate(x1)
        result2 = evaluate(x2)

        for iteration in range(max_iterations - 2):  # Already did 2 evaluations
            # Check convergence
            range_size = high - low
            if range_size / param.max_val < tolerance:
                break

            # For integer parameters, stop if range is too small
            if param.is_integer and range_size < 2:
                break

            if result1.score < result2.score:
                # Best is in [x1, high]
                low = x1
                x1 = x2
                result1 = result2
                x2 = low + (high - low) / phi
                result2 = evaluate(x2)
            else:
                # Best is in [low, x2]
                high = x2
                x2 = x1
                result2 = result1
                x1 = high - (high - low) / phi
                result1 = evaluate(x1)

            logger.debug(
                f"  Iteration {iteration + 3}: range=[{low:.1f}, {high:.1f}], "
                f"best_score={max(r.score for r in results):.1f}"
            )

        # Return best value found
        best_result = max(results, key=lambda r: r.score)
        best_val = getattr(best_result.config, param.name)

        return best_val, results

    def _save_config(self, config: HyperparameterConfig, filename: str) -> None:
        """Save configuration to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            json.dump(asdict(config), f, indent=2)
        logger.info(f"Config saved to {filepath}")

    def quick_benchmark(
        self,
        config: Optional[HyperparameterConfig] = None,
        num_theorems: int = 50,
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
        print(f"  Proofs/hour: {best.proofs_per_hour:.1f} (primary metric)")
        print(f"  Proofs/second: {best.proofs_per_second:.4f}")
        print(f"  Success rate: {best.success_rate:.2%}")
        print(f"  Avg proof time: {best.avg_proof_time:.1f}s")
        print(f"  Total time: {best.total_time:.1f}s")
        print(f"  Proofs: {best.num_proofs_succeeded}/{best.num_proofs_attempted}")
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
                f"  {i + 1}. proofs/hr={result.proofs_per_hour:.1f}, "
                f"rate={result.success_rate:.2%}, "
                f"workers={result.config.num_workers}, "
                f"batch={result.config.batch_size}, "
                f"tactics={result.config.num_tactics_to_expand}, "
                f"iters={result.config.num_iterations}"
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

    results = searcher.grid_search(
        search_space=LAPTOP_SEARCH_SPACE,
        num_theorems=num_theorems,
        max_trials=100,
    )

    searcher.print_summary(results)
    return results


def run_coordinate_descent(
    hardware_profile: str = "laptop",
    num_theorems: int = 50,
    params: Optional[List[str]] = None,
    num_rounds: int = 2,
):
    """
    Run coordinate descent hyperparameter search.

    This is the recommended search method for efficiency.
    It optimizes each parameter independently using binary search.
    """
    searcher = HyperparameterSearcher(hardware_profile=hardware_profile)

    best_config, results = searcher.coordinate_descent_search(
        num_theorems=num_theorems,
        params_to_search=params,
        num_rounds=num_rounds,
    )

    searcher.print_summary(results)
    return best_config, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter search for theorem proving (optimizes proofs/hour)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["benchmark", "grid", "binary", "coordinate"],
        default="benchmark",
        help="Search mode: benchmark (quick test), grid (exhaustive), "
        "binary (single param), coordinate (all params, efficient)",
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
        help="Parameter for binary search mode",
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="+",
        default=None,
        help="Parameters to optimize in coordinate descent (default: all)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=2,
        help="Number of rounds for coordinate descent",
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

    elif args.mode == "coordinate":
        best_config, results = searcher.coordinate_descent_search(
            num_theorems=args.num_theorems,
            params_to_search=args.params,
            num_rounds=args.num_rounds,
        )
        searcher.print_summary(results)
        print("\nOptimal config saved to hyperparam_results/optimal_config.json")

    elif args.mode == "binary":
        # Define ranges for common parameters
        param_ranges_dict = {
            p.name: (p.min_val, p.max_val) for p in searcher.parameter_ranges
        }

        if args.param in param_ranges_dict:
            min_val, max_val = param_ranges_dict[args.param]
            best_val, best_result = searcher.binary_search_parameter(
                args.param,
                min_val,
                max_val,
                num_theorems=args.num_theorems,
            )
            print(f"\nOptimal {args.param}: {best_val}")
            print(f"Proofs/hour: {best_result.proofs_per_hour:.1f}")
            searcher.print_summary([best_result])
        else:
            print(f"Unknown parameter: {args.param}")
            print(f"Available: {list(param_ranges_dict.keys())}")
