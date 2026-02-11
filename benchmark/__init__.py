"""
Benchmark suite for AlphaZero and Guided Rollout MCTS algorithms.

This package contains:
- run_benchmark: Main benchmark runner for 18 training configurations
- evaluate_benchmark: Test-set evaluation of completed benchmark runs
- plot_benchmark: Visualization of benchmark results
"""

from .run_benchmark import (
    BASE_PARAMS,
    SIZE_CONFIGS,
    ALGORITHMS,
    SEEDS,
    SIZES,
    NUM_EPOCHS,
    get_benchmark_dir,
    get_run_dir,
    get_all_runs,
    is_run_complete,
    get_completed_epochs,
)

__all__ = [
    "BASE_PARAMS",
    "SIZE_CONFIGS",
    "ALGORITHMS",
    "SEEDS",
    "SIZES",
    "NUM_EPOCHS",
    "get_benchmark_dir",
    "get_run_dir",
    "get_all_runs",
    "is_run_complete",
    "get_completed_epochs",
]
