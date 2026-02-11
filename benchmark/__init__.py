"""
Benchmark suite for AlphaZero and Guided Rollout MCTS algorithms.

This package contains:
- run_benchmark: Main benchmark runner for 18 training configurations
- evaluate_benchmark: Test-set evaluation of completed benchmark runs
- plot_benchmark: Visualization of benchmark results

Imports are lazy to avoid RuntimeWarning when running:
    python -m benchmark.run_benchmark
"""

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


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    """Lazy import to avoid circular / premature import of run_benchmark."""
    if name in __all__:
        from . import run_benchmark

        return getattr(run_benchmark, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
