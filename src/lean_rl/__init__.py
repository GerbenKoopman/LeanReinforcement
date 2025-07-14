"""
LeanDojo Reinforcement Learning Package

This package provides tools for training reinforcement learning agents
on theorem proving tasks using LeanDojo.
"""

from .environment import LeanEnvironment, StepResult
from .agents import (
    BaseAgent,
    RandomAgent,
    WeightedRandomAgent,
    MCTSAgent,
    HierarchicalTransformerAgent,
)

__all__ = [
    "LeanEnvironment",
    "StepResult",
    "BaseAgent",
    "RandomAgent",
    "WeightedRandomAgent",
    "MCTSAgent",
    "HierarchicalTransformerAgent",
]
