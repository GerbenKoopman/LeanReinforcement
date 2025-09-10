"""
LeanDojo Reinforcement Learning Package

This package provides tools for training reinforcement learning agents
on theorem proving tasks using LeanDojo.
"""

# Import environment classes
from .environment import LeanEnvironment, StepResult
from .environments.persistent_environment import PersistentLeanEnvironment
from .agents import (
    BaseAgent,
    RandomAgent,
    WeightedRandomAgent,
    MCTSAgent,
    HierarchicalTransformerAgent,
)
from .config import Config, get_default_config, setup_logging
from .exceptions import LeanRLError, EnvironmentError, AgentError, TrainingError

__version__ = "0.1.0"

__all__ = [
    # Environment classes
    "LeanEnvironment",
    "PersistentLeanEnvironment",
    "StepResult",
    # Agent classes
    "BaseAgent",
    "RandomAgent",
    "WeightedRandomAgent",
    "MCTSAgent",
    "HierarchicalTransformerAgent",
    # Configuration
    "Config",
    "get_default_config",
    "setup_logging",
    # Exceptions
    "LeanRLError",
    "EnvironmentError",
    "AgentError",
    "TrainingError",
    # Version
    "__version__",
]
