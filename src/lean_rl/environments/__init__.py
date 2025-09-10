"""
Environments package for lean_rl.

This package provides various environment implementations for interacting
with Lean theorem provers through LeanDojo.
"""

from .persistent_environment import PersistentLeanEnvironment

# Re-export classes from the main environment module
from ..environment import LeanEnvironment, StepResult

__all__ = [
    "PersistentLeanEnvironment",
    "LeanEnvironment",
    "StepResult",
]
