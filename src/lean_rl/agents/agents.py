"""
Base Agent for LeanDojo Environment

This module implements an abstract base agent for theorem proving in Lean environments.
"""

from typing import Union
from abc import ABC, abstractmethod

from ..environment import StepResult


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def select_action(self, state, **kwargs) -> Union[str, None]:
        """Select an action given the current state."""
        pass

    @abstractmethod
    def update(self, step_result: StepResult) -> None:
        """Update the agent based on the step result."""
        pass

    def reset(self) -> None:
        """Reset the agent for a new episode."""
        pass

    def end_episode(self, episode_reward: float) -> None:
        """Called at the end of an episode.

        Args:
            episode_reward: Total reward accumulated during the episode.
        """
        pass
