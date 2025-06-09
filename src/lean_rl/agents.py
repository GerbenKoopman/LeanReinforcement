"""
Random Agent for LeanDojo Environment

This module implements a simple random agent that selects tactics randomly
from a predefined set. This serves as a baseline and test case for the RL environment.
"""

import random
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from .environment import LeanEnvironment, StepResult, ActionResult


class BaseAgent(ABC):
    """Abstract base class for RL agents."""

    @abstractmethod
    def select_action(self, state, **kwargs) -> str:
        """Select an action given the current state."""
        pass

    @abstractmethod
    def update(self, step_result: StepResult) -> None:
        """Update the agent based on the step result."""
        pass

    def reset(self) -> None:
        """Reset the agent for a new episode."""
        pass


class RandomAgent(BaseAgent):
    """
    A random agent that selects tactics randomly from a predefined set.

    This agent serves as a baseline and can be used to test the environment
    and gather statistics on random exploration.
    """

    def __init__(
        self,
        tactics: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the random agent.

        Args:
            tactics: List of tactics to choose from. If None, uses default set.
            weights: Probability weights for each tactic. If None, uses uniform.
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Default set of common Lean tactics
        if tactics is None:
            tactics = [
                # Basic tactics
                "rfl",
                "trivial",
                "simp",
                "assumption",
                "exact ?_",
                # Introduction tactics
                "intro",
                "intros",
                "constructor",
                "left",
                "right",
                "use ?_",
                "exists ?_",
                # Elimination tactics
                "cases ?_",
                "induction ?_",
                "apply ?_",
                "have : ?_ := ?_",
                # Rewriting
                "rw [?_]",
                "rw [← ?_]",
                "simp only [?_]",
                "simp at ?_",
                # Logic tactics
                "by_contra",
                "by_cases ?_",
                "contrapose",
                "exfalso",
                # Arithmetic/algebra
                "ring",
                "linarith",
                "norm_num",
                "norm_cast",
                # Advanced tactics
                "tauto",
                "aesop",
                "decide",
                "omega",
                "abel",
                # Meta tactics
                "skip",
                "sorry",
            ]

        self.tactics = tactics
        self.weights = weights

        # Statistics tracking
        self.actions_taken: List[str] = []
        self.results: List[ActionResult] = []
        self.episode_count = 0

    def select_action(self, state, **kwargs) -> str:
        """
        Select a random tactic.

        Args:
            state: Current state (ignored for random agent)
            **kwargs: Additional arguments (ignored)

        Returns:
            Randomly selected tactic string
        """
        action = random.choices(self.tactics, weights=self.weights, k=1)[0]
        self.actions_taken.append(action)
        return action

    def update(self, step_result: StepResult) -> None:
        """
        Update agent statistics with the step result.

        Args:
            step_result: Result from the environment step
        """
        self.results.append(step_result.action_result)

    def reset(self) -> None:
        """Reset for a new episode."""
        self.episode_count += 1

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        if not self.results:
            return {"episodes": self.episode_count, "total_actions": 0}

        total_actions = len(self.results)
        result_counts = {}
        for result in ActionResult:
            result_counts[result.value] = self.results.count(result)

        # Calculate success rate (actions that didn't error)
        success_actions = sum(1 for r in self.results if r != ActionResult.ERROR)
        success_rate = success_actions / total_actions if total_actions > 0 else 0

        # Most common tactics
        tactic_counts = {}
        for tactic in self.actions_taken:
            tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1

        most_common = sorted(tactic_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "episodes": self.episode_count,
            "total_actions": total_actions,
            "success_rate": success_rate,
            "result_counts": result_counts,
            "most_common_tactics": most_common,
            "unique_tactics_used": len(tactic_counts),
        }

    def clear_statistics(self) -> None:
        """Clear all collected statistics."""
        self.actions_taken.clear()
        self.results.clear()
        self.episode_count = 0


class WeightedRandomAgent(RandomAgent):
    """
    Random agent that adapts tactic weights based on success rates.

    This agent starts with uniform weights but gradually adjusts them
    based on which tactics lead to successful outcomes.
    """

    def __init__(
        self,
        tactics: Optional[List[str]] = None,
        learning_rate: float = 0.01,
        min_weight: float = 0.001,
        seed: Optional[int] = None,
    ):
        """
        Initialize the weighted random agent.

        Args:
            tactics: List of tactics to choose from
            learning_rate: How quickly to adapt weights
            min_weight: Minimum weight for any tactic
            seed: Random seed
        """
        super().__init__(tactics=tactics, seed=seed)

        self.learning_rate = learning_rate
        self.min_weight = min_weight

        # Initialize uniform weights
        self.weights = [1.0] * len(self.tactics)

        # Track success rates per tactic
        self.tactic_attempts = {tactic: 0 for tactic in self.tactics}
        self.tactic_successes = {tactic: 0 for tactic in self.tactics}

        # Track last action for updates
        self.last_action: Optional[str] = None

    def select_action(self, state, **kwargs) -> str:
        """Select action using current weights."""
        action = random.choices(self.tactics, weights=self.weights, k=1)[0]
        self.last_action = action
        self.actions_taken.append(action)
        self.tactic_attempts[action] += 1
        return action

    def update(self, step_result: StepResult) -> None:
        """Update weights based on action result."""
        super().update(step_result)

        if self.last_action is None:
            return

        # Update success count
        if step_result.action_result in [
            ActionResult.SUCCESS,
            ActionResult.PROOF_FINISHED,
        ]:
            self.tactic_successes[self.last_action] += 1

        # Update weights based on success rate
        self._update_weights()
        self.last_action = None

    def _update_weights(self) -> None:
        """Update tactic weights based on success rates."""
        for i, tactic in enumerate(self.tactics):
            attempts = self.tactic_attempts[tactic]
            if attempts > 0:
                success_rate = self.tactic_successes[tactic] / attempts
                # Exponential moving average of weights
                target_weight = max(success_rate, self.min_weight)
                self.weights[i] = (1 - self.learning_rate) * self.weights[
                    i
                ] + self.learning_rate * target_weight

        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]

    def get_statistics(self) -> Dict[str, Any]:
        """Get extended statistics including weights."""
        stats = super().get_statistics()

        # Add weight information
        tactic_stats = []
        for i, tactic in enumerate(self.tactics):
            attempts = self.tactic_attempts[tactic]
            successes = self.tactic_successes[tactic]
            success_rate = successes / attempts if attempts > 0 else 0

            tactic_stats.append(
                {
                    "tactic": tactic,
                    "weight": self.weights[i],
                    "attempts": attempts,
                    "successes": successes,
                    "success_rate": success_rate,
                }
            )

        # Sort by weight
        tactic_stats.sort(key=lambda x: x["weight"], reverse=True)

        stats["tactic_weights"] = tactic_stats[:10]  # Top 10 by weight
        return stats
