"""
Random Weighted Agent for LeanDojo Environment

This module implements a weighted random agent for theorem proving in Lean environments.
"""

import random
from typing import List, Dict, Any, Optional
from ..random.agent import RandomAgent
from ...environment import StepResult


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
        if step_result.action_result in ["success", "proof_finished"]:
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
