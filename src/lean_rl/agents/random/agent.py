"""
Random Agent for LeanDojo Environment

This module implements a simple random agent for theorem proving in Lean environments.
"""

import random
from typing import List, Dict, Any, Optional

from ...environment import StepResult
from ...agents import BaseAgent


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
        self.results: List[str] = []
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

        # Define possible result types
        result_types = ["success", "proof_finished", "proof_given_up", "error"]
        result_counts = {}
        for result_type in result_types:
            result_counts[result_type] = self.results.count(result_type)

        # Calculate success rate (actions that didn't error)
        success_actions = sum(1 for r in self.results if r != "error")
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
