"""
A wrapper for the LeanEnvironment to standardize agent-environment interaction.
"""

from typing import Optional, Tuple

from lean_dojo import TacticState, Theorem
from ....environment import LeanEnvironment, StepResult


class LeanEnvWrapper:
    """
    A wrapper for the LeanEnvironment to provide a consistent interface for
    training, testing, and evaluation loops.
    """

    def __init__(self, env: LeanEnvironment):
        self.env = env

    def step(
        self, action: str
    ) -> Tuple[Optional[TacticState], float, bool, StepResult]:
        """
        Takes a step in the environment.

        Args:
            action (str): The action to take.

        Returns:
            A tuple containing:
            - The next state (or None if the episode ends).
            - The reward.
            - A boolean indicating if the episode is done.
            - The full StepResult object.
        """
        result = self.env.step(action)
        return result.state, result.reward, result.done, result

    def reset(self, theorem: Theorem) -> Optional[TacticState]:
        """
        Resets the environment with a new theorem.

        Args:
            theorem (Theorem): The theorem to start the new episode with.

        Returns:
            The initial state of the new episode.
        """
        return self.env.reset(theorem)
