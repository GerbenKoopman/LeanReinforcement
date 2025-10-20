"""
Environment for interacting with LeanDojo via a Gymnasium extension.
"""

from typing import Any
import gymnasium as gym
import os

from lean_dojo import (
    Dojo,
    TacticState,
    Theorem,
    TacticState,
    ProofFinished,
    LeanError,
    TacticState,
)
from ReProver.common import Corpus, Pos


class LeanDojoEnv(gym.Env):
    def __init__(self, theorem: Theorem, theorem_pos: Pos, k: int = 10):
        super().__init__()
        self.theorem = theorem
        self.theorem_pos = theorem_pos

        benchmark_dir = os.getenv("BENCHMARK_DIR", "")
        self.jsonl_path = os.path.join(
            benchmark_dir, "leandojo_benchmark_4/corpus.jsonl"
        )

        self.dojo = Dojo(self.theorem)
        self.observation_space = gym.spaces.Text(max_length=10000)

        # Action space for selecting k premises from an indexed library of size N
        N = self._get_number_of_accessible_premises()
        self.action_space = gym.spaces.MultiDiscrete([N] * k)

        self.reset()
        self.current_state = self.initial_state

    def _get_number_of_accessible_premises(
        self,
    ) -> int:
        corpus = Corpus(self.jsonl_path)

        accessible_premises = corpus.get_accessible_premises(
            str(self.theorem.file_path), self.theorem_pos
        )
        return len(accessible_premises)

    def reset(self, *, seed=None, options=None) -> tuple[str, dict[str, Any]]:
        super().reset(seed=seed)
        self.dojo_instance, self.initial_state = self.dojo.__enter__()
        assert isinstance(self.initial_state, TacticState)
        observation = self.initial_state.pp
        return observation, {}

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        # Interact with Lean
        assert isinstance(self.current_state, TacticState)
        next_state = self.dojo_instance.run_tac(self.current_state, action)
        self.current_state = next_state

        if isinstance(next_state, LeanError):  # Error occurred
            reward = -0.1
            done = True
            observation = str(next_state)
        elif isinstance(next_state, ProofFinished):  # No goals left
            reward = 1.0
            done = True
            observation = str(next_state)
        elif isinstance(next_state, TacticState):  # Proof still ongoing
            reward = 0.1
            done = False
            observation = next_state.pp
        else:
            # Edge case state, next_state is of type ProofGivenUp which
            # only occurs when it contains a sorry
            raise ValueError(f"Unhandled state: {next_state}")

        return observation, reward, done, False, {}
