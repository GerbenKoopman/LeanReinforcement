"""
Environment for interacting with LeanDojo via a Gymnasium extension.
"""

from typing import Any
from loguru import logger

from lean_dojo import (
    Dojo,
    TacticState,
    Theorem,
    ProofFinished,
    LeanError,
    DojoInitError,
)
from lean_dojo.interaction.dojo import DojoTacticTimeoutError
from ReProver.common import Corpus, Pos
from .dataloader import LeanDataLoader
import queue
from contextlib import contextmanager


class LeanDojoEnv:
    def __init__(
        self,
        corpus: Corpus,
        theorem: Theorem,
        theorem_pos: Pos,
        k: int = 10,
        timeout: int = 60,
    ):
        super().__init__()
        self.theorem = theorem
        self.theorem_pos = theorem_pos

        self.dataloader = LeanDataLoader(corpus)

        self.dojo = Dojo(theorem, timeout=timeout)

        self.reset()
        self.current_state = self.initial_state

    def _get_number_of_accessible_premises(self) -> int:
        premise_list = self.dataloader.get_premises(self.theorem, self.theorem_pos)
        return len(premise_list)

    def reset(self) -> None:
        try:
            _, self.initial_state = self.dojo.__enter__()
            assert isinstance(self.initial_state, TacticState)
        except DojoInitError as e:
            logger.error(f"Error during environment reset: {e}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during environment reset: {e}")
            raise e

    def step(self, action: str) -> tuple[str, float, bool, bool, dict[str, Any]]:
        # Interact with Lean
        assert isinstance(self.current_state, TacticState)

        try:
            next_state = self.dojo.run_tac(self.current_state, action)
        except DojoTacticTimeoutError:
            logger.warning(f"Tactic timed out: {action[:100]}")
            # Treat timeout as an error state
            next_state = LeanError(error="Tactic execution timed out")
        except Exception as e:
            logger.error(f"Error running tactic '{action[:100]}': {e}")
            next_state = LeanError(error=f"Exception: {str(e)}")

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
            raise ValueError(f"Unhandled state: {next_state}")

        return observation, reward, done, False, {}

    def close(self):
        """Explicitly clean up the last running 'lean' process."""
        if hasattr(self, "dojo") and self.dojo is not None:
            try:
                self.dojo.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"Warning: Error during Dojo close: {e}")

        self.dojo: Dojo = None  # type: ignore
        logger.info("Environment closed.")

    def __del__(self):
        """Ensure cleanup when object is garbage collected."""
        self.close()


class LeanDojoEnvPool:
    """
    Manages a pool of LeanDojoEnv instances for parallel execution.
    """

    def __init__(
        self,
        corpus: Corpus,
        theorem: Theorem,
        theorem_pos: Pos,
        num_workers: int = 4,
        k: int = 10,
        timeout: int = 60,
    ):
        self.pool = queue.Queue()
        self.envs = []
        for _ in range(num_workers):
            # Create new env instances
            # Note: This will spawn multiple Lean processes
            env = LeanDojoEnv(corpus, theorem, theorem_pos, k, timeout)
            self.envs.append(env)
            self.pool.put(env)

    @contextmanager
    def get_env(self):
        """
        Context manager to safely checkout and return an environment.
        Usage:
            with pool.get_env() as env:
                env.step(...)
        """
        env = self.pool.get()
        try:
            yield env
        finally:
            self.pool.put(env)

    def close(self):
        """Close all environments in the pool."""
        for env in self.envs:
            env.close()
