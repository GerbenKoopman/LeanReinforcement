"""
Environment for interacting with LeanDojo via a Gymnasium extension.
"""

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
from ReProver.common import Pos


class LeanDojoEnv:
    def __init__(
        self,
        theorem: Theorem,
        theorem_pos: Pos,
        timeout: int = 60,
    ):
        super().__init__()
        self.theorem = theorem
        self.theorem_pos = theorem_pos

        self.dojo = Dojo(theorem, timeout=timeout)

        self.reset()
        self.current_state = self.initial_state

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

    def step(self, action: str) -> tuple[str, float, bool]:
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
            reward = -1.0
            done = True
            observation = str(next_state)
        elif isinstance(next_state, ProofFinished):  # No goals left
            reward = 1.0
            done = True
            observation = str(next_state)
        elif isinstance(next_state, TacticState):  # Proof still ongoing
            reward = 0.0
            done = False
            observation = next_state.pp
        else:
            raise ValueError(f"Unhandled state: {next_state}")

        return observation, reward, done

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
