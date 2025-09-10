"""
This module implements a persistent Lean environment, inspired by lean-gym,
to reduce the overhead of starting a new Lean process for each theorem.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

from lean_dojo import (
    Dojo,
    LeanGitRepo,
    Theorem,
    TacticState,
    LeanError,
    ProofFinished,
    ProofGivenUp,
)
from lean_dojo.interaction.dojo import (
    DojoTacticTimeoutError,
    DojoCrashError,
    DojoInitError,
)

from ..exceptions import (
    EnvironmentInitializationError,
    EnvironmentResetError,
    DojoEnvironmentError,
    create_dojo_error,
)

logger = logging.getLogger(__name__)


class PersistentLeanEnvironment:
    """
    A Lean environment that maintains a persistent Dojo process,
    inspired by lean-gym, to reduce reset overhead.
    """

    def __init__(self, repo: LeanGitRepo, timeout: int = 300):
        """Initialize the persistent Lean environment.

        Args:
            repo: LeanGitRepo instance for the repository
            timeout: Timeout for Dojo operations in seconds

        Raises:
            EnvironmentInitializationError: If initialization fails
        """
        self.repo = repo
        self.timeout = timeout
        self.dojo: Optional[Dojo] = None
        self.current_theorem: Optional[Theorem] = None
        self.current_state: Optional[TacticState] = None

        logger.info(f"Initializing PersistentLeanEnvironment with timeout={timeout}s")

    def _init_dojo(self, theorem: Theorem) -> Tuple[Dojo, TacticState]:
        """Initialize a Dojo instance for a specific theorem.

        Args:
            theorem: Theorem to initialize Dojo with

        Returns:
            Tuple of (dojo_instance, initial_state)

        Raises:
            EnvironmentInitializationError: If Dojo initialization fails
        """
        try:
            logger.debug(f"Initializing Dojo for theorem: {theorem.full_name}")
            dojo_context = Dojo(theorem, timeout=self.timeout)
            dojo, initial_state = dojo_context.__enter__()

            if not isinstance(initial_state, TacticState):
                raise EnvironmentInitializationError(
                    f"Expected TacticState, got {type(initial_state)}"
                )

            return dojo, initial_state

        except (DojoInitError, DojoCrashError, DojoTacticTimeoutError) as e:
            raise create_dojo_error(
                f"Failed to initialize Dojo for theorem {theorem.full_name}",
                original_exception=e,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        self._cleanup_dojo()

    def _cleanup_dojo(self) -> None:
        """Safely clean up the current Dojo instance."""
        if self.dojo is not None:
            try:
                self.dojo.__exit__(None, None, None)
                logger.debug("Dojo cleaned up successfully")
            except Exception as e:
                logger.warning(f"Error during Dojo cleanup: {e}")
            finally:
                self.dojo = None
                self.current_state = None

    def reset(self, theorem: Theorem) -> Optional[TacticState]:
        """Reset the environment to a new theorem state.

        Args:
            theorem: Theorem to reset to

        Returns:
            Initial tactic state for the theorem, or None if reset failed

        Raises:
            EnvironmentResetError: If reset fails critically
        """
        logger.info(f"Resetting environment to theorem: {theorem.full_name}")
        self.current_theorem = theorem

        # Clean up existing dojo
        self._cleanup_dojo()

        try:
            # Create new Dojo instance for the theorem
            self.dojo, initial_state = self._init_dojo(theorem)
            self.current_state = initial_state
            logger.debug(
                f"Successfully reset to theorem with {initial_state.num_goals} goals"
            )
            return self.current_state

        except (EnvironmentInitializationError, DojoEnvironmentError) as e:
            logger.error(f"Failed to reset environment: {e}")
            self._cleanup_dojo()
            return None
        except Exception as e:
            logger.error(f"Unexpected error during reset: {e}")
            self._cleanup_dojo()
            raise EnvironmentResetError(f"Unexpected error during reset: {e}")

    def step(
        self, action: str
    ) -> Tuple[Optional[TacticState], float, bool, Dict[str, Union[str, int]]]:
        """Execute a tactic and return the result.

        Args:
            action: Tactic string to execute

        Returns:
            Tuple of (new_state, reward, done, info)

        Raises:
            DojoEnvironmentError: If environment is not in valid state
        """
        if self.dojo is None or self.current_state is None:
            raise DojoEnvironmentError(
                "Environment is not reset or not in a valid tactic state"
            )

        logger.debug(f"Executing tactic: {action}")

        try:
            res = self.dojo.run_tac(self.current_state, action)
        except (DojoCrashError, DojoTacticTimeoutError) as e:
            logger.warning(f"Dojo crashed during tactic execution: {e}")
            self._cleanup_dojo()
            return (
                None,
                -1.0,
                True,
                {"action_result": "dojo_crash", "error": str(e), "num_goals": 0},
            )

        # Process the result
        done = False
        reward = 0.0
        num_goals = (
            getattr(self.current_state, "num_goals", 0) if self.current_state else 0
        )

        if isinstance(res, TacticState):
            self.current_state = res
            reward = -0.01  # Small step penalty
            num_goals = res.num_goals
            info = {
                "action_result": "success",
                "num_goals": num_goals,
                "state_changed": True,
            }
        elif isinstance(res, ProofFinished):
            self.current_state = None
            done = True
            reward = 1.0
            info = {
                "action_result": "proof_finished",
                "num_goals": 0,
                "state_changed": True,
            }
            logger.info("Proof completed successfully!")
        elif isinstance(res, LeanError):
            # Don't change state on error, agent can retry
            reward = -0.1  # Penalty for error
            info = {
                "action_result": "lean_error",
                "error": res.error,
                "num_goals": num_goals,
                "state_changed": False,
            }
            logger.debug(f"Lean error: {res.error}")
        elif isinstance(res, ProofGivenUp):
            self.current_state = None
            done = True
            reward = -1.0
            info = {
                "action_result": "proof_given_up",
                "num_goals": 0,
                "state_changed": True,
            }
            logger.debug("Proof given up")
        else:
            # Unexpected result type
            logger.error(f"Unexpected result type: {type(res)}")
            info = {
                "action_result": "unknown",
                "result_type": str(type(res)),
                "num_goals": num_goals,
                "state_changed": False,
            }

        return self.current_state, reward, done, info
