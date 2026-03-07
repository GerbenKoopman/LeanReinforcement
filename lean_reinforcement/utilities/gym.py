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
    ProofGivenUp,
)
from lean_dojo.interaction.dojo import DojoCrashError, DojoTacticTimeoutError
from ReProver.common import Pos

#: Max ``run_tac`` calls before the Lean REPL is restarted.
#: Lean keeps every previous state in memory, so periodic resets reclaim
#: all leaked Lean-side and pexpect-side memory (~2-5 s cost).
#: Set to 0 to disable. Override via ``LEAN_RL_DOJO_RESET_INTERVAL``.
import os

DOJO_RESET_INTERVAL: int = int(os.environ.get("LEAN_RL_DOJO_RESET_INTERVAL", "500"))


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
        self.dojo = None
        self.initial_state = None
        self.current_state = None
        self._timeout = timeout
        self._tac_count: int = 0  # total run_tac calls since last Dojo reset

        try:
            self.dojo = Dojo(theorem, timeout=timeout)
            self.reset()
            self.current_state = self.initial_state
        except Exception:
            # Ensure cleanup if initialization fails
            self.close()
            raise

    def reset(self) -> None:
        try:
            assert self.dojo is not None, "Dojo not initialized"
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

        next_state = self.run_tactic_stateless(self.current_state, action)
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

    def run_tactic_stateless(
        self, state: TacticState, action: str
    ) -> TacticState | ProofFinished | LeanError | ProofGivenUp:
        """Run a tactic on the given state without modifying env state.

        Transparently resets the Lean REPL when ``DOJO_RESET_INTERVAL``
        is reached (stale state objects will get back a ``LeanError``).
        """
        if self.dojo is None:
            return LeanError(error="Lean REPL is no longer running (crashed or OOM)")

        # --- Periodic Dojo reset to reclaim Lean REPL SID memory ---
        if DOJO_RESET_INTERVAL > 0 and self._tac_count >= DOJO_RESET_INTERVAL:
            self._reset_dojo()

        try:
            next_state = self.dojo.run_tac(state, action)
            self._tac_count += 1
        except DojoTacticTimeoutError:
            logger.warning(f"Tactic timed out: {action[:100]}")
            # Treat timeout as an error state
            next_state = LeanError(error="Tactic execution timed out")
        except DojoCrashError as e:
            if e.is_out_of_memory:
                logger.warning(
                    f"Lean REPL hit memory limit during tactic: {action[:100]}"
                )
            else:
                logger.error(
                    f"Lean REPL crashed (exit: {e}) during tactic: {action[:100]}"
                )
            # The Lean subprocess is dead — mark dojo as unusable so
            # the caller (MCTS/runner) stops sending more tactics.
            self.dojo = None
            next_state = LeanError(error=f"Lean REPL crashed: {e}")
        except Exception as e:
            logger.error(f"Error running tactic '{action[:100]}': {e}")
            next_state = LeanError(error=f"Exception: {str(e)}")

        return next_state

    def _reset_dojo(self) -> None:
        """Kill the current Lean REPL and start a fresh one.

        Reclaims all Lean-side SID state and pexpect pty buffers.
        If the reset fails, marks the environment as unusable.
        """
        logger.info(
            f"Resetting Dojo after {self._tac_count} run_tac calls "
            f"(interval={DOJO_RESET_INTERVAL})"
        )
        try:
            # Kill the old Lean process
            if self.dojo is not None:
                try:
                    self.dojo.__exit__(None, None, None)
                except Exception:
                    pass

            # Kill lean/lake orphans that escaped the parent-child tree.
            try:
                from lean_reinforcement.utilities.memory import (
                    kill_child_processes,
                    kill_lean_orphans,
                )

                kill_child_processes()
                kill_lean_orphans()
            except Exception:
                pass

            # Reclaim Python objects from the old Lean session before
            # allocating the new process.
            import gc

            gc.collect()

            # Start a fresh Lean process
            self.dojo = Dojo(self.theorem, timeout=self._timeout)
            self.reset()
            self.current_state = self.initial_state
            self._tac_count = 0
            logger.info("Dojo reset successful — fresh Lean REPL started")
        except Exception as e:
            logger.error(f"Dojo reset failed: {e}. Environment is now unusable.")
            self.dojo = None
            self._tac_count = 0

    def close(self) -> None:
        """Clean up the Lean process and kill any orphans."""
        if hasattr(self, "dojo") and self.dojo is not None:
            try:
                self.dojo.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"Warning: Error during Dojo close: {e}")
            finally:
                self.dojo = None
                logger.info("Environment closed.")

        # Kill any surviving child/orphan lean/lake processes.
        try:
            from lean_reinforcement.utilities.memory import (
                kill_child_processes,
                kill_lean_orphans,
            )

            kill_child_processes()
            kill_lean_orphans()
        except Exception:
            pass

    def __del__(self) -> None:
        """Ensure cleanup when object is garbage collected."""
        try:
            self.close()
        except Exception:
            pass  # Suppress errors during garbage collection
