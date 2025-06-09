# filepath: /home/gerben-koopman/projects/lean_related/lean_reinforcement/src/lean_rl/environment.py
"""
LeanDojo Environment for Reinforcement Learning

This module provides a standardized RL environment interface for theorem proving in Lean
using LeanDojo. It wraps the LeanDojo API to provide a gym-like interface for RL agents.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

from lean_dojo import (
    LeanGitRepo,
    Dojo,
    TacticState,
    ProofFinished,
    ProofGivenUp,
    LeanError,
    Theorem,
)

# Result mapping for RL interface - using LeanDojo types directly
RESULT_MAPPING = {
    ProofFinished: "proof_finished",
    ProofGivenUp: "proof_given_up",
    LeanError: "error",
    TacticState: "success",
}


@dataclass
class StepResult:
    """Result of taking a step in the environment."""

    state: Optional[TacticState]
    reward: float
    done: bool
    info: Dict[str, Any]
    action_result: str  # String representation from RESULT_MAPPING


class LeanEnvironment:
    """
    Gym-like environment for theorem proving with LeanDojo.

    This environment provides a standardized interface for RL agents to interact
    with Lean theorem proving through LeanDojo.
    """

    def __init__(
        self,
        repo: LeanGitRepo,
        timeout: int = 600,
        max_steps: int = 100,
        reward_scheme: str = "sparse",
        additional_imports: Optional[List[str]] = None,
    ):
        """
        Initialize the Lean environment.

        Args:
            repo: LeanGitRepo instance for the repository to work with
            timeout: Maximum time in seconds for each tactic execution
            max_steps: Maximum number of steps allowed per episode
            reward_scheme: Reward scheme ("sparse", "dense", or "shaped")
            additional_imports: Additional Lean imports to include
        """
        self.repo = repo
        self.timeout = timeout
        self.max_steps = max_steps
        self.reward_scheme = reward_scheme
        self.additional_imports = additional_imports or []

        # Episode tracking
        self.current_theorem: Optional[Theorem] = None
        self.dojo: Optional[Dojo] = None
        self.current_state: Optional[TacticState] = None  # Use TacticState directly
        self.step_count: int = 0
        self.episode_history: List[Tuple[str, str]] = []  # (tactic, result)

        # Reward tracking
        self.initial_num_goals: int = 0
        self.prev_num_goals: int = 0

    def reset(self, theorem: Theorem) -> TacticState:
        """
        Reset the environment with a new theorem.

        Args:
            theorem: The theorem to prove

        Returns:
            Initial state of the theorem
        """
        self.current_theorem = theorem
        self.step_count = 0
        self.episode_history = []

        # Clean up previous dojo if it exists
        if self.dojo is not None:
            try:
                self.dojo.__exit__(None, None, None)
            except:
                pass

        # Initialize new dojo
        self.dojo = Dojo(
            theorem, timeout=self.timeout, additional_imports=self.additional_imports
        )

        # Get initial state
        dojo, initial_state = self.dojo.__enter__()
        self.dojo = dojo

        # initial_state should be a TacticState when using tactics
        if not isinstance(initial_state, TacticState):
            raise ValueError(f"Expected TacticState, got {type(initial_state)}")

        self.current_state = initial_state  # No conversion needed

        # Track initial goals for reward calculation
        self.initial_num_goals = self.current_state.num_goals
        self.prev_num_goals = self.current_state.num_goals

        return self.current_state

    def step(self, action: str) -> StepResult:
        """
        Execute a tactic action in the environment.

        Args:
            action: The tactic string to execute

        Returns:
            StepResult containing the new state, reward, done flag, and info
        """
        if self.dojo is None or self.current_state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self.step_count += 1
        info = {
            "step": self.step_count,
            "action": action,
            "previous_goals": self.current_state.num_goals,
        }

        try:
            # Execute the tactic using current state directly (no conversion)
            result = self.dojo.run_tac(self.current_state, action)

            # Process the result using LeanDojo types directly
            action_result, reward, done, new_state = self._process_result(result)

            # Update info with result-specific data
            info.update(self._build_info_dict(result, action_result))

            # Check if max steps reached
            if self.step_count >= self.max_steps and not done:
                done = True
                info["max_steps_reached"] = True
                if not isinstance(result, ProofFinished):
                    reward += -1.0  # Penalty for not finishing within step limit

            # Record action in history
            self.episode_history.append((action, action_result))

            return StepResult(
                state=new_state,
                reward=reward,
                done=done,
                info=info,
                action_result=action_result,
            )

        except Exception as e:
            # Handle unexpected errors
            info.update({"unexpected_error": True, "error_message": str(e)})

            return StepResult(
                state=self.current_state,
                reward=-1.0,  # Strong penalty for unexpected errors
                done=True,
                info=info,
                action_result="error",
            )

    def _process_result(self, result) -> Tuple[str, float, bool, Optional[TacticState]]:
        """
        Process LeanDojo result into RL components.

        Args:
            result: Result from LeanDojo's run_tac

        Returns:
            Tuple of (action_result, reward, done, new_state)
        """
        # Get action result string from mapping
        action_result = RESULT_MAPPING.get(type(result), "unknown")

        if isinstance(result, ProofFinished):
            reward = self._calculate_reward("proof_finished", 0)
            done = True
            new_state = None

        elif isinstance(result, ProofGivenUp):
            reward = self._calculate_reward(
                "proof_given_up",
                self.current_state.num_goals if self.current_state else 0,
            )
            done = True
            new_state = None

        elif isinstance(result, LeanError):
            reward = self._calculate_reward(
                "error", self.current_state.num_goals if self.current_state else 0
            )
            done = False
            new_state = self.current_state  # State unchanged on error

        elif isinstance(result, TacticState):
            reward = self._calculate_reward("success", result.num_goals)
            done = False
            new_state = result
            self.current_state = new_state
            self.prev_num_goals = new_state.num_goals

        else:
            raise RuntimeError(f"Unexpected result type: {type(result)}")

        return action_result, reward, done, new_state

    def _build_info_dict(self, result, action_result: str) -> Dict[str, Any]:
        """Build info dictionary based on result type."""
        info: Dict[str, Any] = {"action_result": action_result}

        if isinstance(result, ProofFinished):
            info["proof_finished"] = True
            info["success"] = True
            info["steps_taken"] = self.step_count
        elif isinstance(result, ProofGivenUp):
            info["proof_given_up"] = True
            info["success"] = False
        elif isinstance(result, LeanError):
            info["error"] = True
            info["error_message"] = result.error
        elif isinstance(result, TacticState):
            info["new_goals"] = result.num_goals
            info["goals_changed"] = result.num_goals != self.prev_num_goals

        return info

    def _calculate_reward(self, action_result: str, new_num_goals: int) -> float:
        """
        Calculate reward based on the action result and goal changes.

        Args:
            action_result: The result of the action
            new_num_goals: Number of goals after the action

        Returns:
            Reward value
        """
        if self.reward_scheme == "sparse":
            return self._sparse_reward(action_result)
        elif self.reward_scheme == "dense":
            return self._dense_reward(action_result, new_num_goals)
        elif self.reward_scheme == "shaped":
            return self._shaped_reward(action_result, new_num_goals)
        else:
            raise ValueError(f"Unknown reward scheme: {self.reward_scheme}")

    def _sparse_reward(self, action_result: str) -> float:
        """Sparse reward: only reward proof completion."""
        return 1.0 if action_result == "proof_finished" else 0.0

    def _dense_reward(self, action_result: str, new_num_goals: int) -> float:
        """Dense reward: reward goal reduction and penalize errors."""
        if action_result == "proof_finished":
            return 10.0
        elif action_result == "error":
            return -0.1
        elif action_result == "proof_given_up":
            return -1.0
        else:
            # Reward for reducing goals
            goal_reduction = self.prev_num_goals - new_num_goals
            return goal_reduction * 0.1

    def _shaped_reward(self, action_result: str, new_num_goals: int) -> float:
        """Shaped reward: combination of sparse and dense with additional shaping."""
        base_reward = self._dense_reward(action_result, new_num_goals)

        if action_result == "success":
            # Additional shaping based on progress
            progress = (self.initial_num_goals - new_num_goals) / max(
                self.initial_num_goals, 1
            )
            base_reward += progress * 0.1

            # Bonus for maintaining low number of goals
            if new_num_goals <= 2:
                base_reward += 0.05

        return base_reward

    def close(self):
        """Clean up the environment."""
        if self.dojo is not None:
            try:
                self.dojo.__exit__(None, None, None)
            except:
                pass
            self.dojo = None

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary of the current episode."""
        return {
            "theorem": str(self.current_theorem) if self.current_theorem else None,
            "steps_taken": self.step_count,
            "max_steps": self.max_steps,
            "history": self.episode_history.copy(),
            "initial_goals": self.initial_num_goals,
            "final_goals": self.current_state.num_goals if self.current_state else None,
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
