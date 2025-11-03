"""
Main agent loop for running MCTS-based proof search.
"""

import time
import gc
import torch
from typing import Type, Optional

from loguru import logger
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from .mcts import BaseMCTS, MCTS_GuidedRollout
from src.utilities.gym import LeanDojoEnv
from .tactic_generation import TacticGenerator
from .premise_selection import PremiseSelector


class AgentRunner:
    """
    Orchestrates the MCTS-based proof search.
    """

    def __init__(
        self,
        env: LeanDojoEnv,
        premise_selector: PremiseSelector,
        tactic_generator: TacticGenerator,
        all_premises: list[str],
        mcts_class: Type[BaseMCTS] = MCTS_GuidedRollout,
        mcts_kwargs: Optional[dict] = None,
        num_iterations: int = 100,
        max_steps: int = 100,
    ):
        """
        Initialize the agent runner.

        Args:
            env: The LeanDojo environment.
            premise_selector: The premise selector model.
            tactic_generator: The tactic generator model.
            all_premises: All accessible premises for the theorem (pre-fetched).
            mcts_class: The MCTS implementation to use (e.g., MCTS_GuidedRollout).
            mcts_kwargs: Additional keyword arguments for initializing the MCTS class.
            num_iterations: The number of MCTS iterations to run at each step.
            max_steps: The maximum number of tactics to apply before giving up.
        """
        self.env = env
        self.premise_selector = premise_selector
        self.tactic_generator = tactic_generator
        self.all_premises = all_premises
        self.mcts_class = mcts_class
        self.mcts_kwargs = mcts_kwargs if mcts_kwargs is not None else {}
        self.num_iterations = num_iterations
        self.max_steps = max_steps

    def _log_gpu_memory(self, prefix: str = ""):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(
                f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    def run(
        self,
        collect_value_data: bool = False,
        collect_policy_data: bool = False,
    ) -> tuple[bool, list[dict]]:
        """
        Run the proof search loop and collect lightweight training data.

        Args:
            collect_value_data: Whether to collect data for value head training.
            collect_policy_data: Whether to collect data for tactic generator training.

        Returns:
            A tuple containing:
            - bool: True if the proof was successful, False otherwise.
            - list[dict]: Lightweight training data collected during the run.
        """
        start_time = time.time()
        logger.info(f"Starting proof search for: {self.env.theorem.full_name}")
        self._log_gpu_memory("Initial ")

        training_data = []
        step_num = 0

        for step_num in range(1, self.max_steps + 1):
            # Check if the proof is already finished or has failed
            current_state = self.env.current_state
            if not isinstance(current_state, TacticState):
                break

            # Log GPU memory every 5 steps
            if step_num % 5 == 0:
                self._log_gpu_memory(f"Step {step_num} - ")

            # Create a new MCTS tree for the current state
            # TODO: Implement subtree reusage to improve efficiency
            mcts_instance = self.mcts_class(
                env=self.env,
                premise_selector=self.premise_selector,
                tactic_generator=self.tactic_generator,
                all_premises=self.all_premises,
                **self.mcts_kwargs,
            )

            # Run the search to find the best action
            logger.info(
                f"Step {step_num}: Running MCTS search for {self.num_iterations} iterations..."
            )
            mcts_instance.search(self.num_iterations)

            # Extract lightweight data immediately after search
            root_node = mcts_instance.root

            # Get the best child based on visit count
            best_child = None
            best_action = None
            if root_node.children:
                best_child = max(root_node.children, key=lambda c: c.visit_count)
                best_action = best_child.action
            else:
                best_action = mcts_instance.get_best_action()

            # Store lightweight training data (before discarding the tree)
            state_pp = current_state.pp

            if collect_policy_data and best_action:
                training_data.append(
                    {
                        "type": "policy",
                        "state": state_pp,
                        "premises": self.all_premises,
                        "tactic_target": best_action,
                    }
                )

            if collect_value_data:
                training_data.append(
                    {
                        "type": "value",
                        "state": state_pp,
                        "premises": self.all_premises,
                        # Value target will be filled in later with final reward
                    }
                )

            # Discard the MCTS tree immediately to free memory
            del mcts_instance
            del root_node

            # Force garbage collection and clear CUDA cache after each step
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if best_action is None:
                logger.warning("MCTS search returned no action. Stopping.")
                break

            # Take the best action in the environment
            logger.info(f"Step {step_num}: Applying best tactic: {best_action}")
            _, _, terminated, _, _ = self.env.step(best_action)

            if terminated:
                break

        # Final status check
        elapsed_time = time.time() - start_time
        success = isinstance(self.env.current_state, ProofFinished)

        if success:
            logger.success(
                f"Proof finished in {step_num} steps and {elapsed_time:.2f}s."
            )
        else:
            logger.error(
                f"Proof failed after {step_num} steps and {elapsed_time:.2f}s."
            )
            if isinstance(self.env.current_state, (LeanError, ProofGivenUp)):
                logger.warning(f"Final state: {self.env.current_state}")

        # Assign final reward to all value training data
        final_reward = 1.0 if success else -1.0
        for data_point in training_data:
            if data_point["type"] == "value":
                data_point["value_target"] = final_reward

        return success, training_data
