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
from .transformer import Transformer


class AgentRunner:
    """
    Orchestrates the MCTS-based proof search.
    """

    def __init__(
        self,
        env: LeanDojoEnv,
        transformer: Transformer,
        mcts_class: Type[BaseMCTS] = MCTS_GuidedRollout,
        mcts_kwargs: Optional[dict] = None,
        num_iterations: int = 100,
        max_steps: int = 100,
    ):
        """
        Initialize the agent runner.

        Args:
            env: The LeanDojo environment.
            transformer: The Transformer model to use.
            mcts_class: The MCTS implementation to use (e.g., MCTS_GuidedRollout).
            mcts_kwargs: Additional keyword arguments for initializing the MCTS class.
            num_iterations: The number of MCTS iterations to run at each step.
            max_steps: The maximum number of tactics to apply before giving up.
        """
        self.env = env
        self.transformer = transformer
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

    def run(self, collect_value_data: bool = False) -> tuple[bool, list[dict]]:
        """
        Run the proof search loop and collect lightweight training data.

        Args:
            collect_value_data: Whether to collect data for value head training.

        Returns:
            A tuple containing:
            - bool: True if the proof was successful, False otherwise.
            - list[dict]: Lightweight training data collected during the run
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
                transformer=self.transformer,
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

            if collect_value_data:
                training_data.append(
                    {
                        "type": "value",
                        "state": state_pp,
                        # Value target will be filled in later with final reward
                    }
                )

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

            # Clear the MCTS instance to free memory
            del mcts_instance

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
