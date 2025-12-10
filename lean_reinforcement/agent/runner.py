"""
Main agent loop for running MCTS-based proof search.
"""

import time
import gc
import torch
from typing import Type, Optional
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS
from lean_reinforcement.agent.mcts.guidedrollout import MCTS_GuidedRollout
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol


class AgentRunner:
    """
    Orchestrates the MCTS-based proof search.
    """

    def __init__(
        self,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
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
        self.num_iterations = num_iterations
        self.max_steps = max_steps

        self.mcts_kwargs = mcts_kwargs if mcts_kwargs is not None else {}

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
        use_final_reward: bool = True,
        use_wandb: bool = True,
    ) -> tuple[dict, list[dict]]:
        """
        Run the proof search loop and collect lightweight training data.

        Args:
            collect_value_data: Whether to collect data for value head training.
            use_final_reward: Whether to use the final reward for training (True) or the MCTS value estimates (False).
                            Default: True.
            use_wandb: Whether to log metrics to wandb.

        Returns:
            A tuple containing:
            - dict: Metrics about the run (success, steps, time).
            - list[dict]: Lightweight training data collected during the run
        """
        start_time = time.time()
        logger.info(f"Starting proof search for: {self.env.theorem.full_name}")
        self._log_gpu_memory("Initial ")

        training_data = []
        step_num = 0
        mcts_instance = None

        for step_num in range(1, self.max_steps + 1):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                # Check if the proof is already finished or has failed
                current_state = self.env.current_state
                if not isinstance(current_state, TacticState):
                    break

                # Log GPU memory every 5 steps
                if step_num % 5 == 0:
                    self._log_gpu_memory(f"Step {step_num} - ")

                # Create a new MCTS tree for the current state if needed
                if mcts_instance is None:
                    mcts_instance = self.mcts_class(
                        env=self.env,
                        transformer=self.transformer,
                        **self.mcts_kwargs,
                    )

                # Run the search to find the best action
                logger.info(
                    f"Step {step_num}: Running MCTS search for {self.num_iterations} iterations..."
                )

                try:
                    mcts_instance.search(self.num_iterations)
                except Exception as e:
                    logger.error(f"MCTS search failed with error: {e}")
                    break

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
                    # Extract MCTS statistics from the root node
                    mcts_value = root_node.value() if root_node.visit_count > 0 else 0.0
                    visit_count = root_node.visit_count

                    # Calculate visit-count-based policy target (for future policy head training)
                    # This represents the improved policy from MCTS search
                    visit_distribution = {}
                    if root_node.children:
                        total_visits = sum(
                            child.visit_count for child in root_node.children
                        )
                        if total_visits > 0:
                            visit_distribution = {
                                child.action: child.visit_count / total_visits
                                for child in root_node.children
                                if child.action is not None
                            }

                    training_data.append(
                        {
                            "type": "value",
                            "state": state_pp,
                            "step": step_num,
                            "mcts_value": mcts_value,
                            "visit_count": visit_count,
                            "visit_distribution": visit_distribution,  # For future policy training
                            # Value target will be filled in later with final reward
                        }
                    )

                del root_node

                if best_action is None:
                    logger.warning("MCTS search returned no action. Stopping.")
                    break

                # Take the best action in the environment
                logger.info(f"Step {step_num}: Applying best tactic: {best_action}")

                try:
                    _, _, terminated = self.env.step(best_action)
                except Exception as e:
                    logger.error(f"Environment step failed with error: {e}")
                    break

                if isinstance(self.env.current_state, (LeanError, ProofGivenUp)):
                    logger.warning(
                        f"Tactic resulted in error: {self.env.current_state}"
                    )
                    break

                if terminated:
                    break

                # Move the MCTS root to the child corresponding to the chosen action
                mcts_instance.move_root(best_action)

            except Exception as e:
                logger.error(f"Error in agent loop: {e}")
                if mcts_instance:
                    del mcts_instance
                    mcts_instance = None
                break

        # Clean up MCTS instance after loop
        if mcts_instance is not None:
            del mcts_instance

        # Final status check
        elapsed_time = time.time() - start_time
        success = isinstance(self.env.current_state, ProofFinished)

        metrics = {
            "proof_search/success": success,
            "proof_search/steps": step_num,
            "proof_search/time": elapsed_time,
        }

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

        # Assign value targets
        final_reward = 1.0 if success else -1.0

        for i, data_point in enumerate(training_data):
            if data_point["type"] == "value":
                if use_final_reward:
                    data_point["value_target"] = final_reward
                elif "mcts_value" in data_point:
                    data_point["value_target"] = data_point["mcts_value"]
                else:
                    data_point["value_target"] = final_reward  # Fallback

                # Store the raw final reward for analysis
                data_point["final_reward"] = final_reward

        return metrics, training_data
