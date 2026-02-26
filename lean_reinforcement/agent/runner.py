"""
Main agent loop for running MCTS-based proof search.
"""

import time
import gc
import torch
from typing import Type, Optional
from collections import deque
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.agent.mcts import BaseMCTS, MCTS_GuidedRollout
from lean_reinforcement.agent.mcts.base_mcts import Node
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol
from lean_reinforcement.utilities.config import TrainingConfig


class AgentRunner:
    """
    Orchestrates the MCTS-based proof search.
    """

    def __init__(
        self,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
        config: TrainingConfig,
        mcts_class: Type[BaseMCTS] = MCTS_GuidedRollout,
        mcts_kwargs: Optional[dict] = None,
        num_iterations: int = 100,
        max_steps: int = 100,
        proof_timeout: float = 1200.0,
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
            proof_timeout: Maximum time in seconds for entire proof search.
        """
        self.env = env
        self.transformer = transformer
        self.config = config
        self.mcts_class = mcts_class
        self.num_iterations = num_iterations
        self.max_steps = max_steps
        self.proof_timeout = proof_timeout

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

        Dispatches to full-search mode (backtracking via tree exploration) or
        step-by-step mode based on ``config.full_search``.

        Args:
            collect_value_data: Whether to collect data for value head training.
            use_final_reward: Whether to use the final reward for training.
            use_wandb: Whether to log metrics to wandb.

        Returns:
            A tuple containing:
            - dict: Metrics about the run (success, steps, time).
            - list[dict]: Lightweight training data collected during the run
        """
        full_search = getattr(self.config, "full_search", True)
        if full_search:
            return self._run_full_search(
                collect_value_data=collect_value_data,
                use_final_reward=use_final_reward,
                use_wandb=use_wandb,
            )
        return self._run_step_by_step(
            collect_value_data=collect_value_data,
            use_final_reward=use_final_reward,
            use_wandb=use_wandb,
        )

    # ------------------------------------------------------------------
    # Full-search mode: run MCTS from root with full budget, no commitment
    # ------------------------------------------------------------------

    def _run_full_search(
        self,
        collect_value_data: bool = False,
        use_final_reward: bool = True,
        use_wandb: bool = True,
    ) -> tuple[dict, list[dict]]:
        """
        Run MCTS from the initial state with the full iteration and time
        budget.  The tree grows freely across all branches; the UCB/PUCT
        selection mechanism steers exploration away from unpromising paths
        (implicit backtracking).  Only after the search finishes is the
        proof path extracted and applied to the environment.
        """
        start_time = time.time()
        logger.info(
            f"Starting full-search proof search for: {self.env.theorem.full_name}"
        )
        self._log_gpu_memory("Initial ")

        training_data: list[dict] = []
        mcts_instance = None

        if not isinstance(self.env.current_state, TacticState):
            return self._finalise(
                start_time, 0, training_data, collect_value_data, use_final_reward
            )

        # Create a single MCTS tree from the initial state
        mcts_instance = self.mcts_class(
            env=self.env,
            transformer=self.transformer,
            config=self.config,
            **self.mcts_kwargs,
        )

        # Full budget: num_iterations * max_steps iterations,
        # bounded by proof_timeout in wall-clock time.
        total_iterations = self.num_iterations * self.max_steps
        max_time = self.proof_timeout - (time.time() - start_time)
        if max_time < 30:
            logger.warning("Not enough time for full search. Skipping.")
            return self._finalise(
                start_time, 0, training_data, collect_value_data, use_final_reward
            )

        logger.info(
            f"Running full MCTS search: {total_iterations} iterations, "
            f"max {max_time:.0f}s"
        )

        try:
            mcts_instance.search(total_iterations, max_time=max_time)
        except Exception as e:
            logger.error(f"MCTS search failed with error: {e}")
            return self._finalise(
                start_time, 0, training_data, collect_value_data, use_final_reward
            )

        # Collect training data from the tree
        if collect_value_data:
            training_data = self._collect_tree_data(mcts_instance.root)

        # Extract and apply proof path if found
        proof_steps = 0
        if mcts_instance.root.max_value == 1.0:
            proof_path = mcts_instance.extract_proof_path()
            if proof_path:
                logger.info(f"Found complete proof with {len(proof_path)} tactics")
                for tactic in proof_path:
                    logger.info(f"  Applying proof tactic: {tactic}")
                    try:
                        _, _, terminated = self.env.step(tactic)
                    except Exception as e:
                        logger.error(f"Proof path application failed: {e}")
                        break
                    proof_steps += 1
                    if terminated:
                        break

        del mcts_instance
        return self._finalise(
            start_time,
            proof_steps,
            training_data,
            collect_value_data,
            use_final_reward,
        )

    def _collect_tree_data(self, root: Node) -> list[dict]:
        """
        Traverse the MCTS tree and collect (state, value) training data
        from all expanded TacticState nodes.
        """
        data: list[dict] = []
        visited: set[int] = set()
        queue: deque[tuple[Node, int]] = deque([(root, 0)])

        while queue:
            node, depth = queue.popleft()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)

            # Only collect from expanded TacticState nodes
            if (
                isinstance(node.state, TacticState)
                and node.visit_count > 0
                and node.children
            ):
                total_visits = sum(c.visit_count for c in node.children)
                visit_distribution: dict[str, float] = {}
                if total_visits > 0:
                    visit_distribution = {
                        c.action: c.visit_count / total_visits
                        for c in node.children
                        if c.action is not None
                    }

                data.append(
                    {
                        "type": "value",
                        "state": node.state.pp,
                        "step": depth,
                        "mcts_value": node.value(),
                        "visit_count": node.visit_count,
                        "visit_distribution": visit_distribution,
                    }
                )

            for child in node.children:
                if id(child) not in visited:
                    queue.append((child, depth + 1))

        return data

    # ------------------------------------------------------------------
    # Step-by-step mode: commit one action per step (original behaviour)
    # ------------------------------------------------------------------

    def _run_step_by_step(
        self,
        collect_value_data: bool = False,
        use_final_reward: bool = True,
        use_wandb: bool = True,
    ) -> tuple[dict, list[dict]]:
        """
        Original step-by-step proof search: run MCTS, commit to best
        action, advance tree root, repeat.  No backtracking.
        """
        start_time = time.time()
        logger.info(f"Starting proof search for: {self.env.theorem.full_name}")
        self._log_gpu_memory("Initial ")

        # Timeout for entire proof search
        proof_timeout = self.proof_timeout

        training_data: list[dict] = []
        step_num = 0
        mcts_instance = None

        for step_num in range(1, self.max_steps + 1):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Check proof timeout before starting new MCTS search
            elapsed = time.time() - start_time
            remaining_time = proof_timeout - elapsed
            if remaining_time <= 0:
                logger.warning(
                    f"Proof search exceeded {proof_timeout}s timeout after {elapsed:.1f}s. Stopping."
                )
                break

            # Also enforce minimum remaining time (30s)
            if remaining_time < 30:
                logger.warning(
                    f"Only {remaining_time:.1f}s remaining (< 30s minimum). "
                    "Stopping to avoid partial search."
                )
                break

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
                        config=self.config,
                        **self.mcts_kwargs,
                    )

                # Run the search to find the best action
                step_max_time = min(mcts_instance.max_time, remaining_time)
                logger.info(
                    f"Step {step_num}: Running MCTS search for "
                    f"{self.num_iterations} iterations (max {step_max_time:.0f}s)..."
                )

                try:
                    mcts_instance.search(self.num_iterations, max_time=step_max_time)
                except Exception as e:
                    logger.error(f"MCTS search failed with error: {e}")
                    break

                # Extract lightweight data immediately after search
                root_node = mcts_instance.root

                # If a complete proof was found, extract and apply full path
                if root_node.max_value == 1.0:
                    proof_path = mcts_instance.extract_proof_path()
                    if proof_path:
                        logger.info(
                            f"Step {step_num}: Found complete proof with "
                            f"{len(proof_path)} tactics"
                        )
                        # Collect training data for current state before applying
                        if collect_value_data:
                            state_pp = current_state.pp
                            training_data.append(
                                {
                                    "type": "value",
                                    "state": state_pp,
                                    "step": step_num,
                                    "mcts_value": 1.0,
                                    "visit_count": root_node.visit_count,
                                    "visit_distribution": {},
                                }
                            )
                        del root_node
                        # Apply all tactics in the proof path
                        for tactic in proof_path:
                            logger.info(f"  Applying proof tactic: {tactic}")
                            try:
                                _, _, terminated = self.env.step(tactic)
                            except Exception as e:
                                logger.error(f"Proof path application failed: {e}")
                                break
                            if terminated:
                                break
                        break  # Exit step loop - proof applied

                # Get the best child based on visit count (or max_value)
                best_child = None
                best_action = None
                if root_node.children:
                    if root_node.max_value == 1.0:
                        # A proof was found - follow the proof path
                        best_child = max(root_node.children, key=lambda c: c.max_value)
                    else:
                        best_child = max(
                            root_node.children, key=lambda c: c.visit_count
                        )
                    best_action = best_child.action

                # Store lightweight training data (before discarding the tree)
                state_pp = current_state.pp

                if collect_value_data:
                    mcts_value = root_node.value() if root_node.visit_count > 0 else 0.0
                    visit_count = root_node.visit_count

                    visit_distribution: dict[str, float] = {}
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
                            "visit_distribution": visit_distribution,
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

                # Move the MCTS root to the child corresponding to the action
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

        return self._finalise(
            start_time, step_num, training_data, collect_value_data, use_final_reward
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _finalise(
        self,
        start_time: float,
        step_num: int,
        training_data: list[dict],
        collect_value_data: bool,
        use_final_reward: bool,
    ) -> tuple[dict, list[dict]]:
        """Build final metrics, assign value targets, and return."""
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

        for data_point in training_data:
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
