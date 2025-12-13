"""
Guided Rollout MCTS implementation.
"""

import math
from typing import List, Optional
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_dojo.interaction.dojo import DojoTacticTimeoutError

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node

# Max depth for a single rollout in Part 1
MAX_ROLLOUT_DEPTH = 30
# Number of tactics to expand from the generator
NUM_TACTICS_TO_EXPAND = 8


class MCTS_GuidedRollout(BaseMCTS):
    """
    Implements Part 1.
    The _simulate method performs a full "guided rollout"
    using the TacticGenerator greedily until the proof is
    finished or max depth is reached.
    """

    def __init__(self, batch_size: int = 8, *args, **kwargs):
        super().__init__(batch_size=batch_size, *args, **kwargs)

    def _puct_score(self, node: Node) -> float:
        """Calculates the PUCT score for a node."""
        if node.parent is None:
            return 0.0  # Should not happen for children

        # Virtual loss
        v_loss = self._get_virtual_loss(node)
        visit_count = node.visit_count + v_loss

        # Q(s,a): Exploitation term
        # Use max_value instead of mean value for max-backup
        if visit_count == 0:
            q_value = 0.0
        else:
            q_value = node.max_value - (v_loss / visit_count)

        # U(s,a): Exploration term
        exploration = (
            self.exploration_weight
            * node.prior_p
            * (math.sqrt(node.parent.visit_count) / (1 + visit_count))
        )

        return q_value + exploration

    def _get_best_child(self, node: Node) -> Node:
        """Selects the best child based on the PUCT score."""
        return max(node.children, key=self._puct_score)

    def _expand(self, node: Node) -> Node:
        """
        Phase 2: Expansion
        Expand the leaf node by generating all promising actions from the
        policy head, creating a child for each, and storing their prior
        probabilities.
        """
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

        # Use generate_tactics_with_probs to get priors
        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=NUM_TACTICS_TO_EXPAND
        )

        # Create a child for each promising tactic
        for tactic, prob in tactics_with_probs:
            try:
                next_state = self.env.dojo.run_tac(node.state, tactic)
            except DojoTacticTimeoutError:
                logger.warning(f"Tactic timed out: {tactic[:100]}")
                # Treat timeout as an error state and continue with other tactics
                next_state = LeanError(error="Tactic execution timed out")
            except Exception as e:
                logger.warning(f"Error running tactic '{tactic[:100]}': {e}")
                next_state = LeanError(error=f"Exception: {str(e)}")

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob  # Store the Prior
            node.children.append(child)
            self.node_count += 1

        node.untried_actions = []

        # Return the best child based on PUCT score to start simulation from
        return self._get_best_child(node)

    def _expand_batch(self, nodes: List[Node]) -> List[Node]:
        # 1. Generate tactics for all nodes
        states = []
        nodes_to_generate = []

        for node in nodes:
            if isinstance(node.state, TacticState):
                states.append(node.state.pp)
                nodes_to_generate.append(node)

        if not states:
            return nodes

        # Batch generate tactics with probabilities
        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=NUM_TACTICS_TO_EXPAND
        )

        # 2. Prepare tasks for parallel execution
        tasks = []
        for i, tactics_probs in enumerate(batch_tactics_with_probs):
            node = nodes_to_generate[i]
            for tactic, prob in tactics_probs:
                tasks.append((node, tactic, prob))

        # 3. Run tactics sequentially
        results = []
        for node, tactic, prob in tasks:
            try:
                next_state = self.env.dojo.run_tac(node.state, tactic)
            except Exception as e:
                next_state = LeanError(error=str(e))
            results.append((node, tactic, prob, next_state))

        # 4. Create children
        for node, tactic, prob, next_state in results:
            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

        for node in nodes_to_generate:
            node.untried_actions = []

        # Return the best child for each node to start simulation
        return [self._get_best_child(node) if node.children else node for node in nodes]

    def _simulate(self, node: Node, env: Optional[LeanDojoEnv] = None) -> float:
        """
        Phase 3: Simulation (Guided Rollout)
        """
        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0  # Should not happen if checks are correct

        current_state: TacticState = node.state

        # Use provided env or fallback to self.env
        sim_env = env if env else self.env

        for step_idx in range(MAX_ROLLOUT_DEPTH):
            state_str = current_state.pp

            # Get a single greedy tactic
            tactic = self.transformer.generate_tactics(state_str, n=1)[0]

            # Run the tactic with timeout handling
            try:
                result = sim_env.dojo.run_tac(current_state, tactic)
            except DojoTacticTimeoutError:
                logger.warning(f"Tactic timed out during simulation: {tactic[:100]}")
                return -1.0  # Penalize timeouts
            except Exception as e:
                logger.warning(
                    f"Error during simulation with tactic '{tactic[:100]}': {e}"
                )
                return -1.0

            # Check result
            if isinstance(result, ProofFinished):
                # Reward shorter proofs: 1.0 - 0.01 per step
                return 1.0 - 0.01 * (step_idx + 1)
            if isinstance(result, (LeanError, ProofGivenUp)):
                return -1.0  # Penalize errors

            if not isinstance(result, TacticState):
                return -1.0  # Should not happen

            current_state = result  # Continue rollout

        return 0.0  # Reached max depth, count as a draw/timeout

    def _simulate_batch(self, nodes: List[Node]) -> List[float]:
        return [self._simulate(node) for node in nodes]
