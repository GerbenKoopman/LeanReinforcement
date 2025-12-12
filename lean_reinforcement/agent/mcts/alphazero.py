"""
AlphaZero MCTS implementation.
"""

import math
import torch
from typing import List, Optional
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_dojo.interaction.dojo import DojoTacticTimeoutError

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node

# Max depth for a single rollout in Part 1
MAX_ROLLOUT_DEPTH = 30
# Number of tactics to expand from the generator
NUM_TACTICS_TO_EXPAND = 8


class MCTS_AlphaZero(BaseMCTS):
    """
    Implements Part 2.
    Requires a ValueHead to be passed in.
    The _simulate method is replaced by a single call
    to the ValueHead for evaluation.
    """

    def __init__(self, value_head: ValueHead, batch_size: int = 8, *args, **kwargs):
        super().__init__(batch_size=batch_size, *args, **kwargs)
        self.value_head = value_head

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
        Phase 2: Expansion (AlphaZero-style)
        Expand the leaf node by generating all promising actions from the
        policy head, creating a child for each, and storing their prior
        probabilities. Also caches encoder features for efficiency.
        Then, return the node itself for simulation.
        """
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

        # Cache encoder features for this node if not already cached
        if node.encoder_features is None:
            node.encoder_features = self.value_head.encode_states([state_str])

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
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

            # Pre-compute encoder features for children if they're non-terminal TacticStates
            if isinstance(next_state, TacticState):
                child.encoder_features = self.value_head.encode_states([next_state.pp])

        node.untried_actions = []

        return node

    def _expand_batch(self, nodes: List[Node]) -> List[Node]:
        # 1. Generate tactics for all nodes
        states = []
        nodes_to_generate = []

        for node in nodes:
            if isinstance(node.state, TacticState):
                states.append(node.state.pp)
                nodes_to_generate.append(node)

                # Cache encoder features if missing
                if node.encoder_features is None:
                    # We will compute them in batch later or now?
                    # Let's compute them now for simplicity or add to a list
                    pass

        if not states:
            return nodes

        # Batch generate tactics
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
        new_children_nodes = []
        for node, tactic, prob, next_state in results:
            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1
            new_children_nodes.append(child)

        for node in nodes_to_generate:
            node.untried_actions = []

        return nodes

    def _simulate(self, node: Node, env: Optional[LeanDojoEnv] = None) -> float:
        """
        Phase 3: Evaluation (using Value Head)
        Uses cached encoder features if available to avoid recomputation.
        """
        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0

        # Check if we have cached encoder features
        if node.encoder_features is not None:
            # Use the cached features - much more efficient!
            value = self.value_head.predict_from_features(node.encoder_features)
        else:
            # Fall back to full encoding if features aren't cached
            state_str = node.state.pp
            value = self.value_head.predict(state_str)

        return value

    def _simulate_batch(self, nodes: List[Node]) -> List[float]:
        """
        Phase 3: Batch Evaluation
        """
        # Separate nodes by terminal status and feature availability
        results = [0.0] * len(nodes)

        features_list = []
        indices_with_features = []

        states_to_encode = []
        indices_to_encode = []

        for i, node in enumerate(nodes):
            if node.is_terminal:
                if isinstance(node.state, ProofFinished):
                    results[i] = 1.0
                elif isinstance(node.state, (LeanError, ProofGivenUp)):
                    results[i] = -1.0
                continue

            if not isinstance(node.state, TacticState):
                results[i] = -1.0
                continue

            if node.encoder_features is not None:
                features_list.append(node.encoder_features)
                indices_with_features.append(i)
            else:
                states_to_encode.append(node.state.pp)
                indices_to_encode.append(i)

        # Predict from features
        if features_list:
            # Stack features: (batch, feature_dim)
            batch_features = torch.cat(features_list, dim=0)
            values = self.value_head.predict_from_features_batch(batch_features)
            for idx, val in zip(indices_with_features, values):
                results[idx] = val

        # Predict from states (encode + predict)
        if states_to_encode:
            values = self.value_head.predict_batch(states_to_encode)
            for idx, val in zip(indices_to_encode, values):
                results[idx] = val

        return results
