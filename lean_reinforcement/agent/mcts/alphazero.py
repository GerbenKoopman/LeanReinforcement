"""
AlphaZero MCTS implementation.
"""

import math
import torch
from typing import List, Optional

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node, Edge
from lean_reinforcement.agent.transformer import TransformerProtocol


class MCTS_AlphaZero(BaseMCTS):
    """
    Implements Part 2.
    Requires a ValueHead to be passed in.
    The _simulate method is replaced by a single call
    to the ValueHead for evaluation.
    """

    def __init__(
        self,
        value_head: ValueHead,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
        exploration_weight: float = math.sqrt(2),
        max_tree_nodes: int = 10000,
        batch_size: int = 8,
        num_tactics_to_expand: int = 8,
        max_rollout_depth: int = 30,
        max_time: float = 600.0,
        **kwargs,
    ):
        super().__init__(
            env=env,
            transformer=transformer,
            exploration_weight=exploration_weight,
            max_tree_nodes=max_tree_nodes,
            batch_size=batch_size,
            num_tactics_to_expand=num_tactics_to_expand,
            max_rollout_depth=max_rollout_depth,
            max_time=max_time,
        )
        self.value_head = value_head

    def _puct_score(self, parent: Node, edge: Edge) -> float:
        """Calculates the PUCT score for an edge."""
        child = edge.child

        # Virtual loss
        v_loss = self._get_virtual_loss(child)
        visit_count = edge.visit_count + v_loss

        # Q(s,a): Exploitation term
        # Use max_value instead of mean value for max-backup
        if visit_count == 0:
            q_value = 0.0
        else:
            q_value = child.max_value - (v_loss / visit_count)

        # U(s,a): Exploration term
        exploration = (
            self.exploration_weight
            * edge.prior
            * (math.sqrt(parent.visit_count) / (1 + visit_count))
        )

        return q_value + exploration

    def _get_best_edge(self, node: Node) -> Edge:
        """Selects the best edge based on the PUCT score."""
        return max(node.children, key=lambda edge: self._puct_score(node, edge))

    def _expand(self, node: Node) -> tuple[Node, Optional[Edge]]:
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
            state_str, n=self.num_tactics_to_expand
        )

        # Create a child for each promising tactic
        for tactic, prob in tactics_with_probs:
            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Prune Error States
            if isinstance(next_state, (LeanError, ProofGivenUp)):
                continue

            # Check Transposition Table
            child_node = self._get_or_create_node(next_state)

            edge = Edge(action=tactic, prior=prob, child=child_node)
            node.children.append(edge)

            # Pre-compute encoder features for children if they're non-terminal TacticStates
            if (
                isinstance(next_state, TacticState)
                and child_node.encoder_features is None
            ):
                child_node.encoder_features = self.value_head.encode_states(
                    [next_state.pp]
                )

        node.untried_actions = []

        return node, None

    def _expand_batch(self, nodes: List[Node]) -> List[tuple[Node, Optional[Edge]]]:
        # 1. Generate tactics for all nodes
        states = []
        nodes_to_generate = []

        for node in nodes:
            if isinstance(node.state, TacticState):
                states.append(node.state.pp)
                nodes_to_generate.append(node)

        if not states:
            empty_results: List[tuple[Node, Optional[Edge]]] = [
                (node, None) for node in nodes
            ]
            return empty_results

        # Batch generate tactics
        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
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

            # Prune Error States
            if isinstance(next_state, (LeanError, ProofGivenUp)):
                continue

            results.append((node, tactic, prob, next_state))

        # 4. Create children
        for node, tactic, prob, next_state in results:
            child_node = self._get_or_create_node(next_state)
            edge = Edge(action=tactic, prior=prob, child=child_node)
            node.children.append(edge)

        for node in nodes_to_generate:
            node.untried_actions = []

        final_results: List[tuple[Node, Optional[Edge]]] = [
            (node, None) for node in nodes
        ]
        return final_results

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
