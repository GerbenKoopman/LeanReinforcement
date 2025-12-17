"""
Guided Rollout MCTS implementation.
"""

import math
from typing import List, Optional

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node, Edge
from lean_reinforcement.agent.transformer import TransformerProtocol


class MCTS_GuidedRollout(BaseMCTS):
    """
    Implements Part 1.
    The _simulate method performs a full "guided rollout"
    using the TacticGenerator greedily until the proof is
    finished or max depth is reached.
    """

    def __init__(
        self,
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

        node.untried_actions = []

        if node.children:
            best_edge = self._get_best_edge(node)
            return best_edge.child, best_edge
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
            return [(node, None) for node in nodes]

        # Batch generate tactics with probabilities
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
            next_state = self.env.run_tactic_stateless(node.state, tactic)

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

        # Return the best child for each node to start simulation
        final_results: List[tuple[Node, Optional[Edge]]] = []
        for node in nodes:
            if node.children:
                best_edge = self._get_best_edge(node)
                final_results.append((best_edge.child, best_edge))
            else:
                final_results.append((node, None))
        return final_results

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

        for step_idx in range(self.max_rollout_depth):
            state_str = current_state.pp

            # Get a single greedy tactic
            tactic = self.transformer.generate_tactics(state_str, n=1)[0]

            # Run the tactic with timeout handling
            result = sim_env.run_tactic_stateless(current_state, tactic)

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
