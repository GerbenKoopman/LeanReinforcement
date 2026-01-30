"""
Guided Rollout MCTS implementation.
"""

import math
from typing import List, Optional

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.mcts.base_mcts import BaseMCTS, Node
from lean_reinforcement.agent.transformer import TransformerProtocol
from lean_reinforcement.utilities.config import TrainingConfig


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
        config: TrainingConfig,
        exploration_weight: float = math.sqrt(2),
        max_tree_nodes: int = 5000,
        batch_size: int = 8,
        num_tactics_to_expand: int = 8,
        max_rollout_depth: int = 30,
        max_time: float = 300.0,  # Max time per MCTS search step (seconds)
        **kwargs,
    ):
        super().__init__(
            env=env,
            transformer=transformer,
            config=config,
            exploration_weight=exploration_weight,
            max_tree_nodes=max_tree_nodes,
            batch_size=batch_size,
            num_tactics_to_expand=num_tactics_to_expand,
            max_rollout_depth=max_rollout_depth,
            max_time=max_time,
        )

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
        probabilities. Duplicate states are reused (DAG structure) to enable
        multi-path backpropagation.
        """
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

        # Use generate_tactics_with_probs to get priors
        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )

        # Create a child for each promising tactic (reusing existing nodes for duplicates)
        for tactic, prob in tactics_with_probs:
            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Check for duplicate states
            state_key = self._get_state_key(next_state)
            if state_key is not None and state_key in self.seen_states:
                # Reuse existing node - add as child with additional parent edge
                existing_node = self.seen_states[state_key]
                existing_node.add_parent(node, tactic)
                if existing_node not in node.children:
                    node.children.append(existing_node)
                continue

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob  # Store the Prior
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states
            if state_key is not None:
                self.seen_states[state_key] = child

        node.untried_actions = []

        # Return the best child based on PUCT score to start simulation from
        if node.children:
            return self._get_best_child(node)
        else:
            # All tactics were filtered out; return the node itself
            return node

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

        # Early timeout check before expensive model call
        if self._is_timeout():
            return nodes

        # Batch generate tactics with probabilities
        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
        )

        # Early timeout check after model call
        if self._is_timeout():
            return nodes

        # Prepare tasks
        tasks = []
        for i, tactics_probs in enumerate(batch_tactics_with_probs):
            node = nodes_to_generate[i]
            for tactic, prob in tactics_probs:
                tasks.append((node, tactic, prob))

        # Run tactics sequentially with timeout checks
        results = []
        for node, tactic, prob in tasks:
            # Check timeout periodically during Lean calls
            if self._is_timeout():
                break
            next_state = self.env.run_tactic_stateless(node.state, tactic)
            results.append((node, tactic, prob, next_state))

        # Create children (reusing existing nodes for duplicates - DAG structure)
        for node, tactic, prob, next_state in results:
            # Check for duplicate states
            state_key = self._get_state_key(next_state)
            if state_key is not None and state_key in self.seen_states:
                # Reuse existing node - add as child with additional parent edge
                existing_node = self.seen_states[state_key]
                existing_node.add_parent(node, tactic)
                if existing_node not in node.children:
                    node.children.append(existing_node)
                continue

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states
            if state_key is not None:
                self.seen_states[state_key] = child

        for node in nodes_to_generate:
            node.untried_actions = []

        # Return the best child for each node to start simulation
        return [self._get_best_child(node) if node.children else node for node in nodes]

    def _simulate(self, node: Node, env: Optional[LeanDojoEnv] = None) -> float:
        """
        Phase 3: Simulation (Guided Rollout)
        """
        # Early timeout check
        if self._is_timeout():
            return 0.0  # Neutral reward on timeout

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
            # Check timeout at each rollout step
            if self._is_timeout():
                return 0.0  # Neutral reward on timeout

            state_str = current_state.pp

            # Get a single greedy tactic
            tactic = self.transformer.generate_tactics(state_str, n=1)[0]

            # Check timeout after model call
            if self._is_timeout():
                return 0.0

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
