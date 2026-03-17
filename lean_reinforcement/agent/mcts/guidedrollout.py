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
from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    malloc_trim,
    get_rss_gb,
    get_available_memory_gb,
    MAX_WORKER_RSS_GB,
    MCTS_MIN_AVAILABLE_GB,
)

# Cap on proof-state string length to prevent degenerate states from
# bloating worker RSS. States beyond this are treated as dead ends.
_MAX_ROLLOUT_STATE_CHARS = 20_000
_MIN_MAX_UNIQUE_STATES = 5_000

# Shorten rollouts for large states to limit per-batch memory.
_LARGE_STATE_CHARS = 5_000
_LARGE_STATE_MAX_DEPTH = 15
_VERY_LARGE_STATE_CHARS = 10_000
_VERY_LARGE_STATE_MAX_DEPTH = 8


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
        max_tree_nodes: int = 1000,
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
            **kwargs,
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
        if len(state_str) > _MAX_ROLLOUT_STATE_CHARS:
            return node

        # Use generate_tactics_with_probs to get priors
        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )
        max_unique_states = max(self.max_tree_nodes * 2, _MIN_MAX_UNIQUE_STATES)

        # Create a child for each promising tactic (reusing existing nodes for duplicates)
        for tactic, prob in tactics_with_probs:
            # Memory/timeout guard before expensive Lean call
            if self._is_timeout():
                break
            if get_available_memory_gb() < MCTS_MIN_AVAILABLE_GB:
                break
            if get_rss_gb() > (MAX_WORKER_RSS_GB * 0.9):
                break

            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Skip child states whose pretty-printed representation is
            # excessively large — these bloat the worker RSS when glibc
            # keeps the freed pages, and they are unlikely to lead to
            # useful proofs.
            if (
                isinstance(next_state, TacticState)
                and len(next_state.pp) > _MAX_ROLLOUT_STATE_CHARS
            ):
                continue

            # Check for duplicate states
            state_key = self._get_state_key(next_state)
            if state_key is not None and state_key in self.seen_states:
                # Reuse existing node - add as child with additional parent edge
                existing_node = self.seen_states[state_key]
                existing_node.add_parent(node, tactic)
                if existing_node not in node.children:
                    node.children.append(existing_node)
                continue

            if state_key is not None and len(self.seen_states) >= max_unique_states:
                continue

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob  # Store the Prior
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states
            if state_key is not None:
                self.seen_states[state_key] = child

        node.untried_actions = []

        # If any child reached ProofFinished, return it immediately
        for child in node.children:
            if isinstance(child.state, ProofFinished):
                return child

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
                state_pp = node.state.pp
                if len(state_pp) > _MAX_ROLLOUT_STATE_CHARS:
                    continue
                states.append(state_pp)
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
        max_unique_states = max(self.max_tree_nodes * 2, _MIN_MAX_UNIQUE_STATES)
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
            if get_available_memory_gb() < MCTS_MIN_AVAILABLE_GB:
                break
            if get_rss_gb() > (MAX_WORKER_RSS_GB * 0.9):
                break
            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Skip child states whose pretty-printed representation is
            # excessively large — these bloat worker RSS via glibc heap
            # fragmentation and are unlikely to lead to useful proofs.
            if (
                isinstance(next_state, TacticState)
                and len(next_state.pp) > _MAX_ROLLOUT_STATE_CHARS
            ):
                continue

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

            if state_key is not None and len(self.seen_states) >= max_unique_states:
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

        # Return best child for each node, preferring ProofFinished children
        result = []
        for node in nodes:
            proof_child = next(
                (c for c in node.children if isinstance(c.state, ProofFinished)),
                None,
            )
            if proof_child:
                result.append(proof_child)
            elif node.children:
                result.append(self._get_best_child(node))
            else:
                result.append(node)
        return result

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

        # Dynamic depth: reduce rollout depth for large states to limit
        # the total number of temporary string allocations (which cause
        # glibc heap fragmentation and RSS growth).
        state_len = len(current_state.pp)
        effective_depth = self.max_rollout_depth
        if state_len > _VERY_LARGE_STATE_CHARS:
            effective_depth = min(effective_depth, _VERY_LARGE_STATE_MAX_DEPTH)
        elif state_len > _LARGE_STATE_CHARS:
            effective_depth = min(effective_depth, _LARGE_STATE_MAX_DEPTH)

        for step_idx in range(effective_depth):
            # Check timeout at each rollout step
            if self._is_timeout():
                return 0.0  # Neutral reward on timeout

            # Abort rollout if system memory is critically low.
            # Guided rollouts create many Lean Dojo states (up to
            # 3 tactics × 30 steps = 90 per rollout) that persist
            # in the Lean subprocess and are never freed.  Checking
            # system-wide available memory catches the collective
            # pressure from all workers.
            if get_available_memory_gb() < MCTS_MIN_AVAILABLE_GB:
                return 0.0

            if get_rss_gb() > (MAX_WORKER_RSS_GB * 0.9):
                return 0.0

            state_str = current_state.pp
            if len(state_str) > _MAX_ROLLOUT_STATE_CHARS:
                return 0.0

            # Generate multiple tactics for robustness (try first successful one)
            tactics = self.transformer.generate_tactics(state_str, n=3)
            # Deduplicate while preserving order
            tactics = list(dict.fromkeys(tactics))

            # Check timeout after model call
            if self._is_timeout():
                return 0.0

            # Try each tactic, use the first successful one
            next_state = None
            for tactic in tactics:
                if self._is_timeout():
                    return 0.0
                result = sim_env.run_tactic_stateless(current_state, tactic)

                if isinstance(result, ProofFinished):
                    # Reward shorter proofs: 1.0 - 0.01 per step
                    return 1.0 - 0.01 * (step_idx + 1)

                if isinstance(result, TacticState) and next_state is None:
                    # Skip states whose string representation is too
                    # large — they would bloat RSS via glibc heap
                    # fragmentation and are rarely useful.
                    if len(result.pp) > _MAX_ROLLOUT_STATE_CHARS:
                        continue
                    next_state = result

            if next_state is None:
                return -1.0  # All tactics failed

            current_state = next_state  # Continue rollout

        return 0.0  # Reached max depth, count as a draw/timeout

    def _simulate_batch(self, nodes: List[Node]) -> List[float]:
        """Run simulations with inter-simulation RSS tracking.

        After every 2 simulations we check the worker's RSS.  If it has
        grown significantly, we force a ``malloc_trim`` to reclaim freed
        pages.  If RSS exceeds the soft cap even after cleanup, the
        remaining simulations are skipped with a neutral reward (0.0) to
        prevent the worker from OOM-killing the system.
        """
        results: List[float] = []
        rss_before_batch = get_rss_gb()
        rss_soft_cap = MAX_WORKER_RSS_GB * 0.85

        for i, node in enumerate(nodes):
            reward = self._simulate(node)
            results.append(reward)

            # Periodic RSS check inside the batch
            if (i + 1) % 2 == 0:
                rss_now = get_rss_gb()
                # If RSS grew noticeably during this batch, reclaim pages
                if rss_now - rss_before_batch > 0.2:
                    malloc_trim()
                # Abort remaining simulations if RSS is dangerously high
                if rss_now > rss_soft_cap:
                    aggressive_cleanup()
                    if get_rss_gb() > rss_soft_cap:
                        results.extend([0.0] * (len(nodes) - i - 1))
                        break

        return results
