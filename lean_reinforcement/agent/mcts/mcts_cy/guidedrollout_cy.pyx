from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, BaseMCTS
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_reinforcement.utilities.memory import (
    aggressive_cleanup,
    malloc_trim,
    get_rss_gb,
    get_available_memory_gb,
    MCTS_MIN_AVAILABLE_GB,
    MAX_WORKER_RSS_GB,
)

cdef int _MAX_ROLLOUT_STATE_CHARS = 20000
cdef int _MIN_MAX_UNIQUE_STATES = 5000

# Shorten rollouts for large states to limit per-batch memory.
cdef int _LARGE_STATE_CHARS = 5000
cdef int _LARGE_STATE_MAX_DEPTH = 15
cdef int _VERY_LARGE_STATE_CHARS = 10000
cdef int _VERY_LARGE_STATE_MAX_DEPTH = 8


cdef class MCTS_GuidedRollout(BaseMCTS):

    def __init__(
        self,
        env,
        transformer,
        config,
        float exploration_weight=1.41421356,
        int max_tree_nodes=1000,
        int batch_size=8,
        int num_tactics_to_expand=8,
        int max_rollout_depth=30,
        float max_time=300.0,
        float q_weight=1.0,
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
            q_weight=q_weight,
            **kwargs,
        )

    cpdef Node _expand(self, Node node):
        cdef object next_state
        cdef object state_key
        cdef Node child
        cdef int max_unique_states

        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp
        if len(state_str) > _MAX_ROLLOUT_STATE_CHARS:
            return node

        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )
        max_unique_states = max(self.max_tree_nodes * 2, _MIN_MAX_UNIQUE_STATES)

        for tactic, prob in tactics_with_probs:
            # Memory/timeout guard before expensive Lean call
            if self._is_timeout():
                break
            if get_available_memory_gb() < MCTS_MIN_AVAILABLE_GB:
                break
            if get_rss_gb() > (MAX_WORKER_RSS_GB * 0.9):
                break

            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Skip oversized states that bloat RSS
            if (
                isinstance(next_state, TacticState)
                and len(next_state.pp) > _MAX_ROLLOUT_STATE_CHARS
            ):
                continue

            # Check for duplicate states
            state_key = self._get_state_key(next_state)
            if state_key is not None and state_key in self.seen_states:
                child = self.seen_states[state_key]
                child.add_parent(node, tactic)
                if child not in node.children:
                    node.children.append(child)
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

        node.untried_actions = []

        # If any child reached ProofFinished, return it immediately
        for child in node.children:
            if isinstance(child.state, ProofFinished):
                return child

        # Return the best child based on PUCT score to start simulation from
        if node.children:
            return self._get_best_child(node)
        return node

    cpdef list _expand_batch(self, list nodes):
        cdef list states = []
        cdef list nodes_to_generate = []
        cdef Node node
        cdef list batch_tactics_with_probs
        cdef list tasks = []
        cdef list results = []
        cdef int i
        cdef object tactic
        cdef float prob
        cdef object next_state
        cdef Node child
        cdef object state_key
        cdef int max_unique_states

        for node in nodes:
            if isinstance(node.state, TacticState):
                state_pp = node.state.pp
                if len(state_pp) > _MAX_ROLLOUT_STATE_CHARS:
                    continue
                states.append(state_pp)
                nodes_to_generate.append(node)

        if not states:
            return nodes

        # Check timeout before expensive model call
        if self._is_timeout():
            return nodes

        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
        )

        # Check timeout after model call
        if self._is_timeout():
            return nodes

        max_unique_states = max(self.max_tree_nodes * 2, _MIN_MAX_UNIQUE_STATES)
        for i in range(len(batch_tactics_with_probs)):
            node = nodes_to_generate[i]
            for tactic, prob in batch_tactics_with_probs[i]:
                tasks.append((node, tactic, prob))

        for node, tactic, prob in tasks:
            # Check timeout before each Lean call
            if self._is_timeout():
                break
            if get_available_memory_gb() < MCTS_MIN_AVAILABLE_GB:
                break
            if get_rss_gb() > (MAX_WORKER_RSS_GB * 0.9):
                break
            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Skip oversized states that bloat RSS
            if (
                isinstance(next_state, TacticState)
                and len(next_state.pp) > _MAX_ROLLOUT_STATE_CHARS
            ):
                continue

            results.append((node, tactic, prob, next_state))

        # Create children (reusing existing nodes for duplicates - DAG structure)
        for node, tactic, prob, next_state in results:
            state_key = self._get_state_key(next_state)
            if state_key is not None and state_key in self.seen_states:
                child = self.seen_states[state_key]
                child.add_parent(node, tactic)
                if child not in node.children:
                    node.children.append(child)
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
        cdef list result_nodes = []
        for node in nodes:
            proof_child = None
            for child in node.children:
                if isinstance(child.state, ProofFinished):
                    proof_child = child
                    break
            if proof_child is not None:
                result_nodes.append(proof_child)
            elif node.children:
                result_nodes.append(self._get_best_child(node))
            else:
                result_nodes.append(node)
        return result_nodes

    cpdef float _simulate(self, Node node, object env=None):
        cdef object current_state
        cdef object sim_env
        cdef int step_idx
        cdef str state_str
        cdef str tactic
        cdef object result

        # Check timeout at start of simulation
        if self._is_timeout():
            return 0.0  # Neutral reward on timeout

        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0

        current_state = node.state
        sim_env = env if env else self.env

        # Dynamic depth: reduce rollout depth for large states to limit
        # the total number of temporary string allocations.
        cdef int effective_depth = self.max_rollout_depth
        cdef int state_len = len(current_state.pp)
        if state_len > _VERY_LARGE_STATE_CHARS:
            effective_depth = min(effective_depth, _VERY_LARGE_STATE_MAX_DEPTH)
        elif state_len > _LARGE_STATE_CHARS:
            effective_depth = min(effective_depth, _LARGE_STATE_MAX_DEPTH)

        for step_idx in range(effective_depth):
            # Check timeout at each rollout step
            if self._is_timeout():
                return 0.0  # Neutral reward on timeout

            # Abort rollout if system memory is critically low.
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
                    # Skip states whose string representation is too large.
                    if len(result.pp) > _MAX_ROLLOUT_STATE_CHARS:
                        continue
                    next_state = result

            if next_state is None:
                return -1.0  # All tactics failed

            current_state = next_state  # Continue rollout

        return 0.0

    cpdef list _simulate_batch(self, list nodes):
        """Run simulations with inter-simulation RSS tracking."""
        cdef list results = []
        cdef float rss_before_batch = get_rss_gb()
        cdef float rss_soft_cap = MAX_WORKER_RSS_GB * 0.85
        cdef Node node

        for i, node in enumerate(nodes):
            reward = self._simulate(node)
            results.append(reward)

            # Periodic RSS check inside the batch
            if (i + 1) % 2 == 0:
                rss_now = get_rss_gb()
                if rss_now - rss_before_batch > 0.2:
                    malloc_trim()
                if rss_now > rss_soft_cap:
                    aggressive_cleanup()
                    if get_rss_gb() > rss_soft_cap:
                        results.extend([0.0] * (len(nodes) - i - 1))
                        break

        return results
