from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, BaseMCTS
import torch
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_reinforcement.utilities.memory import (
    get_rss_gb,
    get_available_memory_gb,
    MCTS_MIN_AVAILABLE_GB,
    MAX_WORKER_RSS_GB,
)


cdef class MCTS_AlphaZero(BaseMCTS):
    cdef public object value_head

    def __init__(
        self,
        value_head,
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
        self.value_head = value_head
        self.config = config

    cpdef Node _expand(self, Node node):
        cdef object next_state
        cdef object state_key
        cdef Node child
        cdef list children_to_encode = []
        cdef list states_to_encode = []

        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

        if node.encoder_features is None and self.config.use_caching:
            node.encoder_features = self.value_head.encode_states([state_str])

        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )

        for tactic, prob in tactics_with_probs:
            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Check for duplicate states
            state_key = self._get_state_key(next_state)
            if state_key is not None and state_key in self.seen_states:
                child = self.seen_states[state_key]
                child.add_parent(node, tactic)
                if child not in node.children:
                    node.children.append(child)
                continue

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states and collect for batch encoding
            if isinstance(next_state, TacticState):
                if state_key is not None:
                    self.seen_states[state_key] = child
                children_to_encode.append(child)
                states_to_encode.append(next_state.pp)

        # Batch encode all children's states at once for efficiency
        if children_to_encode and self.config.use_caching:
            batch_features = self.value_head.encode_states(states_to_encode)
            for i, child in enumerate(children_to_encode):
                child.encoder_features = batch_features[i : i + 1]

        node.untried_actions = []

        # If any child reached ProofFinished, return it immediately
        for child in node.children:
            if isinstance(child.state, ProofFinished):
                return child

        return node

    cpdef list _expand_batch(self, list nodes):
        cdef list states = []
        cdef list nodes_to_generate = []
        cdef list nodes_needing_features = []
        cdef list states_for_features = []
        cdef Node node
        cdef list batch_tactics_with_probs
        cdef list tasks = []
        cdef list results = []
        cdef list children_to_encode = []
        cdef list states_to_encode = []
        cdef int i
        cdef object tactic
        cdef float prob
        cdef object next_state
        cdef Node child
        cdef object batch_features
        cdef object state_key

        for node in nodes:
            if isinstance(node.state, TacticState):
                states.append(node.state.pp)
                nodes_to_generate.append(node)

                # Collect nodes needing encoder features for batch encoding
                if node.encoder_features is None:
                    nodes_needing_features.append(node)
                    states_for_features.append(node.state.pp)

        # Early timeout check before expensive operations
        if self._is_timeout():
            return nodes

        # Batch encode parent nodes' features if any are missing
        if nodes_needing_features and self.config.use_caching:
            batch_features = self.value_head.encode_states(states_for_features)
            for i, node in enumerate(nodes_needing_features):
                node.encoder_features = batch_features[i : i + 1]

        if not states:
            return nodes

        # Early timeout check after encoding
        if self._is_timeout():
            return nodes

        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
        )

        # Early timeout check after model call
        if self._is_timeout():
            return nodes

        for i in range(len(batch_tactics_with_probs)):
            node = nodes_to_generate[i]
            for tactic, prob in batch_tactics_with_probs[i]:
                tasks.append((node, tactic, prob))

        assert self.env.dojo is not None, "Dojo not initialized"
        for node, tactic, prob in tasks:
            if self._is_timeout():
                break
            if get_available_memory_gb() < MCTS_MIN_AVAILABLE_GB:
                break
            if get_rss_gb() > (MAX_WORKER_RSS_GB * 0.9):
                break
            try:
                next_state = self.env.dojo.run_tac(node.state, tactic)
            except Exception as e:
                next_state = LeanError(error=str(e))
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

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states and collect for batch encoding
            if isinstance(next_state, TacticState):
                if state_key is not None:
                    self.seen_states[state_key] = child
                children_to_encode.append(child)
                states_to_encode.append(next_state.pp)

        # Batch encode all children's states at once for efficiency
        if children_to_encode and self.config.use_caching:
            batch_features = self.value_head.encode_states(states_to_encode)
            for i, child in enumerate(children_to_encode):
                child.encoder_features = batch_features[i : i + 1]

        for node in nodes_to_generate:
            node.untried_actions = []

        # Return ProofFinished child for each node if present; otherwise return the node
        cdef list result_nodes = []
        for node in nodes:
            proof_child = None
            for child in node.children:
                if isinstance(child.state, ProofFinished):
                    proof_child = child
                    break
            if proof_child is not None:
                result_nodes.append(proof_child)
            else:
                result_nodes.append(node)
        return result_nodes

    cpdef float _simulate(self, Node node, object env=None):
        cdef float value

        # Check timeout before evaluation
        if self._is_timeout():
            return 0.0  # Neutral reward on timeout

        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0

        if node.encoder_features is not None:
            value = self.value_head.predict_from_features(node.encoder_features)
        else:
            state_str = node.state.pp
            value = self.value_head.predict(state_str)

        return value

    cpdef list _simulate_batch(self, list nodes):
        cdef list results = [0.0] * len(nodes)
        cdef list features_list = []
        cdef list indices_with_features = []
        cdef list states_to_encode = []
        cdef list indices_to_encode = []
        cdef int i
        cdef Node node
        cdef object batch_features
        cdef list values

        # Check timeout before batch evaluation
        if self._is_timeout():
            return results  # Return neutral rewards on timeout

        for i in range(len(nodes)):
            node = nodes[i]
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

        if features_list:
            batch_features = torch.cat(features_list, dim=0)
            values = self.value_head.predict_from_features_batch(batch_features)
            for idx, val in zip(indices_with_features, values):
                results[idx] = val

        if states_to_encode:
            values = self.value_head.predict_batch(states_to_encode)
            for idx, val in zip(indices_to_encode, values):
                results[idx] = val

        return results
