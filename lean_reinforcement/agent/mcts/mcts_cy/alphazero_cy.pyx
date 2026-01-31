from libc.math cimport sqrt
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, BaseMCTS
import math
import torch
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

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
        max_time: float = 300.0,
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
        )
        self.value_head = value_head
        self.config = config



    cpdef Node _expand(self, Node node):
        cdef object next_state
        cdef Node child

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
            child = self._create_child_node(node, tactic, next_state, prob)
            
            # If a new node was created, encode its features
            if self.config.use_caching and child.visit_count == 0 and isinstance(child.state, TacticState):
                child.encoder_features = self.value_head.encode_states([child.state.pp])

        node.untried_actions = []
        
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
        cdef list new_children_nodes = []
        cdef list new_children_states = []
        cdef int i
        cdef object tactic
        cdef float prob
        cdef object next_state
        cdef Node child
        cdef object batch_features

        for node in nodes:
            if isinstance(node.state, TacticState):
                # If caching is on and we don't have features, encode them now.
                if node.encoder_features is None and self.config.use_caching:
                    # This is suboptimal (batching would be better), but preserves original logic.
                    node.encoder_features = self.value_head.encode_states([node.state.pp])
                states.append(node.state.pp)
                nodes_to_generate.append(node)

        if not states:
            return nodes

        if self._is_timeout():
            return nodes

        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
        )

        if self._is_timeout():
            return nodes

        for i in range(len(batch_tactics_with_probs)):
            node = nodes_to_generate[i]
            for tactic, prob in batch_tactics_with_probs[i]:
                tasks.append((node, tactic, prob))

        for node, tactic, prob in tasks:
            if self._is_timeout():
                break
            try:
                next_state = self.env.run_tactic_stateless(node.state, tactic)
            except Exception as e:
                next_state = LeanError(error=str(e))
            results.append((node, tactic, prob, next_state))

        for node, tactic, prob, next_state in results:
            child = self._create_child_node(node, tactic, next_state, prob)
            # If a new node was created, add it to the list for feature encoding
            if child.visit_count == 0 and isinstance(child.state, TacticState):
                 if self.config.use_caching and child not in new_children_nodes:
                    new_children_nodes.append(child)
                    new_children_states.append(child.state.pp)

        # Batch encode features for all new nodes
        if self.config.use_caching and new_children_nodes:
            if self._is_timeout():
                 # Fallback if timeout occurs before encoding
                 pass
            else:
                batch_features = self.value_head.encode_states(new_children_states)
                for i in range(len(new_children_nodes)):
                    new_children_nodes[i].encoder_features = batch_features[i].unsqueeze(0)

        for node in nodes_to_generate:
            node.untried_actions = []

        return [self._get_best_child(node) if node.children else node for node in nodes]

    cpdef float _simulate(self, Node node):
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
