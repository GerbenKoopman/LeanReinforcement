from libc.math cimport sqrt
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, BaseMCTS
from lean_reinforcement.agent.mcts.base_mcts import MAX_TACTIC_LENGTH
from collections import Counter
import math
import torch
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

cdef int REPETITION_THRESHOLD = 5

cdef bint _has_excessive_repetition_cy(str tactic):
    """Inlined Cython version of _has_excessive_repetition for performance."""
    cdef list parts
    cdef int threshold = REPETITION_THRESHOLD
    
    parts = tactic.replace("[", "").replace("]", "").split(",")
    if len(parts) < threshold:
        return False
    
    counts = Counter(p.strip() for p in parts if p.strip())
    if not counts:
        return False
    
    return counts.most_common(1)[0][1] >= threshold

cdef class MCTS_AlphaZero(BaseMCTS):
    cdef public object value_head

    def __init__(
        self,
        value_head,
        env,
        transformer,
        float exploration_weight=1.41421356,
        int max_tree_nodes=10000,
        int batch_size=8,
        int num_tactics_to_expand=8,
        int max_rollout_depth=30,
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
        )
        self.value_head = value_head

    cpdef float _puct_score(self, Node node):
        cdef float q_value
        cdef float exploration
        cdef int v_loss
        cdef int visit_count
        cdef Node parent = node.parent

        if parent is None:
            return 0.0

        v_loss = self._get_virtual_loss(node)
        visit_count = node.visit_count + v_loss

        if visit_count == 0:
            q_value = 0.0
        else:
            q_value = node.max_value - (v_loss / <float>visit_count)

        exploration = (
            self.exploration_weight
            * node.prior_p
            * (sqrt(parent.visit_count) / (1 + visit_count))
        )

        return q_value + exploration

    cpdef Node _get_best_child(self, Node node):
        cdef Node child
        cdef Node best_child = None
        cdef float max_score = -1e9
        cdef float score

        if not node.children:
            raise ValueError("Node has no children")

        for child in node.children:
            score = self._puct_score(child)
            if best_child is None or score > max_score:
                max_score = score
                best_child = child
        
        return best_child

    cpdef Node _expand(self, Node node):
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

        if node.encoder_features is None:
            node.encoder_features = self.value_head.encode_states([state_str])

        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )

        children_to_encode = []
        states_to_encode = []

        for tactic, prob in tactics_with_probs:
            # Filter 1: Skip excessively long tactics (likely truncated/malformed)
            if len(tactic) > MAX_TACTIC_LENGTH:
                continue

            # Filter 2: Skip tactics with excessive repetition
            if _has_excessive_repetition_cy(tactic):
                continue

            next_state = self.env.run_tactic_stateless(node.state, tactic)

            # Filter 3: Skip no-op tactics (state unchanged)
            if isinstance(next_state, TacticState):
                if next_state.pp == state_str:
                    continue

                # Filter 4: Skip if we've already seen this state
                if next_state.pp in self.seen_states:
                    continue

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states and collect for batch encoding
            if isinstance(next_state, TacticState):
                self.seen_states[next_state.pp] = child
                children_to_encode.append(child)
                states_to_encode.append(next_state.pp)

        # Batch encode all children's states at once for efficiency
        if children_to_encode:
            batch_features = self.value_head.encode_states(states_to_encode)
            for i, child in enumerate(children_to_encode):
                child.encoder_features = batch_features[i : i + 1]

        node.untried_actions = []
        return node

    cpdef list _expand_batch(self, list nodes):
        cdef list states = []
        cdef list nodes_to_generate = []
        cdef Node node
        cdef list batch_tactics_with_probs
        cdef list tasks = []
        cdef list results = []
        cdef list new_children_nodes = []
        cdef int i
        cdef object tactic
        cdef float prob
        cdef object next_state
        cdef Node child

        for node in nodes:
            if isinstance(node.state, TacticState):
                states.append(node.state.pp)
                nodes_to_generate.append(node)

        if not states:
            return nodes

        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
        )

        for i in range(len(batch_tactics_with_probs)):
            tactics_probs = batch_tactics_with_probs[i]
            node = nodes_to_generate[i]
            for tactic, prob in tactics_probs:
                # Filter 1: Skip excessively long tactics
                if len(tactic) > MAX_TACTIC_LENGTH:
                    continue

                # Filter 2: Skip tactics with excessive repetition
                if _has_excessive_repetition_cy(tactic):
                    continue

                tasks.append((node, tactic, prob))

        for node, tactic, prob in tasks:
            try:
                next_state = self.env.run_tactic_stateless(node.state, tactic)
            except Exception as e:
                next_state = LeanError(error=str(e))
            results.append((node, tactic, prob, next_state))

        # Collect children with TacticState for batch encoding
        cdef list children_to_encode = []
        cdef list states_to_encode = []

        for node, tactic, prob, next_state in results:
            # Filter 3: Skip no-op tactics (state unchanged)
            if isinstance(next_state, TacticState):
                if next_state.pp == node.state.pp:
                    continue

                # Filter 4: Skip if we've already seen this state
                if next_state.pp in self.seen_states:
                    continue

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1
            new_children_nodes.append(child)

            # Register new state in seen_states and collect for batch encoding
            if isinstance(next_state, TacticState):
                self.seen_states[next_state.pp] = child
                children_to_encode.append(child)
                states_to_encode.append(next_state.pp)

        # Batch encode all children's states at once for efficiency
        if children_to_encode:
            batch_features = self.value_head.encode_states(states_to_encode)
            for i, child in enumerate(children_to_encode):
                child.encoder_features = batch_features[i : i + 1]

        for node in nodes_to_generate:
            node.untried_actions = []

        return nodes

    cpdef float _simulate(self, Node node):
        cdef float value
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
