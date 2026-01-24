from libc.math cimport sqrt
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, Edge, BaseMCTS
import math
import torch
from lean_reinforcement.agent.value_head import ValueHead
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol

cdef class MCTS_AlphaZero(BaseMCTS):
    cdef public object value_head

    def __init__(
        self,
        value_head: ValueHead,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
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
            exploration_weight=exploration_weight,
            max_tree_nodes=max_tree_nodes,
            batch_size=batch_size,
            num_tactics_to_expand=num_tactics_to_expand,
            max_rollout_depth=max_rollout_depth,
            max_time=max_time,
        )
        self.value_head = value_head

    cpdef float _puct_score(self, Node parent, Edge edge):
        cdef float q_value
        cdef float exploration
        cdef int v_loss
        cdef int visit_count
        cdef Node child = edge.child

        v_loss = self._get_virtual_loss(child)
        visit_count = edge.visit_count + v_loss

        if visit_count == 0:
            q_value = 0.0
        else:
            q_value = child.max_value - (v_loss / <float>visit_count)

        exploration = (
            self.exploration_weight
            * edge.prior
            * (sqrt(parent.visit_count) / (1 + visit_count))
        )

        return q_value + exploration

    cpdef Edge _get_best_edge(self, Node node):
        cdef Edge edge
        cdef Edge best_edge = None
        cdef float max_score = -1e9
        cdef float score

        if not node.children:
            raise ValueError("Node has no children")

        for edge in node.children:
            score = self._puct_score(node, edge)
            if best_edge is None or score > max_score:
                max_score = score
                best_edge = edge
        
        return best_edge

    cpdef Node _expand(self, Node node):
        cdef object state_key
        cdef object next_state
        cdef Node child
        cdef Node existing_node

        if not isinstance(node.state, TacticState):
            return node, None

        state_str = node.state.pp

        if node.encoder_features is None:
            node.encoder_features = self.value_head.encode_states([state_str])

        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )

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
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1
            
            # Register new state in seen_states and encode features
            if isinstance(next_state, TacticState):
                self.seen_states[state_key] = child
                child.encoder_features = self.value_head.encode_states([next_state.pp])

        node.untried_actions = []
        return node, None

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
        cdef Node existing_node
        cdef object state_key

        for node in nodes:
            if isinstance(node.state, TacticState):
                states.append(node.state.pp)
                nodes_to_generate.append(node)

        if not states:
            return [(node, None) for node in nodes]

        # Check timeout before expensive model call
        if self._is_timeout():
            return nodes

        batch_tactics_with_probs = self.transformer.generate_tactics_with_probs_batch(
            states, n=self.num_tactics_to_expand
        )

        # Check timeout after model call
        if self._is_timeout():
            return nodes

        for i in range(len(batch_tactics_with_probs)):
            tactics_probs = batch_tactics_with_probs[i]
            node = nodes_to_generate[i]
            for tactic, prob in tactics_probs:
                tasks.append((node, tactic, prob))

        for node, tactic, prob in tasks:
            # Check timeout before each Lean call
            if self._is_timeout():
                break
            try:
                next_state = self.env.run_tactic_stateless(node.state, tactic)
            except Exception as e:
                next_state = LeanError(error=str(e))
            
            if isinstance(next_state, (LeanError, ProofGivenUp)):
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

            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

            # Register new state in seen_states
            if isinstance(next_state, TacticState):
                self.seen_states[state_key] = child
                new_children_nodes.append(child)

        for node in nodes_to_generate:
            node.untried_actions = []

        return [(node, None) for node in nodes]

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
