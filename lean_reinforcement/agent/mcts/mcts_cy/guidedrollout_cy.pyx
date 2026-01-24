from libc.math cimport sqrt
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, Edge, BaseMCTS
import math
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol

cdef class MCTS_GuidedRollout(BaseMCTS):

    def __init__(
        self,
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

    cpdef tuple _expand(self, Node node):
        cdef object state_str
        cdef list tactics_with_probs
        cdef object tactic
        cdef float prob
        cdef object next_state
        cdef Node child_node
        cdef Edge edge

    cpdef Node _expand(self, Node node):
        cdef object state_key
        cdef object next_state
        cdef Node child
        cdef Node existing_node

        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

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

            # Register new state in seen_states
            if state_key is not None:
                self.seen_states[state_key] = child

        node.untried_actions = []

        if node.children:
            return (node, self._get_best_edge(node))
        return (node, None)

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

        return [(node, self._get_best_edge(node) if node.children else None) for node in nodes]

    cpdef float _simulate(self, Node node):
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
        sim_env = self.env

        for step_idx in range(self.max_rollout_depth):
            # Check timeout at each rollout step
            if self._is_timeout():
                return 0.0  # Neutral reward on timeout
                
            state_str = current_state.pp
            tactic = self.transformer.generate_tactics(state_str, n=1)[0]
            
            # Check timeout after model call
            if self._is_timeout():
                return 0.0
                
            result = sim_env.run_tactic_stateless(current_state, tactic)

            if isinstance(result, ProofFinished):
                return 1.0 - 0.01 * (step_idx + 1)
            if isinstance(result, (LeanError, ProofGivenUp)):
                return -1.0
            if not isinstance(result, TacticState):
                return -1.0
            
            current_state = result

        return 0.0

    cpdef list _simulate_batch(self, list nodes):
        return [self._simulate(node) for node in nodes]
