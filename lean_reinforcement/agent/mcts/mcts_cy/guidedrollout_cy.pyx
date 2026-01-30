from libc.math cimport sqrt
from lean_reinforcement.agent.mcts.mcts_cy.base_mcts_cy cimport Node, BaseMCTS
import math
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

cdef class MCTS_GuidedRollout(BaseMCTS):

    def __init__(
        self,
        env,
        transformer,
        config,
        float exploration_weight=1.41421356,
        int max_tree_nodes=5000,
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



    cpdef Node _expand(self, Node node):
        cdef object next_state
        cdef Node child

        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        state_str = node.state.pp

        tactics_with_probs = self.transformer.generate_tactics_with_probs(
            state_str, n=self.num_tactics_to_expand
        )

        for tactic, prob in tactics_with_probs:
            next_state = self.env.run_tactic_stateless(node.state, tactic)
            child = self._create_child_node(node, tactic, next_state, prob)

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
            child = self._create_child_node(node, tactic, next_state, prob)

        for node in nodes_to_generate:
            node.untried_actions = []

        return [self._get_best_child(node) if node.children else node for node in nodes]

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
