import math
import torch
from typing import List, Optional, Dict
from loguru import logger
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

cdef class Node:

    def __init__(self, state, Node parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_p = 0.0
        self.children = []
        self.visit_count = 0
        self.max_value = float("-inf")
        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions = None
        self.encoder_features = None

    cpdef float value(self):
        if self.visit_count == 0:
            return 0.0
        return self.max_value

    cpdef bint is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

cdef class BaseMCTS:

    def __init__(
        self,
        env,
        transformer,
        float exploration_weight=1.41421356,
        int max_tree_nodes=10000,
        int batch_size=8,
        int num_tactics_to_expand=8,
        int max_rollout_depth=30,
    ):
        self.env = env
        self.transformer = transformer
        self.exploration_weight = exploration_weight
        self.max_tree_nodes = max_tree_nodes
        self.batch_size = batch_size
        self.num_tactics_to_expand = num_tactics_to_expand
        self.max_rollout_depth = max_rollout_depth
        self.node_count = 0
        self.virtual_losses = {}
        self.seen_states = {}

        self.theorem = env.theorem
        self.theorem_pos = env.theorem_pos

        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = Node(state=env.current_state)
        self.node_count = 1

    cpdef int _get_virtual_loss(self, Node node):
        return self.virtual_losses.get(node, 0)

    cpdef void _add_virtual_loss(self, Node node, int loss=1):
        self.virtual_losses[node] = self.virtual_losses.get(node, 0) + loss

    cpdef void _remove_virtual_loss(self, Node node, int loss=1):
        if node in self.virtual_losses:
            self.virtual_losses[node] -= loss
            if self.virtual_losses[node] <= 0:
                del self.virtual_losses[node]

    def _log_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(
                f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    def search(self, int num_iterations, batch_size=None):
        cdef int iteration
        cdef int current_batch_size
        cdef list leaves
        cdef Node leaf
        cdef list expanded_nodes
        cdef list rewards
        cdef int i
        cdef float reward
        cdef Node child

        if batch_size is None:
            batch_size = self.batch_size
        
        cdef int b_size = batch_size

        with torch.no_grad():
            for iteration in range(0, num_iterations, b_size):
                if self.node_count >= self.max_tree_nodes:
                    break

                current_batch_size = min(b_size, num_iterations - iteration)
                leaves = []

                for _ in range(current_batch_size):
                    leaf = self._select(self.root)

                    if leaf.is_terminal:
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(leaf, 1.0)
                        elif isinstance(leaf.state, (LeanError, ProofGivenUp)):
                            self._backpropagate(leaf, -1.0)
                        continue

                    if not isinstance(leaf.state, TacticState):
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(leaf, 1.0)
                        else:
                            self._backpropagate(leaf, -1.0)
                        continue

                    self._add_virtual_loss(leaf)
                    leaves.append(leaf)

                if not leaves:
                    continue

                expanded_nodes = self._expand_batch(leaves)
                rewards = self._simulate_batch(expanded_nodes)

                for i in range(len(leaves)):
                    leaf = leaves[i]
                    self._remove_virtual_loss(leaf)
                    child = expanded_nodes[i]
                    reward = rewards[i]
                    self._backpropagate(child, reward)

                if torch.cuda.is_available() and iteration % 20 == 0 and iteration > 0:
                    torch.cuda.empty_cache()

    cpdef Node _select(self, Node node):
        cdef Node current = node
        while not current.is_terminal and current.is_fully_expanded():
            if not current.children:
                return current
            current = self._get_best_child(current)
        return current

    cpdef Node _get_best_child(self, Node node):
        raise NotImplementedError

    cpdef Node _expand(self, Node node):
        raise NotImplementedError

    cpdef list _expand_batch(self, list nodes):
        return [self._expand(node) for node in nodes]

    cpdef float _simulate(self, Node node):
        raise NotImplementedError

    cpdef list _simulate_batch(self, list nodes):
        return [self._simulate(node) for node in nodes]

    cpdef void _backpropagate(self, Node node, float reward):
        cdef Node current = node
        while current is not None:
            current.visit_count += 1
            if reward > current.max_value:
                current.max_value = reward
            current = current.parent

    def get_best_action(self):
        cdef Node best_child
        if not self.root.children:
            if self.root.untried_actions is None and isinstance(
                self.root.state, TacticState
            ):
                state_str = self.root.state.pp
                self.root.untried_actions = self.transformer.generate_tactics(
                    state_str, n=self.num_tactics_to_expand
                )

            if self.root.untried_actions:
                return self.root.untried_actions[0]
            return None

        best_child = max(self.root.children, key=lambda c: c.visit_count)
        return best_child.action

    def move_root(self, str action):
        cdef Node found_child = None
        cdef Node child
        for child in self.root.children:
            if child.action == action:
                found_child = child
                break

        if found_child:
            self.root = found_child
            self.root.parent = None
            self.node_count = self._count_nodes(self.root)
        else:
            if not isinstance(
                self.env.current_state,
                (TacticState, ProofFinished, LeanError, ProofGivenUp),
            ):
                raise TypeError(
                    f"Invalid state type for new root: {type(self.env.current_state)}"
                )

            self.root = Node(state=self.env.current_state)
            self.node_count = 1

    cpdef int _count_nodes(self, Node node):
        cdef int count = 1
        cdef Node child
        for child in node.children:
            count += self._count_nodes(child)
        return count
