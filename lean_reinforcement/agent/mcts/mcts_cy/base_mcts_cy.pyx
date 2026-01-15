import math
import time
import torch
from typing import List, Optional, Dict
from loguru import logger
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

cdef class Node:

    def __init__(self, state, Node parent=None, action=None):
        self.state = state
        # Support multiple parents for DAG structure
        # Each entry is (parent_node, action_that_led_here)
        self.parents = []
        if parent is not None:
            self.parents.append((parent, action))
        self.action = action  # Keep for compatibility (first action that created this node)
        self.prior_p = 0.0
        self.children = []
        self.visit_count = 0
        self.max_value = float("-inf")
        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions = None
        self.encoder_features = None

    cpdef void add_parent(self, Node parent, object action=None):
        """Add an additional parent to this node (for DAG structure)."""
        # Avoid duplicate parent-action pairs
        cdef tuple pair
        for pair in self.parents:
            if pair[0] is parent and pair[1] == action:
                return
        self.parents.append((parent, action))

    cpdef Node get_parent(self):
        """Backward compatibility: returns first parent or None."""
        if self.parents:
            return <Node>self.parents[0][0]
        return None

    @property
    def parent(self):
        """Backward compatibility property."""
        return self.get_parent()

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
        float max_time=600.0,
    ):
        self.env = env
        self.transformer = transformer
        self.exploration_weight = exploration_weight
        self.max_tree_nodes = max_tree_nodes
        self.batch_size = batch_size
        self.num_tactics_to_expand = num_tactics_to_expand
        self.max_rollout_depth = max_rollout_depth
        self.max_time = max_time
        self.node_count = 0
        self.virtual_losses = {}
        # Seen states dictionary for deduplication (maps state string to Node)
        self.seen_states = {}

        self.theorem = env.theorem
        self._search_deadline = 0.0  # Will be set in search()
        self.theorem_pos = env.theorem_pos

        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = Node(state=env.current_state)
        self.node_count = 1

        # Register root state in seen_states
        if isinstance(env.current_state, TacticState):
            self.seen_states[env.current_state.pp] = self.root

    cpdef int _get_virtual_loss(self, Node node):
        return self.virtual_losses.get(node, 0)

    cpdef void _add_virtual_loss(self, Node node, int loss=1):
        self.virtual_losses[node] = self.virtual_losses.get(node, 0) + loss

    cpdef void _remove_virtual_loss(self, Node node, int loss=1):
        if node in self.virtual_losses:
            self.virtual_losses[node] -= loss
            if self.virtual_losses[node] <= 0:
                del self.virtual_losses[node]

    cpdef bint _is_timeout(self):
        """Check if the search has exceeded its deadline."""
        if self._search_deadline <= 0:
            return False
        return time.time() > self._search_deadline

    cpdef object _get_state_key(self, object state):
        """Get a hashable key for a state for deduplication purposes."""
        if isinstance(state, TacticState):
            return state.pp
        return None

    def _log_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(
                f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    def search(self, int num_iterations, batch_size=None, max_time=None):
        cdef int iteration
        cdef int current_batch_size
        cdef list leaves
        cdef Node leaf
        cdef list expanded_nodes
        cdef list rewards
        cdef int i
        cdef float reward
        cdef Node child
        cdef double start_time
        cdef double effective_max_time

        if batch_size is None:
            batch_size = self.batch_size
        if max_time is None:
            effective_max_time = self.max_time
        else:
            effective_max_time = max_time
        
        cdef int b_size = batch_size
        start_time = time.time()
        self._search_deadline = start_time + effective_max_time

        with torch.no_grad():
            for iteration in range(0, num_iterations, b_size):
                # Check time limit
                if self._is_timeout():
                    break

                if self.node_count >= self.max_tree_nodes:
                    break

                current_batch_size = min(b_size, num_iterations - iteration)
                leaves = []

                for _ in range(current_batch_size):
                    if self._is_timeout():
                        break
                    leaf = self._select(self.root)

                    if leaf.is_terminal:
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(leaf, 1.0)
                            return
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

                if not leaves or self._is_timeout():
                    # Clean up virtual losses on timeout
                    if self._is_timeout():
                        for leaf in leaves:
                            self._remove_virtual_loss(leaf)
                    continue

                expanded_nodes = self._expand_batch(leaves)
                
                if self._is_timeout():
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    break
                    
                rewards = self._simulate_batch(expanded_nodes)

                for i in range(len(leaves)):
                    leaf = leaves[i]
                    self._remove_virtual_loss(leaf)
                    child = expanded_nodes[i]
                    reward = rewards[i]
                    self._backpropagate(child, reward)

                    if reward == 1.0:
                        return

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
        """
        Phase 4: Backpropagation
        Update visit counts and value sums from the given node
        all the way back up to the root through ALL parent paths (DAG traversal).
        Uses BFS with visited set to avoid updating nodes multiple times.
        """
        from collections import deque
        
        cdef set visited = set()
        cdef object queue = deque([node])
        cdef Node current
        cdef object node_id
        cdef tuple parent_tuple
        cdef Node parent_node
        
        while queue:
            current = queue.popleft()
            node_id = id(current)
            
            if node_id in visited:
                continue
            visited.add(node_id)
            
            # Update this node
            current.visit_count += 1
            if reward > current.max_value:
                current.max_value = reward
            
            # Add all parents to the queue
            for parent_tuple in current.parents:
                parent_node = <Node>parent_tuple[0]
                if id(parent_node) not in visited:
                    queue.append(parent_node)

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
            # Clear all parent references for the new root (it becomes the root)
            self.root.parents = []
            self.node_count = self._count_nodes(self.root)
            # Rebuild seen_states for the new subtree
            self.seen_states = {}
            self._rebuild_seen_states(self.root)
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
            # Reset seen_states with new root
            self.seen_states = {}
            if isinstance(self.env.current_state, TacticState):
                self.seen_states[self.env.current_state.pp] = self.root

    cpdef int _count_nodes(self, Node node):
        cdef int count = 1
        cdef Node child
        for child in node.children:
            count += self._count_nodes(child)
        return count

    cpdef void _rebuild_seen_states(self, Node node):
        """Recursively rebuild seen_states dictionary from a subtree."""
        cdef object state_key
        cdef Node child
        state_key = self._get_state_key(node.state)
        if state_key is not None:
            self.seen_states[state_key] = node
        for child in node.children:
            self._rebuild_seen_states(child)
