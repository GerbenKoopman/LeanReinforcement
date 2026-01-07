import gc
import math
import time
import torch
from typing import List, Optional, Dict, Union
from loguru import logger
from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp
from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol

cdef class Edge:
    def __init__(self, action, float prior, Node child):
        self.action = action
        self.prior = prior
        self.child = child
        self.visit_count = 0

cdef class Node:

    def __init__(self, state: Union[TacticState, ProofFinished, LeanError, ProofGivenUp]):
        self.state = state
        # self.parent and self.action are removed for Graph Search
        # self.prior_p is moved to Edge
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
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
        float exploration_weight=1.41421356,
        int max_tree_nodes=1000,
        int batch_size=8,
        int num_tactics_to_expand=8,
        int max_rollout_depth=30,
        float max_time=1200.0,
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
        
        # Transposition Table
        self.nodes = {}
        self.terminal_nodes = []

        self.theorem = env.theorem
        self.theorem_pos = env.theorem_pos

        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = self._get_or_create_node(env.current_state)

    cpdef Node _get_or_create_node(self, object state):
        """
        Retrieve an existing node from the transposition table or create a new one.
        """
        if isinstance(state, TacticState):
            key = state.pp
            node = self.nodes.get(key)
            if node is not None:
                return node

            node = Node(state)
            self.nodes[key] = node
            self.node_count += 1
            return node

        node = Node(state)
        self.terminal_nodes.append(node)
        self.node_count += 1
        return node

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

    cpdef set _get_reachable_nodes(self, Node root):
        """
        Find all nodes reachable from the given root via BFS.
        Used for pruning unreachable nodes from the transposition table.
        """
        cdef set reachable = set()
        cdef list queue = [root]
        cdef Node node
        cdef Edge edge
        
        while queue:
            node = queue.pop(0)
            if node in reachable:
                continue
            reachable.add(node)
            
            for edge in node.children:
                if edge.child not in reachable:
                    queue.append(edge.child)
        
        return reachable

    cpdef int _prune_unreachable_nodes(self, Node new_root):
        """
        Remove nodes from the transposition table that are not reachable from new_root.
        Returns the number of nodes pruned.
        """
        cdef set reachable = self._get_reachable_nodes(new_root)
        cdef list keys_to_remove = []
        cdef str key
        cdef Node node
        cdef int pruned_count
        
        # Find keys to remove
        for key, node in self.nodes.items():
            if node not in reachable:
                keys_to_remove.append(key)
                # Clear any cached tensors on unreachable nodes
                if node.encoder_features is not None:
                    del node.encoder_features
                    node.encoder_features = None
        
        # Remove unreachable nodes
        for key in keys_to_remove:
            del self.nodes[key]
        
        # Clean up virtual losses for unreachable nodes
        cdef list nodes_with_vl = list(self.virtual_losses.keys())
        for node in nodes_with_vl:
            if node not in reachable:
                del self.virtual_losses[node]
        
        # Clean up terminal nodes that are no longer reachable
        self.terminal_nodes = [n for n in self.terminal_nodes if n in reachable]
        
        for node in reachable:
            if node is not new_root:
                if node.encoder_features is not None:
                    del node.encoder_features
                    node.encoder_features = None
        
        pruned_count = len(keys_to_remove)
        self.node_count = len(self.nodes) + len(self.terminal_nodes)
        
        return pruned_count

    def search(self, int num_iterations, batch_size=None):
        if self.root is None:
            raise RuntimeError("Cannot search: MCTS has been cleaned up or not initialized")

        cdef int iteration
        cdef int current_batch_size
        cdef list leaves
        cdef Node leaf
        cdef list paths
        cdef list path
        cdef list expanded_results
        cdef list rewards
        cdef int i
        cdef float reward
        cdef Node node_to_sim
        cdef Edge edge_to_sim

        if batch_size is None:
            batch_size = self.batch_size
        
        cdef int b_size = batch_size
        cdef int gc_interval = 50
        cdef int prune_interval = 100
        cdef int pruned
        start_time = time.time()

        with torch.no_grad():
            for iteration in range(0, num_iterations, b_size):
                # Periodic garbage collection to prevent memory buildup
                if iteration > 0 and iteration % gc_interval == 0:
                    gc.collect()

                if iteration > 0 and iteration % prune_interval == 0:
                    pruned = self._prune_unreachable_nodes(self.root)
                    if pruned > 0:
                        logger.debug(
                            f"Iteration {iteration}: Pruned {pruned} unreachable nodes"
                        )

                if len(self.nodes) > self.max_tree_nodes:
                    pruned = self._prune_unreachable_nodes(self.root)
                    logger.warning(
                        f"Transposition table exceeded max ({self.max_tree_nodes}), "
                        f"pruned {pruned} nodes"
                    )

                # Early stopping if solution found
                if self.root.max_value == 1.0:
                    break

                # Check time limit
                if time.time() - start_time > self.max_time:
                    break

                if self.node_count >= self.max_tree_nodes:
                    break

                current_batch_size = min(b_size, num_iterations - iteration)
                leaves = []
                paths = []

                # 1. Selection Phase (Batch)
                for _ in range(current_batch_size):
                    leaf, path = self._select(self.root)

                    if leaf.is_terminal:
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(path, 1.0)
                            return
                        elif isinstance(leaf.state, (LeanError, ProofGivenUp)):
                            self._backpropagate(path, -1.0)
                        continue

                    if not isinstance(leaf.state, TacticState):
                        if isinstance(leaf.state, ProofFinished):
                            self._backpropagate(path, 1.0)
                        else:
                            self._backpropagate(path, -1.0)
                        continue

                    # Apply virtual loss to encourage diversity in the batch
                    self._add_virtual_loss(leaf)
                    leaves.append(leaf)
                    paths.append(path)

                if not leaves:
                    continue

                # 2. Expansion Phase
                expanded_results = self._expand_batch(leaves)

                # 3. Simulation Phase
                nodes_to_simulate = [res[0] for res in expanded_results]
                rewards = self._simulate_batch(nodes_to_simulate)

                # 4. Backpropagation Phase
                for i in range(len(leaves)):
                    leaf = leaves[i]
                    self._remove_virtual_loss(leaf)
                    reward = rewards[i]
                    
                    path = paths[i]
                    node_to_sim = expanded_results[i][0]
                    edge_to_sim = expanded_results[i][1]
                    
                    if edge_to_sim is not None:
                        path.append((node_to_sim, edge_to_sim))
                        
                    self._backpropagate(path, reward)
                    if reward == 1.0:
                        return

                # Clear CUDA cache periodically
                if torch.cuda.is_available() and iteration % 20 == 0 and iteration > 0:
                    torch.cuda.empty_cache()

    cpdef tuple _select(self, Node node):
        cdef Node current = node
        cdef list path = [(current, None)]
        cdef Edge edge
        
        while not current.is_terminal and current.is_fully_expanded():
            if not current.children:
                return current, path
            edge = self._get_best_edge(current)
            current = edge.child
            path.append((current, edge))
        return current, path

    cpdef Edge _get_best_edge(self, Node node):
        raise NotImplementedError

    cpdef tuple _expand(self, Node node):
        raise NotImplementedError

    cpdef list _expand_batch(self, list nodes):
        return [self._expand(node) for node in nodes]

    cpdef float _simulate(self, Node node):
        raise NotImplementedError

    cpdef list _simulate_batch(self, list nodes):
        return [self._simulate(node) for node in nodes]

    cpdef void _backpropagate(self, list path, float reward):
        cdef Node node
        cdef Edge edge
        cdef tuple item
        
        for item in reversed(path):
            node = item[0]
            edge = item[1]
            node.visit_count += 1
            node.max_value = max(node.max_value, reward)
            if edge is not None:
                edge.visit_count += 1

    def get_best_action(self):
        cdef Edge best_edge

        if self.root is None:
            return None

        if not self.root.children:
            should_generate = (
                self.root.untried_actions is None
                or len(self.root.untried_actions) == 0
            )

            if should_generate and isinstance(self.root.state, TacticState):
                state_str = self.root.state.pp
                self.root.untried_actions = self.transformer.generate_tactics(
                    state_str, n=self.num_tactics_to_expand
                )

            if self.root.untried_actions:
                return self.root.untried_actions[0]
            return None

        winning_edges = [e for e in self.root.children if e.child.max_value == 1.0]
        if winning_edges:
            best_edge = max(winning_edges, key=lambda Edge e: e.visit_count)
            return best_edge.action

        best_edge = max(self.root.children, key=lambda Edge e: e.visit_count)
        return best_edge.action

    def move_root(self, str action):
        """
        Moves the root of the tree to the child corresponding to the given action.
        This allows for subtree reuse while pruning unreachable nodes.
        """
        if self.root is None:
            raise RuntimeError("Cannot move root: MCTS has been cleaned up or not initialized")

        cdef Edge edge
        cdef int pruned
        found_edge = None
        for edge in self.root.children:
            if edge.action == action:
                found_edge = edge
                break

        if found_edge:
            self.root = found_edge.child
            # Prune nodes that are no longer reachable from the new root
            pruned = self._prune_unreachable_nodes(self.root)
            if pruned > 0:
                logger.debug(f"Pruned {pruned} unreachable nodes from transposition table")
        else:
            if not isinstance(
                self.env.current_state,
                (TacticState, ProofFinished, LeanError, ProofGivenUp),
            ):
                raise TypeError(
                    f"Invalid state type for new root: {type(self.env.current_state)}"
                )

            # Fully reset transposition table and virtual losses to avoid mixing trees.
            self.nodes.clear()
            self.terminal_nodes = []
            self.virtual_losses.clear()
            self.node_count = 0

            self.root = self._get_or_create_node(self.env.current_state)

    def cleanup(self):
        """
        Explicitly clean up all MCTS resources to prevent memory leaks.
        """
        cdef Node node
        
        # Clear encoder features and children from all nodes
        for node in self.nodes.values():
            if node.encoder_features is not None:
                del node.encoder_features
                node.encoder_features = None
            node.children = []

        for node in self.terminal_nodes:
            node.children = []

        # Clear all data structures
        self.nodes.clear()
        self.terminal_nodes = []
        self.virtual_losses.clear()
        self.node_count = 0
        self.root = None

        # Force garbage collection
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cpdef int _count_nodes(self, Node node):
        return len(self.nodes)
