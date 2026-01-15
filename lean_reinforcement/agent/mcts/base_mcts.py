"""
Implementations of MCTS algorithms. Guided-Rollout MCTS does greedy rollout for
simulation, AlphaZero MCTS calls a trained value network for evaluation.
"""

import math
import torch
from typing import List, Optional, Dict
from loguru import logger
import time

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol


class Node:
    """
    A node in the Monte Carlo Tree Search.
    Holds state, statistics, and child nodes.
    Supports DAG structure with multiple parents for state deduplication.
    """

    def __init__(
        self,
        state: TacticState | ProofFinished | LeanError | ProofGivenUp,
        parent: Optional["Node"] = None,
        action: Optional[str] = None,
    ):
        self.state = state
        # Support multiple parents for DAG structure
        # Each entry is (parent_node, action_that_led_here)
        self.parents: List[tuple["Node", Optional[str]]] = []
        if parent is not None:
            self.parents.append((parent, action))
        self.action = (
            action  # Keep for compatibility (first action that created this node)
        )
        self.prior_p = 0.0

        self.children: List["Node"] = []
        self.visit_count = 0
        self.max_value = float("-inf")

        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions: Optional[List[str]] = None

        self.encoder_features: Optional[torch.Tensor] = None

    def add_parent(self, parent: "Node", action: Optional[str] = None) -> None:
        """Add an additional parent to this node (for DAG structure)."""
        # Avoid duplicate parent-action pairs
        if not any(p == parent and a == action for p, a in self.parents):
            self.parents.append((parent, action))

    @property
    def parent(self) -> Optional["Node"]:
        """Backward compatibility: returns first parent or None."""
        return self.parents[0][0] if self.parents else None

    def value(self) -> float:
        """Calculates the value of this node. Using max_value for max-backup."""
        if self.visit_count == 0:
            return 0.0
        # Return max_value instead of mean value
        return self.max_value

    def is_fully_expanded(self) -> bool:
        """Checks if all promising actions from this node have been expanded."""
        return self.untried_actions is not None and len(self.untried_actions) == 0


class BaseMCTS:
    """
    A base class for MCTS, containing the shared logic for the MCTS algorithm framework.
    Subclasses must implement the expansion and simulation strategies.
    """

    def __init__(
        self,
        env: LeanDojoEnv,
        transformer: TransformerProtocol,
        exploration_weight: float = math.sqrt(2),
        max_tree_nodes: int = 10000,
        batch_size: int = 8,
        num_tactics_to_expand: int = 8,
        max_rollout_depth: int = 30,
        max_time: float = 600.0,
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
        self.virtual_losses: Dict[Node, int] = {}

        # Timeout tracking for search operations
        self._search_deadline: Optional[float] = None

        # Seen states dictionary for deduplication (maps state string to Node)
        # This prevents adding duplicate states to the tree, similar to ReProver's approach
        self.seen_states: Dict[str, Node] = {}

        # Get theorem info from the environment
        self.theorem = env.theorem
        self.theorem_pos = env.theorem_pos

        # Initialize the root node with the initial state from the env
        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = Node(state=env.current_state)
        self.node_count = 1

        # Register root state in seen_states
        if isinstance(env.current_state, TacticState):
            self.seen_states[env.current_state.pp] = self.root

    def _get_state_key(
        self, state: TacticState | ProofFinished | LeanError | ProofGivenUp
    ) -> Optional[str]:
        """
        Get a hashable key for a state for deduplication purposes.
        Returns None for terminal states (errors, proof finished) as these
        are not deduplicated.
        """
        if isinstance(state, TacticState):
            return str(state.pp)
        return None

    def _get_virtual_loss(self, node: Node) -> int:
        return self.virtual_losses.get(node, 0)

    def _add_virtual_loss(self, node: Node, loss: int = 1):
        self.virtual_losses[node] = self.virtual_losses.get(node, 0) + loss

    def _remove_virtual_loss(self, node: Node, loss: int = 1):
        if node in self.virtual_losses:
            self.virtual_losses[node] -= loss
            if self.virtual_losses[node] <= 0:
                del self.virtual_losses[node]

    def _log_gpu_memory(self) -> None:
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.debug(
                f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
            )

    def _is_timeout(self) -> bool:
        """Check if the search has exceeded its time limit."""
        if self._search_deadline is None:
            return False
        return time.time() > self._search_deadline

    def search(
        self,
        num_iterations: int,
        batch_size: Optional[int] = None,
        max_time: Optional[float] = None,
    ) -> None:
        """
        Run the MCTS search for a given number of iterations with batching.

        Args:
            num_iterations: Number of MCTS iterations to run.
            batch_size: Batch size for parallel expansion/simulation.
            max_time: Maximum time in seconds for this search. If None, uses self.max_time.
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_time is None:
            max_time = self.max_time

        start_time = time.time()
        # Store deadline as instance var so _expand/_simulate can check it
        self._search_deadline = start_time + max_time

        with torch.no_grad():
            for iteration in range(0, num_iterations, batch_size):
                # Early stopping if solution found
                if self.root.max_value == 1.0:
                    break

                # Check time limit (more frequent check)
                if self._is_timeout():
                    logger.debug(f"MCTS search timeout after {iteration} iterations")
                    break

                # Stop if tree is too large
                if self.node_count >= self.max_tree_nodes:
                    break

                current_batch_size = min(batch_size, num_iterations - iteration)
                leaves = []

                # 1. Selection Phase (Batch)
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

                    # Apply virtual loss to encourage diversity in the batch
                    self._add_virtual_loss(leaf)
                    leaves.append(leaf)

                if not leaves or self._is_timeout():
                    # Clean up virtual losses on early exit
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    if self._is_timeout():
                        break
                    continue

                # 2. Expansion Phase (with timeout awareness)
                expanded_nodes = self._expand_batch(leaves)

                # Check timeout after expansion (it can be slow)
                if self._is_timeout():
                    for leaf in leaves:
                        self._remove_virtual_loss(leaf)
                    break

                # 3. Simulation Phase
                rewards = self._simulate_batch(expanded_nodes)

                # 4. Backpropagation Phase
                for i, leaf in enumerate(leaves):
                    self._remove_virtual_loss(leaf)
                    child = expanded_nodes[i]
                    reward = rewards[i]
                    self._backpropagate(child, reward)

                    if reward == 1.0:
                        return

                # Clear CUDA cache periodically
                if torch.cuda.is_available() and iteration % 20 == 0 and iteration > 0:
                    torch.cuda.empty_cache()

    def _select(self, node: Node) -> Node:
        """
        Phase 1: Selection
        Traverse the tree from the root, picking the best child until a leaf node is reached.
        """
        current = node
        while not current.is_terminal and current.is_fully_expanded():
            if not current.children:
                return current
            current = self._get_best_child(current)
        return current

    def _get_best_child(self, node: Node) -> Node:
        """
        Selects the best child based on the specific MCTS strategy (e.g., UCB1, PUCT).
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _expand(self, node: Node) -> Node:
        """
        Phase 2: Expansion
        This method should be implemented by subclasses. It should expand the
        tree from the given node and return the node from which to start the simulation.
        """
        raise NotImplementedError

    def _expand_batch(self, nodes: List[Node]) -> List[Node]:
        """
        Phase 2: Batch Expansion
        Default implementation calls _expand sequentially.
        Subclasses should override this for parallelism/batching.
        """
        return [self._expand(node) for node in nodes]

    def _simulate(self, node: Node, env: Optional[LeanDojoEnv] = None) -> float:
        """
        Phase 3: Simulation / Evaluation
        This method is meant to be implemented by the child classes.
        """
        raise NotImplementedError

    def _simulate_batch(self, nodes: List[Node]) -> List[float]:
        """
        Phase 3: Batch Simulation
        Default implementation calls _simulate sequentially.
        Subclasses should override this for parallelism/batching.
        """
        return [self._simulate(node) for node in nodes]

    def _backpropagate(self, node: Node, reward: float):
        """
        Phase 4: Backpropagation
        Update visit counts and value sums from the given node
        all the way back up to the root through ALL parent paths (DAG traversal).
        Uses BFS with visited set to avoid updating nodes multiple times.
        """
        from collections import deque

        visited: set[int] = set()  # Track visited nodes by id
        queue: deque[Node] = deque([node])

        while queue:
            current = queue.popleft()
            node_id = id(current)

            if node_id in visited:
                continue
            visited.add(node_id)

            # Update this node
            current.visit_count += 1
            current.max_value = max(current.max_value, reward)

            # Add all parents to the queue
            for parent, _ in current.parents:
                if id(parent) not in visited:
                    queue.append(parent)

    def get_best_action(self) -> Optional[str]:
        """
        After searching, returns the best tactic (action)
        from the root node, based on the highest visit count.
        """
        if not self.root.children:
            # If no children, we might need to generate tactics from the root
            if self.root.untried_actions is None and isinstance(
                self.root.state, TacticState
            ):
                state_str = self.root.state.pp

                self.root.untried_actions = self.transformer.generate_tactics(
                    state_str, n=self.num_tactics_to_expand
                )

            if self.root.untried_actions:
                # Fallback: if search is shallow, return a generated tactic
                return self.root.untried_actions[0]
            return None

        # Select the child with the most visits (most robust)
        best_child = max(self.root.children, key=lambda c: c.visit_count)
        return best_child.action

    def move_root(self, action: str):
        """
        Moves the root of the tree to the child corresponding to the given action.
        This allows for subtree reuse.
        """
        found_child = None
        for child in self.root.children:
            if child.action == action:
                found_child = child
                break

        old_root = self.root

        if found_child:
            self.root = found_child
            # Clear all parent references for the new root (it becomes the root)
            self.root.parents = []

            # CRITICAL: Clear old tree to prevent memory leaks
            # Break circular references before discarding
            if old_root != self.root:
                self._clear_subtree(old_root, keep_node=self.root)

            self.node_count = self._count_nodes(self.root)
            # Rebuild seen_states for the new subtree
            self.seen_states = {}
            self._rebuild_seen_states(self.root)
        else:
            # If child not found, reset the tree with the current environment state
            if not isinstance(
                self.env.current_state,
                (TacticState, ProofFinished, LeanError, ProofGivenUp),
            ):
                raise TypeError(
                    f"Invalid state type for new root: {type(self.env.current_state)}"
                )

            # Clear old tree before creating new root
            self._clear_subtree(old_root, keep_node=None)

            self.root = Node(state=self.env.current_state)
            self.node_count = 1
            # Reset seen_states with new root
            self.seen_states = {}
            if isinstance(self.env.current_state, TacticState):
                self.seen_states[self.env.current_state.pp] = self.root

    def _clear_subtree(self, node: Node, keep_node: Optional[Node]) -> None:
        """
        Recursively clear node references to help garbage collection.
        Breaks circular references in the DAG structure.
        """
        if node is keep_node or node is None:
            return

        # Recursively clear children first
        for child in node.children:
            if child is not keep_node:
                self._clear_subtree(child, keep_node)

        # Clear references to break cycles
        node.children.clear()
        node.parents.clear()
        node.encoder_features = None
        node.state = None
        node.untried_actions = None

    def _count_nodes(self, node: Node) -> int:
        """Recursively counts the number of nodes in the subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _rebuild_seen_states(self, node: Node) -> None:
        """Recursively rebuild seen_states dictionary from a subtree."""
        state_key = self._get_state_key(node.state)
        if state_key is not None:
            self.seen_states[state_key] = node
        for child in node.children:
            self._rebuild_seen_states(child)
