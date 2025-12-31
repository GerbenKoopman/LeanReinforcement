"""
Implementations of MCTS algorithms. Guided-Rollout MCTS does greedy rollout for
simulation, AlphaZero MCTS calls a trained value network for evaluation.
"""

import math
import time
import torch
from typing import List, Optional, Dict
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol


class Edge:
    """
    An edge in the Monte Carlo Search Graph.
    Stores action, prior probability, and points to a child Node.
    """

    def __init__(self, action: str, prior: float, child: "Node"):
        self.action = action
        self.prior = prior
        self.child = child
        self.visit_count = 0


class Node:
    """
    A node in the Monte Carlo Search Graph.
    Holds state, statistics, and outgoing edges.
    """

    def __init__(
        self,
        state: TacticState | ProofFinished | LeanError | ProofGivenUp,
    ):
        self.state = state
        # self.parent and self.action are removed for Graph Search
        # self.prior_p is moved to Edge

        self.children: List[Edge] = []
        self.visit_count = 0
        self.max_value = float("-inf")

        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions: Optional[List[str]] = None

        self.encoder_features: Optional[torch.Tensor] = None

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
        max_time: float = 1200.0,
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

        # Transposition Table
        self.nodes: Dict[str, Node] = {}

        # Get theorem info from the environment
        self.theorem = env.theorem
        self.theorem_pos = env.theorem_pos

        # Initialize the root node with the initial state from the env
        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = self._get_or_create_node(env.current_state)

    def _get_or_create_node(
        self, state: TacticState | ProofFinished | LeanError | ProofGivenUp
    ) -> Node:
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
        self.node_count += 1
        return node

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

    def search(self, num_iterations: int, batch_size: Optional[int] = None) -> None:
        """
        Run the MCTS search for a given number of iterations with batching.
        """
        if batch_size is None:
            batch_size = self.batch_size

        start_time = time.time()

        with torch.no_grad():
            for iteration in range(0, num_iterations, batch_size):
                # Early stopping if solution found
                if self.root.max_value == 1.0:
                    break

                # Check time limit
                if time.time() - start_time > self.max_time:
                    break

                # Stop if tree is too large
                if self.node_count >= self.max_tree_nodes:
                    break

                current_batch_size = min(batch_size, num_iterations - iteration)
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
                for i, leaf in enumerate(leaves):
                    self._remove_virtual_loss(leaf)
                    reward = rewards[i]

                    path = paths[i]
                    node_to_sim, edge_to_sim = expanded_results[i]
                    if edge_to_sim is not None:
                        path.append((node_to_sim, edge_to_sim))

                    self._backpropagate(path, reward)
                    if reward == 1.0:
                        return

                # Clear CUDA cache periodically
                # Clear CUDA cache periodically
                if torch.cuda.is_available() and iteration % 20 == 0 and iteration > 0:
                    torch.cuda.empty_cache()

    def _select(self, node: Node) -> tuple[Node, List[tuple[Node, Optional[Edge]]]]:
        """
        Phase 1: Selection
        Traverse the tree from the root, picking the best child until a leaf node is reached.
        Returns the leaf node and the path taken (list of (node, edge)).
        """
        current = node
        path: List[tuple[Node, Optional[Edge]]] = [(current, None)]

        while not current.is_terminal and current.is_fully_expanded():
            if not current.children:
                return current, path
            edge = self._get_best_edge(current)
            current = edge.child
            path.append((current, edge))
        return current, path

    def _get_best_edge(self, node: Node) -> Edge:
        """
        Selects the best edge based on the specific MCTS strategy (e.g., UCB1, PUCT).
        This method should be implemented by subclasses.
        """
        raise NotImplementedError

    def _expand(self, node: Node) -> tuple[Node, Optional[Edge]]:
        """
        Phase 2: Expansion
        This method should be implemented by subclasses. It should expand the
        tree from the given node and return the node from which to start the simulation,
        and optionally the edge taken to reach it.
        """
        raise NotImplementedError

    def _expand_batch(self, nodes: List[Node]) -> List[tuple[Node, Optional[Edge]]]:
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

    def _backpropagate(self, path: List[tuple[Node, Optional[Edge]]], reward: float):
        """
        Phase 4: Backpropagation
        Update visit counts and value maximums along the path.
        """
        for node, edge in reversed(path):
            node.visit_count += 1
            node.max_value = max(node.max_value, reward)
            if edge:
                edge.visit_count += 1

    def get_best_action(self) -> Optional[str]:
        """
        After searching, returns the best tactic (action)
        from the root node, based on the highest visit count.
        """
        if not self.root.children:
            # If no children, we might need to generate tactics from the root
            should_generate = (
                self.root.untried_actions is None or len(self.root.untried_actions) == 0
            )

            if should_generate and isinstance(self.root.state, TacticState):
                state_str = self.root.state.pp

                self.root.untried_actions = self.transformer.generate_tactics(
                    state_str, n=self.num_tactics_to_expand
                )

            if self.root.untried_actions:
                # Fallback: if search is shallow, return a generated tactic
                return self.root.untried_actions[0]
            return None

        # Select the child with the most visits (most robust)
        winning_edges = [e for e in self.root.children if e.child.max_value == 1.0]
        if winning_edges:
            best_edge = max(winning_edges, key=lambda e: e.visit_count)
            return best_edge.action

        best_edge = max(self.root.children, key=lambda e: e.visit_count)
        return best_edge.action

    def move_root(self, action: str):
        """
        Moves the root of the tree to the child corresponding to the given action.
        This allows for subtree reuse.
        """
        found_edge = None
        for edge in self.root.children:
            if edge.action == action:
                found_edge = edge
                break

        if found_edge:
            self.root = found_edge.child
        else:
            # If child not found, reset the tree with the current environment state
            if not isinstance(
                self.env.current_state,
                (TacticState, ProofFinished, LeanError, ProofGivenUp),
            ):
                raise TypeError(
                    f"Invalid state type for new root: {type(self.env.current_state)}"
                )

            # Fully reset transposition table and virtual losses to avoid mixing trees.
            self.nodes.clear()
            self.virtual_losses.clear()
            self.node_count = 0

            self.root = self._get_or_create_node(self.env.current_state)
