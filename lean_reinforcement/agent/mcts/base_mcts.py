"""
Implementations of MCTS algorithms. Guided-Rollout MCTS does greedy rollout for
simulation, AlphaZero MCTS calls a trained value network for evaluation.
"""

import math
import torch
from typing import List, Optional, Dict
from loguru import logger

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from lean_reinforcement.utilities.gym import LeanDojoEnv
from lean_reinforcement.agent.transformer import TransformerProtocol

# Max depth for a single rollout in Part 1
MAX_ROLLOUT_DEPTH = 30
# Number of tactics to expand from the generator
NUM_TACTICS_TO_EXPAND = 8


class Node:
    """
    A node in the Monte Carlo Tree Search.
    Holds state, statistics, and child nodes.
    """

    def __init__(
        self,
        state: TacticState | ProofFinished | LeanError | ProofGivenUp,
        parent: Optional["Node"] = None,
        action: Optional[str] = None,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior_p = 0.0

        self.children: List["Node"] = []
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
    ):
        self.env = env
        self.transformer = transformer
        self.exploration_weight = exploration_weight
        self.max_tree_nodes = max_tree_nodes
        self.batch_size = batch_size
        self.node_count = 0
        self.virtual_losses: Dict[Node, int] = {}

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

    def _get_virtual_loss(self, node: Node) -> int:
        return self.virtual_losses.get(node, 0)

    def _add_virtual_loss(self, node: Node, loss: int = 1):
        self.virtual_losses[node] = self.virtual_losses.get(node, 0) + loss

    def _remove_virtual_loss(self, node: Node, loss: int = 1):
        if node in self.virtual_losses:
            self.virtual_losses[node] -= loss
            if self.virtual_losses[node] <= 0:
                del self.virtual_losses[node]

    def _log_gpu_memory(self):
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

        with torch.no_grad():
            for iteration in range(0, num_iterations, batch_size):
                # Stop if tree is too large
                if self.node_count >= self.max_tree_nodes:
                    break

                current_batch_size = min(batch_size, num_iterations - iteration)
                leaves = []

                # 1. Selection Phase (Batch)
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

                    # Apply virtual loss to encourage diversity in the batch
                    self._add_virtual_loss(leaf)
                    leaves.append(leaf)

                if not leaves:
                    continue

                # 2. Expansion Phase
                expanded_nodes = self._expand_batch(leaves)

                # 3. Simulation Phase
                rewards = self._simulate_batch(expanded_nodes)

                # 4. Backpropagation Phase
                for i, leaf in enumerate(leaves):
                    self._remove_virtual_loss(leaf)
                    child = expanded_nodes[i]
                    reward = rewards[i]
                    self._backpropagate(child, reward)

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
        all the way back up to the root.
        """
        # Optional because current.parent is later assigned, which can be None
        current: Optional[Node] = node
        while current is not None:
            current.visit_count += 1
            current.max_value = max(current.max_value, reward)
            current = current.parent

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
                    state_str, n=NUM_TACTICS_TO_EXPAND
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

        if found_child:
            self.root = found_child
            self.root.parent = None
            self.node_count = self._count_nodes(self.root)
        else:
            # If child not found, reset the tree with the current environment state
            if not isinstance(
                self.env.current_state,
                (TacticState, ProofFinished, LeanError, ProofGivenUp),
            ):
                raise TypeError(
                    f"Invalid state type for new root: {type(self.env.current_state)}"
                )

            self.root = Node(state=self.env.current_state)
            self.node_count = 1

    def _count_nodes(self, node: Node) -> int:
        """Recursively counts the number of nodes in the subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
