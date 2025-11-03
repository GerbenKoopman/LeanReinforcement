import math
import random
from typing import List, Optional

from lean_dojo import TacticState, ProofFinished, LeanError, ProofGivenUp

from src.utilities.gym import LeanDojoEnv
from .tactic_generation import TacticGenerator
from .value_head import ValueHead
from .premise_selection import PremiseSelector

# Max depth for a single rollout in Part 1
MAX_ROLLOUT_DEPTH = 30
# Number of tactics to expand from the generator
NUM_TACTICS_TO_EXPAND = 5


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
        self.value_sum = 0.0

        self.is_terminal = isinstance(state, (ProofFinished, LeanError, ProofGivenUp))
        self.untried_actions: Optional[List[str]] = None

    def value(self) -> float:
        """Calculates the UCT value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

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
        premise_selector: PremiseSelector,
        tactic_generator: TacticGenerator,
        exploration_weight: float = math.sqrt(2),
        max_tree_nodes: int = 1000,
    ):
        self.env = env
        self.premise_selector = premise_selector
        self.tactic_generator = tactic_generator
        self.exploration_weight = exploration_weight
        self.max_tree_nodes = max_tree_nodes
        self.node_count = 0

        # Get theorem info from the environment
        self.theorem = env.theorem
        self.theorem_pos = env.theorem_pos

        # Load all accessible premises once
        self.all_premises = self.env.dataloader.get_premises(
            self.theorem, self.theorem_pos
        )

        # Initialize the root node with the initial state from the env
        if not isinstance(
            env.current_state, (TacticState, ProofFinished, LeanError, ProofGivenUp)
        ):
            raise TypeError(f"Invalid initial state type: {type(env.current_state)}")

        self.root = Node(state=env.current_state)
        self.node_count = 1

    def search(self, num_iterations: int) -> None:
        """
        Run the MCTS search for a given number of iterations.
        """
        for _ in range(num_iterations):
            # Stop if tree is too large
            if self.node_count >= self.max_tree_nodes:
                break

            leaf = self._select(self.root)

            if leaf.is_terminal:
                # If the leaf is terminal, we can't expand it.
                # We just backpropagate its known value.
                if isinstance(leaf.state, ProofFinished):
                    self._backpropagate(leaf, 1.0)
                elif isinstance(leaf.state, (LeanError, ProofGivenUp)):
                    self._backpropagate(leaf, -1.0)
                continue

            if not isinstance(leaf.state, TacticState):
                # Cannot expand a non-tactic state
                if isinstance(leaf.state, ProofFinished):
                    self._backpropagate(leaf, 1.0)
                else:
                    self._backpropagate(leaf, -1.0)
                continue

            simulation_node = self._expand(leaf)
            reward = self._simulate(simulation_node)
            self._backpropagate(simulation_node, reward)

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

    def _simulate(self, node: Node) -> float:
        """
        Phase 3: Simulation / Evaluation
        This method is meant to be implemented by the child classes.
        """
        raise NotImplementedError

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
            current.value_sum += reward
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
                retrieved = self.premise_selector.retrieve(
                    state_str, self.all_premises, k=10
                )
                # In AlphaZero, we'd use the policy head. Here we adapt.
                self.root.untried_actions = self.tactic_generator.generate_tactics(
                    state_str, retrieved, n=NUM_TACTICS_TO_EXPAND
                )

            if self.root.untried_actions:
                # Fallback: if search is shallow, return a generated tactic
                return self.root.untried_actions[0]
            return None

        # Select the child with the most visits (most robust)
        best_child = max(self.root.children, key=lambda c: c.visit_count)
        return best_child.action


# --- Part 1: MCTS with Guided Rollouts ---


class MCTS_GuidedRollout(BaseMCTS):
    """
    Implements Part 1.
    The _simulate method performs a full "guided rollout"
    using the TacticGenerator greedily until the proof is
    finished or max depth is reached.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _ucb1(self, node: Node) -> float:
        """Calculates the UCB1 score for a node."""
        if node.visit_count == 0:
            return float("inf")

        if (
            node.parent is None
        ):  # Should not happen for nodes in _select, but as a safeguard.
            return node.value()

        # Exploitation term
        exploitation = node.value()

        # Exploration term
        exploration = self.exploration_weight * math.sqrt(
            math.log(node.parent.visit_count) / node.visit_count
        )

        return exploitation + exploration

    def _get_best_child(self, node: Node) -> Node:
        """Selects the best child based on the UCB1 score."""
        return max(node.children, key=self._ucb1)

    def _expand(self, node: Node) -> Node:
        """
        Phase 2: Expansion
        If the node hasn't been expanded yet, generate tactics,
        pick one untried tactic, and create a new child node for it.
        """
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        if node.untried_actions is None:
            # First visit to this node: get premises and generate tactics
            state_str = node.state.pp

            retrieved = self.premise_selector.retrieve(
                state_str, self.all_premises, k=10
            )

            # Use the tactic generator to get N promising tactics
            node.untried_actions = self.tactic_generator.generate_tactics(
                state_str, retrieved, n=NUM_TACTICS_TO_EXPAND
            )
            if node.untried_actions:
                random.shuffle(node.untried_actions)  # Add randomness

        # Pop one untried action
        tactic = ""
        if node.untried_actions:
            tactic = node.untried_actions.pop()

        # Run the tactic in the environment to get the next state
        # This is fast as it doesn't modify the main env, just computes the next state
        next_state_or_result = self.env.dojo_instance.run_tac(node.state, tactic)

        # Create the new child node
        child = Node(next_state_or_result, parent=node, action=tactic)
        node.children.append(child)
        self.node_count += 1

        return child

    def _simulate(self, node: Node) -> float:
        """
        Phase 3: Simulation (Guided Rollout)
        """
        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0  # Should not happen if checks are correct

        current_state: TacticState = node.state

        for _ in range(MAX_ROLLOUT_DEPTH):
            state_str = current_state.pp

            # Get premises and a single greedy tactic
            retrieved = self.premise_selector.retrieve(
                state_str, self.all_premises, k=10
            )
            tactic = self.tactic_generator.generate_tactics(state_str, retrieved, n=1)[
                0
            ]

            # Run the tactic
            result = self.env.dojo_instance.run_tac(current_state, tactic)

            # Check result
            if isinstance(result, ProofFinished):
                return 1.0
            if isinstance(result, (LeanError, ProofGivenUp)):
                return -1.0  # Penalize errors

            if not isinstance(result, TacticState):
                return -1.0  # Should not happen

            current_state = result  # Continue rollout

        return 0.0  # Reached max depth, count as a draw/timeout


# --- Part 2: MCTS with Value Head (AlphaZero-Style) ---


class MCTS_AlphaZero(BaseMCTS):
    """
    Implements Part 2.
    Requires a ValueHead to be passed in.
    The _simulate method is replaced by a single call
    to the ValueHead for evaluation.
    """

    def __init__(self, value_head: ValueHead, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_head = value_head

    def _puct_score(self, node: Node) -> float:
        """Calculates the PUCT score for a node."""
        if node.parent is None:
            return 0.0  # Should not happen for children

        # Q(s,a): Exploitation term (average value)
        q_value = node.value()

        # U(s,a): Exploration term
        exploration = (
            self.exploration_weight
            * node.prior_p
            * (math.sqrt(node.parent.visit_count) / (1 + node.visit_count))
        )

        return q_value + exploration

    def _get_best_child(self, node: Node) -> Node:
        """Selects the best child based on the PUCT score."""
        return max(node.children, key=self._puct_score)

    def _expand(self, node: Node) -> Node:
        """
        Phase 2: Expansion (AlphaZero-style)
        Expand the leaf node by generating all promising actions from the
        policy head, creating a child for each, and storing their prior
        probabilities. Then, return the node itself for simulation.
        """
        if not isinstance(node.state, TacticState):
            raise TypeError("Cannot expand a node without a TacticState.")

        # Get tactics and their prior probabilities from the policy head
        state_str = node.state.pp
        retrieved = self.premise_selector.retrieve(state_str, self.all_premises, k=10)

        # Assume generate_tactics_with_logprobs returns [(tactic, log_prob), ...]
        # And that we convert log_prob to prob.
        # For now, let's assume it returns (tactic, prob)
        tactics_with_probs = self.tactic_generator.generate_tactics_with_probs(
            state_str, retrieved, n=NUM_TACTICS_TO_EXPAND
        )

        # Create a child for each promising tactic
        for tactic, prob in tactics_with_probs:
            next_state = self.env.dojo_instance.run_tac(node.state, tactic)
            child = Node(next_state, parent=node, action=tactic)
            child.prior_p = prob
            node.children.append(child)
            self.node_count += 1

        node.untried_actions = []

        return node

    def _simulate(self, node: Node) -> float:
        """
        Phase 3: Evaluation (using Value Head)
        """
        if node.is_terminal:
            if isinstance(node.state, ProofFinished):
                return 1.0
            if isinstance(node.state, (LeanError, ProofGivenUp)):
                return -1.0

        if not isinstance(node.state, TacticState):
            return -1.0

        # Not terminal, so evaluate with the ValueHead
        state_str = node.state.pp

        # Get premises and predict value
        retrieved = self.premise_selector.retrieve(state_str, self.all_premises, k=10)
        value = self.value_head.predict(state_str, retrieved)

        return value
