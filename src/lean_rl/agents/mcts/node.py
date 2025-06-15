"""
MCTS Node implementation.

This module contains the MCTSNode class which represents nodes in the
Monte Carlo Tree Search tree for theorem proving.
"""

import math
from typing import Dict, Optional, Union


class MCTSNode:
    """Node in the MCTS tree representing a game state."""

    def __init__(self, state_key: str, parent=None, action: Union[str, None] = None):
        """
        Initialize an MCTS node.

        Args:
            state_key: Unique identifier for this state (e.g., hash of goals)
            parent: Parent node
            action: Action that led to this node from parent
        """
        self.state_key = state_key
        self.parent = parent
        self.action = action  # Action that led to this state

        # MCTS statistics
        self.visits = 0
        self.total_reward = 0.0
        self.children: Dict[str, "MCTSNode"] = {}  # action -> MCTSNode
        self.untried_actions: Optional[list[str]] = None  # Will be set when expanded

        # Terminal state tracking
        self.is_terminal = False
        self.terminal_reward = 0.0

    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried."""
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0

    def get_ucb1_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """Calculate UCB1 value for action selection."""
        if self.visits == 0:
            return float("inf")

        if self.parent is None or self.parent.visits == 0:
            return self.total_reward / self.visits

        exploitation = self.total_reward / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration

    def add_child(self, action: str, child_state_key: str) -> "MCTSNode":
        """Add a child node for the given action."""
        child = MCTSNode(child_state_key, parent=self, action=action)
        self.children[action] = child
        if self.untried_actions is not None and action in self.untried_actions:
            self.untried_actions.remove(action)
        return child

    def update(self, reward: float) -> None:
        """Update node statistics with a reward."""
        self.visits += 1
        self.total_reward += reward

    def get_best_child(self, exploration_constant: float = 0.0) -> "MCTSNode":
        """Get the best child based on UCB1 or exploitation."""
        return max(
            self.children.values(),
            key=lambda child: child.get_ucb1_value(exploration_constant),
        )

    def get_most_visited_child(self) -> "MCTSNode":
        """Get the child with the most visits."""
        return max(self.children.values(), key=lambda child: child.visits)
