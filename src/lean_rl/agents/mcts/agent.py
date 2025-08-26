"""
MCTS Agent implementation for theorem proving.

This module contains the MCTSAgent class which implements Monte Carlo Tree Search
for automated theorem proving in Lean environments.
"""

import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Union

from ...environment import StepResult
from ...agents import BaseAgent
from node import MCTSNode
from ...heuristic import SimpleNeuralHeuristic


class MCTSAgent(BaseAgent):
    """
    Monte Carlo Tree Search (MCTS) agent for LeanDojo environment.

    This agent implements the standard MCTS algorithm with UCB1 selection,
    random rollouts, and proper backpropagation for theorem proving.
    """

    def __init__(
        self,
        tactics: Optional[List[str]] = None,
        iterations: int = 100,
        exploration_constant: float = math.sqrt(2),
        max_rollout_depth: int = 10,
        rollout_policy: str = "random",
        seed: Optional[int] = None,
        # Neural heuristic parameters
        use_neural_heuristic: bool = False,
        heuristic_learning_rate: float = 0.01,
        feature_size: int = 20,
        hidden_size: int = 32,
    ):
        """
        Initialize the MCTS agent.

        Args:
            tactics: List of available tactics. If None, uses default set.
            iterations: Number of MCTS iterations per action selection
            exploration_constant: UCB1 exploration parameter (typically sqrt(2))
            max_rollout_depth: Maximum depth for rollout simulations
            rollout_policy: Policy for rollouts ("random", "weighted")
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        # Default tactics
        if tactics is None:
            tactics = [
                # High-value tactics that often work
                "rfl",
                "trivial",
                "simp",
                "assumption",
                "exact ?_",
                # Introduction tactics
                "intro",
                "intros",
                "constructor",
                "left",
                "right",
                "use ?_",
                # Elimination tactics
                "cases ?_",
                "apply ?_",
                "have : ?_ := ?_",
                # Rewriting
                "rw [?_]",
                "simp only [?_]",
                # Logic tactics
                "by_contra",
                "exfalso",
                # Arithmetic
                "ring",
                "linarith",
                "norm_num",
                # Advanced
                "tauto",
                "aesop",
                "decide",
                # Meta
                "sorry",
            ]

        self.tactics = tactics
        self.iterations = iterations
        self.exploration_constant = exploration_constant
        self.max_rollout_depth = max_rollout_depth
        self.rollout_policy = rollout_policy

        # Neural heuristic configuration
        self.use_neural_heuristic = use_neural_heuristic
        self.heuristic_learning_rate = heuristic_learning_rate
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # Tree and episode state
        self.root = None
        self.current_node = None
        self.tree_nodes = {}  # state_key -> MCTSNode
        self.previous_state = None  # Track previous state for training

        # Statistics and performance tracking
        self.total_simulations = 0
        self.action_history = []
        self.node_creation_count = 0
        self.cache_hits = 0

        # Tactic weights for weighted rollout policy
        self.tactic_weights = [1.0] * len(self.tactics)

        # Neural heuristic (initialized lazily)
        self.neural_heuristic: Optional[SimpleNeuralHeuristic] = None

        # Experience buffer for training
        self.experience_buffer = []
        self.max_experience = 1000
        self.training_frequency = 10  # Train every N updates

    def _state_to_key(self, state) -> str:
        """Convert a game state to a unique string key."""
        if state is None:
            return "terminal"

        # For TacticState, we can use the goals as the key
        # This assumes states with same goals are equivalent
        if hasattr(state, "goals"):
            try:
                # Try to create a stable string representation of goals
                goals_str = str(state.goals)
                goals_hash = hash(goals_str)
            except:
                # Fallback if goals can't be converted to string
                goals_hash = hash(f"num_goals_{state.num_goals}")

            return f"goals_{state.num_goals}_{goals_hash}"
        else:
            # Fallback to string representation
            return str(state)

    def _get_or_create_node(
        self, state_key: str, parent=None, action: Union[str, None] = None
    ) -> MCTSNode:
        """Get existing node or create new one."""
        if state_key in self.tree_nodes:
            self.cache_hits += 1
            return self.tree_nodes[state_key]

        node = MCTSNode(state_key, parent, action)
        self.tree_nodes[state_key] = node
        self.node_creation_count += 1
        return node

    def _extract_state_features(self, state) -> Dict[str, float]:
        """Extract features from the current state for learning."""
        features = {}

        if state is None:
            return {"terminal": 1.0}

        # Basic features
        num_goals = getattr(state, "num_goals", 1)
        features["num_goals"] = min(num_goals / 10.0, 1.0)  # Normalized
        features["has_goals"] = 1.0 if num_goals > 0 else 0.0
        features["few_goals"] = 1.0 if num_goals <= 2 else 0.0
        features["many_goals"] = 1.0 if num_goals > 5 else 0.0
        features["very_many_goals"] = 1.0 if num_goals > 10 else 0.0

        # Goal complexity features (if available)
        if hasattr(state, "goals") and state.goals:
            goal_text = str(state.goals)
            features["goal_length"] = min(len(goal_text) / 1000.0, 1.0)

            # Common patterns in goals
            features["has_eq"] = 1.0 if "=" in goal_text else 0.0
            features["has_forall"] = (
                1.0 if "∀" in goal_text or "forall" in goal_text else 0.0
            )
            features["has_exists"] = (
                1.0 if "∃" in goal_text or "exists" in goal_text else 0.0
            )
            features["has_implies"] = (
                1.0 if "→" in goal_text or "->" in goal_text else 0.0
            )
            features["has_and"] = 1.0 if "∧" in goal_text or "And" in goal_text else 0.0
            features["has_or"] = 1.0 if "∨" in goal_text or "Or" in goal_text else 0.0
            features["has_not"] = 1.0 if "¬" in goal_text or "Not" in goal_text else 0.0
        else:
            # Default values when no goal text available
            for key in [
                "goal_length",
                "has_eq",
                "has_forall",
                "has_exists",
                "has_implies",
                "has_and",
                "has_or",
                "has_not",
            ]:
                features[key] = 0.0

        return features

    def _extract_action_features(self, action: str) -> Dict[str, float]:
        """Extract features from an action."""
        features = {}

        # Action type features
        features["is_basic"] = (
            1.0 if action in ["rfl", "trivial", "assumption"] else 0.0
        )
        features["is_simp"] = 1.0 if "simp" in action else 0.0
        features["is_intro"] = 1.0 if action in ["intro", "intros"] else 0.0
        features["is_constructor"] = 1.0 if action == "constructor" else 0.0
        features["is_cases"] = 1.0 if "cases" in action else 0.0
        features["is_apply"] = 1.0 if "apply" in action else 0.0
        features["is_rw"] = 1.0 if "rw" in action else 0.0
        features["is_arithmetic"] = (
            1.0 if action in ["ring", "linarith", "norm_num"] else 0.0
        )
        features["has_placeholder"] = 1.0 if "?_" in action else 0.0
        features["is_logic"] = (
            1.0 if action in ["by_contra", "exfalso", "left", "right"] else 0.0
        )
        features["is_advanced"] = 1.0 if action in ["tauto", "aesop", "decide"] else 0.0
        features["is_sorry"] = 1.0 if action == "sorry" else 0.0

        return features

    def _create_feature_vector(self, action: str, state) -> np.ndarray:
        """Create fixed-size feature vector for neural network."""
        state_features = self._extract_state_features(state)
        action_features = self._extract_action_features(action)

        # Create fixed-size vector
        feature_vector = np.zeros(self.feature_size, dtype=np.float32)
        idx = 0

        # Add state features (first 12 slots)
        state_keys = [
            "num_goals",
            "has_goals",
            "few_goals",
            "many_goals",
            "very_many_goals",
            "goal_length",
            "has_eq",
            "has_forall",
            "has_exists",
            "has_implies",
            "has_and",
            "has_or",
        ]
        for key in state_keys:
            if idx < self.feature_size:
                feature_vector[idx] = state_features.get(key, 0.0)
                idx += 1

        # Add action features (remaining slots)
        action_keys = [
            "is_basic",
            "is_simp",
            "is_intro",
            "is_constructor",
            "is_cases",
            "is_apply",
            "is_rw",
            "is_arithmetic",
            "has_placeholder",
            "is_logic",
            "is_advanced",
            "is_sorry",
        ]
        for key in action_keys:
            if idx < self.feature_size:
                feature_vector[idx] = action_features.get(key, 0.0)
                idx += 1

        return feature_vector

    def _init_neural_heuristic(self):
        """Initialize neural network heuristic."""
        if self.neural_heuristic is None:
            self.neural_heuristic = SimpleNeuralHeuristic(
                input_size=self.feature_size,
                hidden_size=self.hidden_size,
                learning_rate=self.heuristic_learning_rate,
            )

    def select_action(self, state, **kwargs) -> str:
        """
        Select an action using MCTS.

        Args:
            state: Current state of the environment
            **kwargs: Additional arguments

        Returns:
            Selected tactic string
        """
        # Store current state as previous state for next update
        self.previous_state = state

        state_key = self._state_to_key(state)

        # Initialize or update root
        if self.root is None or self.root.state_key != state_key:
            self.root = self._get_or_create_node(state_key)
            self.current_node = self.root

        # If this is a terminal state, we can't take actions
        if state is None or getattr(state, "num_goals", 1) == 0:
            return "sorry"  # Fallback action

        # Perform MCTS iterations
        for _ in range(self.iterations):
            self._mcts_iteration(state)

        # Select best action based on visit counts (most robust)
        if not self.root.children:
            # No children explored, fall back to random
            action = random.choice(self.tactics)
        else:
            best_child = self.root.get_most_visited_child()
            action = best_child.action

        # Ensure we always return a valid action
        if action is None:
            action = random.choice(self.tactics)

        self.action_history.append(action)
        return action

    def _mcts_iteration(self, root_state) -> None:
        """Perform one MCTS iteration: Selection, Expansion, Simulation, Backpropagation."""
        self.total_simulations += 1

        # 1. Selection: Start from root and select path to leaf
        node = self.root
        simulated_state = root_state

        if not node:
            return

        # If we reach here, node is a valid MCTSNode.
        path: List[MCTSNode] = [node]

        # Follow best UCB1 path to a leaf
        while not node.is_leaf() and not node.is_terminal:
            if not node.is_fully_expanded():
                break
            node = node.get_best_child(self.exploration_constant)
            path.append(node)
            # Note: We can't actually simulate the state progression here
            # since we don't have access to the environment during planning

        # 2. Expansion: Add a new child if possible
        if not node.is_terminal and (
            node.untried_actions is None or len(node.untried_actions) > 0
        ):
            if node.untried_actions is None:
                # First time expanding this node
                node.untried_actions = self.tactics.copy()
                random.shuffle(node.untried_actions)

            if node.untried_actions:
                # Select an untried action
                action = node.untried_actions[0]
                child_state_key = f"{node.state_key}_child_{len(node.children)}"
                child = node.add_child(action, child_state_key)
                path.append(child)
                node = child

        # 3. Simulation: Estimate value through rollout
        reward = self._simulate_rollout(node, simulated_state)

        # 4. Backpropagation: Update all nodes in path
        for path_node in path:
            path_node.update(reward)

    def _simulate_rollout(self, node: MCTSNode, state) -> float:
        """
        Simulate a random rollout from the given node.

        Since we can't actually execute tactics during planning,
        we use heuristics to estimate the value.
        """
        # Terminal state handling
        if state is None:
            return 1.0  # Proof finished

        if hasattr(state, "num_goals"):
            num_goals = state.num_goals
            if num_goals == 0:
                return 1.0  # Proof finished
            elif num_goals > 20:
                return -0.5  # Too many goals, likely bad

        # Simulate random rollout with heuristic evaluation
        rollout_reward = 0.0
        depth = 0
        current_goals = getattr(state, "num_goals", 1)

        while depth < self.max_rollout_depth:
            depth += 1

            # Select action based on rollout policy
            if self.rollout_policy == "weighted":
                action = random.choices(self.tactics, weights=self.tactic_weights, k=1)[
                    0
                ]
            else:
                action = random.choice(self.tactics)

            # Heuristic evaluation of action
            action_value = self._heuristic_action_value(action, current_goals, state)
            rollout_reward += action_value

            # Simulate effect on goals (very rough heuristic)
            if action in ["rfl", "trivial", "assumption", "simp"]:
                # These often solve goals completely
                if random.random() < 0.3:  # 30% chance of solving
                    rollout_reward += 1.0
                    break
                elif random.random() < 0.4:  # 40% chance of reducing goals
                    current_goals = max(1, current_goals - 1)
                    rollout_reward += 0.1

            elif action in ["intro", "intros", "constructor"]:
                # These might increase goals but make progress
                if random.random() < 0.6:
                    current_goals += random.randint(0, 2)
                    rollout_reward += 0.05

            # Early termination if too many goals
            if current_goals > 10:
                rollout_reward -= 0.2
                break

        return rollout_reward

    def _heuristic_action_value(self, action: str, num_goals: int, state=None) -> float:
        """Learned neural network heuristic value of an action based on context."""
        if not self.use_neural_heuristic:
            # Fallback to simple heuristic
            return self._simple_heuristic_action_value(action, num_goals)

        # Initialize neural network if needed
        if self.neural_heuristic is None:
            self._init_neural_heuristic()

        # At this point, neural_heuristic should not be None
        assert (
            self.neural_heuristic is not None
        ), "Neural heuristic should be initialized"

        # Create feature vector for neural network
        feature_vector = self._create_feature_vector(action, state)

        # Get neural network prediction
        neural_value = self.neural_heuristic.predict(feature_vector)

        # Blend with simple heuristic for stability (especially early in training)
        simple_value = self._simple_heuristic_action_value(action, num_goals)

        # Use confidence based on training count
        confidence = min(self.neural_heuristic.training_count / 100.0, 0.8)

        return confidence * neural_value + (1 - confidence) * simple_value

    def _simple_heuristic_action_value(self, action: str, num_goals: int) -> float:
        """Simple hardcoded heuristic as fallback."""
        # High-value tactics that often work
        if action in ["rfl", "trivial", "assumption"]:
            return 0.3
        elif action in ["simp", "ring", "linarith"]:
            return 0.2
        elif action in ["intro", "intros", "constructor"]:
            return 0.1 if num_goals <= 3 else 0.05
        elif action == "sorry":
            return -0.5  # Discourage unless desperate
        else:
            return 0.0

    def update(self, step_result: StepResult) -> None:
        """Update the agent based on the step result."""
        if self.current_node is None:
            return

        # Get the actual result and update our tree
        actual_reward = step_result.reward

        # Store experience for neural network training
        if self.use_neural_heuristic and self.action_history:
            last_action = self.action_history[-1]

            # Use the tracked previous state
            prev_state = self.previous_state

            # Add to experience buffer
            experience = {
                "action": last_action,
                "state": prev_state,
                "reward": actual_reward,
                "num_goals": getattr(prev_state, "num_goals", 1) if prev_state else 1,
            }
            self.experience_buffer.append(experience)

            # Limit buffer size
            if len(self.experience_buffer) > self.max_experience:
                self.experience_buffer.pop(0)

            # Train neural network periodically
            if (
                len(self.experience_buffer) % self.training_frequency == 0
                and self.neural_heuristic is not None
            ):
                self._train_neural_heuristic()

        # Update the current node with the actual result
        if step_result.state is not None:
            next_state_key = self._state_to_key(step_result.state)

            # If we took an action, find or create the corresponding child
            if self.action_history:
                last_action = self.action_history[-1]
                if last_action in self.current_node.children:
                    self.current_node = self.current_node.children[last_action]
                else:
                    # Create child for this transition
                    self.current_node = self.current_node.add_child(
                        last_action, next_state_key
                    )

        # Update tactic weights based on result (for weighted rollout)
        if self.action_history and self.rollout_policy == "weighted":
            last_action = self.action_history[-1]
            if last_action in self.tactics:
                action_idx = self.tactics.index(last_action)
                learning_rate = 0.1

                if step_result.action_result in ["success", "proof_finished"]:
                    # Increase weight for successful actions
                    self.tactic_weights[action_idx] = (
                        1 - learning_rate
                    ) * self.tactic_weights[action_idx] + learning_rate * 2.0
                elif step_result.action_result == "error":
                    # Decrease weight for failed actions
                    self.tactic_weights[action_idx] = (
                        1 - learning_rate
                    ) * self.tactic_weights[action_idx] + learning_rate * 0.5

        # Handle terminal states
        if step_result.done:
            if self.current_node:
                self.current_node.is_terminal = True
                self.current_node.terminal_reward = actual_reward

            # Train on successful episodes with credit assignment
            if (
                step_result.action_result == "proof_finished"
                and self.use_neural_heuristic
                and self.neural_heuristic is not None
            ):
                self._credit_assignment_training()

    def _train_neural_heuristic(self):
        """Train the neural network on recent experiences."""
        if not self.experience_buffer or self.neural_heuristic is None:
            return

        # Sample recent experiences for training
        batch_size = min(10, len(self.experience_buffer))
        recent_experiences = self.experience_buffer[-batch_size:]

        total_loss = 0.0
        for exp in recent_experiences:
            feature_vector = self._create_feature_vector(exp["action"], exp["state"])
            loss = self.neural_heuristic.train_step(feature_vector, exp["reward"])
            total_loss += loss

        # Optional: print training progress
        if self.neural_heuristic.training_count % 50 == 0:
            avg_loss = self.neural_heuristic.get_average_loss()
            print(
                f"Neural heuristic training: {self.neural_heuristic.training_count} examples, avg loss: {avg_loss:.4f}"
            )

    def _credit_assignment_training(self):
        """Train neural network with credit assignment for successful proofs."""
        if not self.experience_buffer or self.neural_heuristic is None:
            return

        # Give credit to recent actions that led to success
        success_reward = 1.0
        recent_count = min(10, len(self.experience_buffer))

        for i, exp in enumerate(self.experience_buffer[-recent_count:]):
            # Exponential decay for credit assignment
            credit = success_reward * (0.8**i)
            feature_vector = self._create_feature_vector(exp["action"], exp["state"])
            self.neural_heuristic.train_step(feature_vector, credit)

    def reset(self) -> None:
        """Reset the MCTS agent for a new episode."""
        # Don't clear the entire tree - reuse knowledge across episodes
        # Just reset current position
        self.root = None
        self.current_node = None
        self.action_history = []

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the MCTS agent's performance."""
        tree_size = len(self.tree_nodes)
        avg_visits = sum(node.visits for node in self.tree_nodes.values()) / max(
            tree_size, 1
        )

        # Most visited actions
        action_visits = {}
        for node in self.tree_nodes.values():
            for action, child in node.children.items():
                action_visits[action] = action_visits.get(action, 0) + child.visits

        top_actions = sorted(action_visits.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "iterations": self.iterations,
            "tree_size": tree_size,
            "total_simulations": self.total_simulations,
            "node_creation_count": self.node_creation_count,
            "cache_hits": self.cache_hits,
            "average_node_visits": avg_visits,
            "exploration_constant": self.exploration_constant,
            "max_rollout_depth": self.max_rollout_depth,
            "rollout_policy": self.rollout_policy,
            "top_actions_by_visits": top_actions,
            "actions_taken": len(self.action_history),
        }
