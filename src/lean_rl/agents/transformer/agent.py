"""
Hierarchical Transformer Agent for Lean Theorem Proving.

This module implements the main HierarchicalTransformerAgent that coordinates
all components of the transformer-based hierarchical RL architecture.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from queue import PriorityQueue
import time

from .hierarchy import (
    HierarchicalPolicyNetwork,
    HierarchyLevel,
    StrategicActions,
    TacticalFamilies,
)
from .pointer_network import TacticPointerNetwork, ParameterPointerNetwork

from .parameter_generator import TacticParameterGenerator
from .utils import (
    ProofStateTokenizer,
    TacticEncoder,
    StateEncoder,
)

from ...environment import StepResult
from ..agents import BaseAgent

from lean_dojo import TacticState


@dataclass
class SearchNode:
    """Node in the hierarchical search tree."""

    state: TacticState
    parent: Optional["SearchNode"]
    children: List["SearchNode"]
    level: HierarchyLevel
    action: Optional[str]
    value: float
    visits: int
    strategic_action: Optional[str] = None
    tactic_family: Optional[str] = None
    depth: int = 0

    def __lt__(self, other):
        """For priority queue ordering."""
        return self.value > other.value  # Higher value = higher priority


@dataclass
class HierarchicalAction:
    """Structured action containing decisions at all hierarchy levels."""

    strategic_action: str
    tactic_family: str
    specific_tactic: str
    parameters: List[str]
    confidence: float


class HierarchicalTransformerAgent(BaseAgent):
    """
    Main hierarchical transformer agent for Lean theorem proving.

    This agent coordinates strategic, tactical, and execution levels using
    transformer-based attention mechanisms and pointer networks.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_search_time: float = 60.0,
        beam_width: int = 16,
        device: Optional[str] = None,
    ):

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Model hyperparameters
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Search parameters
        self.max_search_time = max_search_time
        self.beam_width = beam_width

        # Initialize components
        self.tokenizer = ProofStateTokenizer(vocab_size)
        self.tactic_encoder = TacticEncoder()

        # Core neural networks
        self.hierarchical_policy = HierarchicalPolicyNetwork(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        ).to(self.device)

        self.tactic_pointer = TacticPointerNetwork(
            embedding_dim=d_model, hidden_dim=d_model, n_heads=n_heads
        ).to(self.device)

        self.parameter_generator = TacticParameterGenerator(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads
        ).to(self.device)

        # Additional parameter pointer network for more sophisticated parameter selection
        self.parameter_pointer = ParameterPointerNetwork(
            vocab_size=vocab_size, embedding_dim=d_model, hidden_dim=d_model
        ).to(self.device)

        # Training components
        self.optimizer = torch.optim.AdamW(
            list(self.hierarchical_policy.parameters())
            + list(self.tactic_pointer.parameters())
            + list(self.parameter_generator.parameters())
            + list(self.parameter_pointer.parameters()),
            lr=1e-4,
            weight_decay=0.01,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=5
        )

        # Search tree
        self.search_tree = None

        # Validate all required components are initialized
        self._validate_initialization()

    def _validate_initialization(self):
        """Validate that all required components are properly initialized."""
        required_components = [
            ("hierarchical_policy", self.hierarchical_policy),
            ("tactic_pointer", self.tactic_pointer),
            ("parameter_generator", self.parameter_generator),
            ("parameter_pointer", self.parameter_pointer),
            ("tokenizer", self.tokenizer),
            ("tactic_encoder", self.tactic_encoder),
        ]

        for name, component in required_components:
            if component is None:
                raise ValueError(f"Required component '{name}' is not initialized")

            # Check if neural network components are on the correct device
            if hasattr(component, "parameters"):
                try:
                    component_device = next(component.parameters()).device
                    if component_device != self.device:
                        print(
                            f"Warning: Component '{name}' is on device {component_device}, expected {self.device}"
                        )
                        # Move component to correct device
                        component.to(self.device)
                except StopIteration:
                    # Component has no parameters, skip device check
                    pass

        # Verify all components can interact properly
        try:
            # Create a simple test state using TacticState structure
            # This creates a minimal test without using Mock
            test_pp = "test proof state"
            test_goals = ["test goal"]

            # Create a basic encoded state structure
            test_encoded = {
                "input_ids": torch.zeros((1, 10), dtype=torch.long, device=self.device),
                "goal_mask": torch.ones((1, 10), dtype=torch.bool, device=self.device),
                "hypothesis_mask": torch.zeros(
                    (1, 10), dtype=torch.bool, device=self.device
                ),
                "attention_mask": torch.ones(
                    (1, 10), dtype=torch.bool, device=self.device
                ),
            }

            # Test hierarchical forward pass
            output = self.hierarchical_forward(test_encoded, HierarchyLevel.STRATEGIC)
            assert isinstance(output, dict)
            assert "policy_logits" in output

            print("Agent validation completed successfully")

        except Exception as e:
            print(f"Agent validation warning: {e}")
            # Don't fail initialization, just warn

    def select_action(self, state: TacticState, **kwargs) -> Union[str, None]:
        """
        Select action using hierarchical search.

        Args:
            state: Current TacticState from LeanDojo
            **kwargs: Additional arguments

        Returns:
            Selected tactic string or None if no action found
        """
        # Initialize search tree
        self.search_tree = HierarchicalSearchTree(state, self)

        # Perform hierarchical search
        best_action = self.search_tree.search(
            max_time=self.max_search_time, beam_width=self.beam_width
        )

        return best_action

    def update(self, step_result: StepResult) -> None:
        """
        Update the agent based on step result.

        Args:
            step_result: Result from environment step
        """
        # Implement learning update logic with policy gradient
        reward = getattr(step_result, "reward", 0.0)

        # Track episode rewards
        if hasattr(self, "episode_rewards"):
            self.episode_rewards.append(reward)
        else:
            self.episode_rewards = [reward]

        # Initialize action info storage if not exists
        if not hasattr(self, "_last_action_info"):
            self._last_action_info = {}

        # Store experience for batch update if we have action information
        if self._last_action_info:
            experience = {
                "log_prob": self._last_action_info.get("log_prob", torch.tensor(0.0)),
                "value": self._last_action_info.get("value", torch.tensor(0.0)),
                "reward": reward,
            }

            if not hasattr(self, "experience_buffer"):
                self.experience_buffer = []
            self.experience_buffer.append(experience)

            # Perform update every few steps or at episode end
            if len(self.experience_buffer) >= 32 or getattr(step_result, "done", False):
                self._update_policy()

    def _update_policy(self):
        """Update policy using collected experiences."""
        if not hasattr(self, "experience_buffer") or len(self.experience_buffer) == 0:
            return

        # Calculate discounted returns
        returns = []
        R = 0
        for exp in reversed(self.experience_buffer):
            R = exp["reward"] + 0.99 * R  # discount factor = 0.99
            returns.insert(0, R)

        # Convert to tensors
        returns = torch.tensor(returns, device=self.device)
        log_probs = torch.stack([exp["log_prob"] for exp in self.experience_buffer])
        values = torch.stack([exp["value"] for exp in self.experience_buffer])

        # Compute advantages
        advantages = returns - values.squeeze()

        # Policy loss (REINFORCE with baseline)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.hierarchical_policy.parameters(), 0.5)
        self.optimizer.step()

        # Clear experience buffer
        self.experience_buffer = []

    def reset(self) -> None:
        """Reset agent for new episode."""
        self.search_tree = None
        self.episode_rewards = []

    def end_episode(self, episode_reward: float) -> None:
        """Called at end of episode."""
        # Update learning rate based on episode performance
        self.scheduler.step(episode_reward)

    def encode_state(self, state: TacticState) -> Dict[str, torch.Tensor]:
        """
        Encode LeanDojo state for neural networks.

        Args:
            state: TacticState from LeanDojo

        Returns:
            Dictionary with encoded tensors
        """
        # Parse the proof state
        proof_state = self.tokenizer.parse_proof_state(state.pp)

        # Encode as tensors
        encoded = StateEncoder.encode_proof_state_tensor(
            proof_state, self.tokenizer, max_length=512
        )

        # Move to device and add batch dimension
        for key, tensor in encoded.items():
            encoded[key] = tensor.unsqueeze(0).to(self.device)

        return encoded

    def hierarchical_forward(
        self, encoded_state: Dict[str, torch.Tensor], level: HierarchyLevel, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical policy at specified level.

        Args:
            encoded_state: Encoded proof state
            level: Hierarchy level
            **kwargs: Additional arguments for specific levels

        Returns:
            Policy outputs for the specified level
        """
        return self.hierarchical_policy(
            input_ids=encoded_state["input_ids"],
            level=level,
            goal_mask=encoded_state["goal_mask"],
            hypothesis_mask=encoded_state["hypothesis_mask"],
            attention_mask=encoded_state["attention_mask"],
            **kwargs,
        )

    def select_strategic_action(self, state: TacticState) -> str:
        """Select strategic action for given state."""
        encoded_state = self.encode_state(state)

        with torch.no_grad():
            output = self.hierarchical_forward(encoded_state, HierarchyLevel.STRATEGIC)

        # Sample from policy
        logits = output["policy_logits"]
        probabilities = F.softmax(logits, dim=-1)
        action_idx = torch.multinomial(probabilities, 1).item()

        # Ensure action_idx is an integer and within bounds
        action_idx = int(action_idx)
        if action_idx >= len(StrategicActions.ALL_ACTIONS):
            action_idx = 0

        return StrategicActions.ALL_ACTIONS[action_idx]

    def select_tactic_family(self, state: TacticState, strategic_action: str) -> str:
        """Select tactic family given strategic action."""
        encoded_state = self.encode_state(state)

        with torch.no_grad():
            output = self.hierarchical_forward(
                encoded_state,
                HierarchyLevel.TACTICAL,
                strategic_action=strategic_action,
            )

        # Sample from policy
        logits = output["policy_logits"]
        probabilities = F.softmax(logits, dim=-1)
        family_idx = torch.multinomial(probabilities, 1).item()

        # Ensure family_idx is an integer and within bounds
        family_idx = int(family_idx)
        if family_idx >= len(TacticalFamilies.ALL_FAMILIES):
            family_idx = 0

        return TacticalFamilies.ALL_FAMILIES[family_idx]

    def generate_tactic_parameters(
        self, state: TacticState, tactic_family: str, use_pointer_network: bool = False
    ) -> List[str]:
        """Generate parameters for specific tactic family."""
        encoded_state = self.encode_state(state)

        with torch.no_grad():
            if use_pointer_network:
                # Use parameter pointer network for more sophisticated generation
                param_output = self.parameter_pointer(
                    proof_encoding=encoded_state["input_ids"]
                    .squeeze(0)
                    .unsqueeze(0),  # Remove batch dim, re-add
                    tactic_family=tactic_family,
                )

                # Extract parameters from pointer network output
                parameters = []
                if "vocab_logits" in param_output:
                    vocab_logits = param_output["vocab_logits"]
                    if vocab_logits.dim() == 3:  # [batch, seq_len, vocab_size]
                        # Take argmax for each position
                        token_ids = torch.argmax(vocab_logits, dim=-1)
                        for seq in token_ids:
                            valid_tokens = seq[seq != 0].tolist()
                            if valid_tokens:
                                param_str = self.tokenizer.decode(valid_tokens)
                                parameters.append(param_str.strip())
            else:
                # Use original parameter generator
                # Get proof encoding from hierarchical policy
                hierarchical_output = self.hierarchical_forward(
                    encoded_state, HierarchyLevel.EXECUTION, tactic_family=tactic_family
                )

                # Generate parameters using parameter generator
                proof_encoding = hierarchical_output["representation"].unsqueeze(
                    1
                )  # Add seq dim
                param_output = self.parameter_generator(
                    proof_state=proof_encoding, tactic_family=tactic_family
                )

                # Extract and decode generated parameters
                parameters = []

                if "generated_ids" in param_output:
                    # Decode token IDs to parameter strings
                    generated_ids = param_output["generated_ids"]
                    if isinstance(generated_ids, torch.Tensor):
                        # Handle batch of sequences
                        if generated_ids.dim() == 2:
                            for seq in generated_ids:
                                # Filter out padding tokens (0) and decode
                                valid_tokens = seq[seq != 0].tolist()
                                if valid_tokens:
                                    param_str = self.tokenizer.decode(valid_tokens)
                                    parameters.append(param_str.strip())
                        else:
                            # Single sequence
                            valid_tokens = generated_ids[generated_ids != 0].tolist()
                            if valid_tokens:
                                param_str = self.tokenizer.decode(valid_tokens)
                                parameters.append(param_str.strip())

                elif "vocab_logits" in param_output:
                    # Sample from vocabulary distribution
                    vocab_logits = param_output["vocab_logits"]
                    if vocab_logits.dim() == 3:  # [batch, seq_len, vocab_size]
                        # Take argmax for each position
                        token_ids = torch.argmax(vocab_logits, dim=-1)
                        for seq in token_ids:
                            valid_tokens = seq[seq != 0].tolist()
                            if valid_tokens:
                                param_str = self.tokenizer.decode(valid_tokens)
                                parameters.append(param_str.strip())

        # Fallback: extract common parameters based on tactic family (updated for all 16 families)
        if not parameters:
            if tactic_family == "apply_family":
                parameters = ["h"]  # Common hypothesis name
            elif tactic_family == "rewrite_family":
                parameters = ["h", "eq_refl"]  # Common rewrite rules
            elif tactic_family == "intro_family":
                parameters = ["h"]  # Common intro name
            elif tactic_family == "case_family":
                parameters = ["h"]  # Common variable for cases
            elif tactic_family == "calc_family":
                parameters = ["_"]  # Placeholder for calc steps
            elif tactic_family == "finish_family":
                parameters = []  # No parameters needed
            elif tactic_family == "automation_family":
                parameters = []  # Automation tactics usually need no parameters
            elif tactic_family == "proof_family":
                parameters = ["_"]  # Placeholder for proof term
            elif tactic_family == "structural_family":
                parameters = []  # Usually no parameters
            elif tactic_family == "assumption_family":
                parameters = []  # No parameters needed
            elif tactic_family == "advanced_rewrite_family":
                parameters = ["h"]  # Common rewrite rule
            elif tactic_family == "induction_family":
                parameters = ["n"]  # Common induction variable
            elif tactic_family == "quantifier_family":
                parameters = ["_"]  # Witness term placeholder
            elif tactic_family == "conversion_family":
                parameters = ["_"]  # Target expression placeholder
            elif tactic_family == "goal_management_family":
                parameters = []  # Usually no parameters
            elif tactic_family == "specialized_family":
                parameters = ["h"]  # Common hypothesis
            else:
                parameters = []  # Default empty

        return parameters

    def construct_full_action(self, state: TacticState) -> HierarchicalAction:
        """Construct complete hierarchical action."""
        # Strategic level
        strategic_action = self.select_strategic_action(state)

        # Tactical level
        tactic_family = self.select_tactic_family(state, strategic_action)

        # Execution level
        parameters = self.generate_tactic_parameters(state, tactic_family)

        # Map to specific tactic (updated for all 16 families)
        if tactic_family == "apply_family":
            specific_tactic = "apply"
        elif tactic_family == "rewrite_family":
            specific_tactic = "rw"
        elif tactic_family == "intro_family":
            specific_tactic = "intro"
        elif tactic_family == "case_family":
            specific_tactic = "cases"
        elif tactic_family == "calc_family":
            specific_tactic = "calc"
        elif tactic_family == "finish_family":
            specific_tactic = "sorry"
        elif tactic_family == "automation_family":
            specific_tactic = "aesop"
        elif tactic_family == "proof_family":
            specific_tactic = "have"
        elif tactic_family == "structural_family":
            specific_tactic = "constructor"
        elif tactic_family == "assumption_family":
            specific_tactic = "assumption"
        elif tactic_family == "advanced_rewrite_family":
            specific_tactic = "simp_rw"
        elif tactic_family == "induction_family":
            specific_tactic = "induction'"
        elif tactic_family == "quantifier_family":
            specific_tactic = "exists"
        elif tactic_family == "conversion_family":
            specific_tactic = "change"
        elif tactic_family == "goal_management_family":
            specific_tactic = "swap"
        elif tactic_family == "specialized_family":
            specific_tactic = "interval_cases"
        else:
            specific_tactic = "sorry"  # Fallback

        return HierarchicalAction(
            strategic_action=strategic_action,
            tactic_family=tactic_family,
            specific_tactic=specific_tactic,
            parameters=parameters,
            confidence=0.8,  # Placeholder
        )

    def save_model(self, filepath: str) -> None:
        """Save model state with consistent structure."""
        state_dict = {
            "hierarchical_policy": self.hierarchical_policy.state_dict(),
            "tactic_pointer": self.tactic_pointer.state_dict(),
            "parameter_generator": self.parameter_generator.state_dict(),
            "parameter_pointer": self.parameter_pointer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        # Save both formats for compatibility
        checkpoint = {
            "agent_state_dict": state_dict,  # New nested format for training
            "config": {
                "vocab_size": self.vocab_size,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers,
            },
            # Also save individual components for backward compatibility
            **state_dict,
        }

        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str) -> None:
        """Load model state."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)

        # Check if using new nested format
        if "agent_state_dict" in checkpoint:
            state_dict = checkpoint["agent_state_dict"]
        else:
            # Use old format for backward compatibility
            state_dict = checkpoint

        # Load each component
        if "hierarchical_policy" in state_dict:
            self.hierarchical_policy.load_state_dict(state_dict["hierarchical_policy"])
        if "tactic_pointer" in state_dict:
            self.tactic_pointer.load_state_dict(state_dict["tactic_pointer"])
        if "parameter_generator" in state_dict:
            self.parameter_generator.load_state_dict(state_dict["parameter_generator"])
        if "parameter_pointer" in state_dict:
            self.parameter_pointer.load_state_dict(state_dict["parameter_pointer"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "scheduler" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_family_tactics(self, tactic_family: str) -> List[str]:
        """Get all tactics for a given family using metadata."""
        metadata = TacticalFamilies.get_family_metadata()
        family_info = metadata.get(tactic_family, {})
        return family_info.get("tactics", [])

    def get_optimal_tactic_for_family(self, tactic_family: str, goal_str: str) -> str:
        """Select optimal tactic from family based on goal structure."""
        tactics = self.get_family_tactics(tactic_family)
        if not tactics:
            return "sorry"

        # Enhanced heuristics using family metadata
        metadata = TacticalFamilies.get_family_metadata()
        family_info = metadata.get(tactic_family, {})

        if tactic_family == "apply_family":
            # Choose based on goal complexity and available tactics
            if "=" in goal_str and len(goal_str) < 50 and "exact" in tactics:
                return "exact"
            elif "use" in tactics and "∃" in goal_str:
                return "use"
            elif "refine" in tactics and "?" in goal_str:
                return "refine"
            else:
                return "apply"

        elif tactic_family == "rewrite_family":
            if any(op in goal_str for op in ["+", "*", "-", "/"]) and "simp" in tactics:
                return "simp"
            elif "rwa" in tactics and goal_str.endswith("True"):
                return "rwa"
            elif "conv" in tactics and "=" in goal_str:
                return "conv"
            else:
                return "rw"

        elif tactic_family == "automation_family":
            if any(op in goal_str for op in ["+", "*", "-", "^"]) and "ring" in tactics:
                return "ring"
            elif (
                any(op in goal_str for op in ["≤", "≥", "<", ">"])
                and "linarith" in tactics
            ):
                return "linarith"
            elif "norm_num" in tactics and any(str(i) in goal_str for i in range(10)):
                return "norm_num"
            elif "omega" in tactics and any(
                op in goal_str for op in ["≤", "≥", "=", "≠"]
            ):
                return "omega"
            else:
                return tactics[0]  # Default to first available

        # For other families, use simple selection
        return tactics[0]

    def estimate_tactic_success_probability(self, tactic: str, goal_str: str) -> float:
        """Estimate probability of tactic success based on goal structure."""
        # Simple heuristics - could be enhanced with learned models
        if tactic in ["ring", "norm_num"] and any(
            op in goal_str for op in ["+", "*", "-", "^"]
        ):
            return 0.9
        elif tactic in ["linarith", "omega"] and any(
            op in goal_str for op in ["≤", "≥", "<", ">"]
        ):
            return 0.8
        elif tactic == "simp" and any(op in goal_str for op in ["+", "*", "-", "/"]):
            return 0.7
        elif tactic == "exact" and "=" in goal_str and len(goal_str) < 50:
            return 0.8
        elif tactic == "constructor" and any(op in goal_str for op in ["∃", "∧", "↔"]):
            return 0.7
        elif tactic in ["assumption", "trivial"] and len(goal_str) < 20:
            return 0.6
        else:
            return 0.5  # Default probability


class HierarchicalSearchTree:
    """
    Hierarchical search tree for best-first proof search.

    This implements a search algorithm that considers hierarchical action
    decomposition and uses neural heuristics for node evaluation.
    """

    def __init__(self, initial_state: TacticState, agent: HierarchicalTransformerAgent):
        self.agent = agent
        self.root = SearchNode(
            state=initial_state,
            parent=None,
            children=[],
            level=HierarchyLevel.STRATEGIC,
            action=None,
            value=0.0,
            visits=0,
            depth=0,
        )

        # Priority queue for best-first search
        self.open_nodes = PriorityQueue()
        self.open_nodes.put(self.root)

        # Track search statistics
        self.nodes_expanded = 0
        self.max_depth_reached = 0

    def search(self, max_time: float = 60.0, beam_width: int = 16) -> Optional[str]:
        """
        Perform hierarchical best-first search.

        Args:
            max_time: Maximum search time in seconds
            beam_width: Maximum number of nodes to keep in beam

        Returns:
            Best action string or None if no action found
        """
        start_time = time.time()
        best_action = None

        while (
            not self.open_nodes.empty()
            and time.time() - start_time < max_time
            and self.nodes_expanded < 1000
        ):  # Max iterations

            # Get best node from priority queue
            current_node = self.open_nodes.get()

            # Check if we can generate a complete action from this node
            if current_node.depth >= 2:  # Strategic + Tactical levels
                action = self._construct_action_from_node(current_node)
                if action:
                    best_action = action
                    break

            # Expand node
            self._expand_node(current_node)

            # Limit beam width
            if self.open_nodes.qsize() > beam_width:
                self._prune_beam(beam_width)

        return best_action

    def _expand_node(self, node: SearchNode) -> None:
        """Expand a search node by generating child nodes."""
        self.nodes_expanded += 1

        if node.level == HierarchyLevel.STRATEGIC:
            # Generate strategic actions
            for action in StrategicActions.ALL_ACTIONS[:3]:  # Limit for efficiency
                child = SearchNode(
                    state=node.state,
                    parent=node,
                    children=[],
                    level=HierarchyLevel.TACTICAL,
                    action=action,
                    value=self._evaluate_strategic_action(node.state, action),
                    visits=0,
                    strategic_action=action,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                self.open_nodes.put(child)

        elif node.level == HierarchyLevel.TACTICAL:
            # Generate tactical families given strategic action
            for family in TacticalFamilies.ALL_FAMILIES[:3]:  # Limit for efficiency
                child = SearchNode(
                    state=node.state,
                    parent=node,
                    children=[],
                    level=HierarchyLevel.EXECUTION,
                    action=family,
                    value=self._evaluate_tactical_family(
                        node.state, node.strategic_action or "direct_proof", family
                    ),
                    visits=0,
                    strategic_action=node.strategic_action,
                    tactic_family=family,
                    depth=node.depth + 1,
                )
                node.children.append(child)
                self.open_nodes.put(child)

    def _evaluate_strategic_action(self, state: TacticState, action: str) -> float:
        """Evaluate strategic action using neural network."""
        try:
            encoded_state = self.agent.encode_state(state)
            with torch.no_grad():
                output = self.agent.hierarchical_forward(
                    encoded_state, HierarchyLevel.STRATEGIC
                )
                # Simple evaluation - use value network output
                return output["value"].item()
        except Exception:
            return 0.5  # Default value

    def _evaluate_tactical_family(
        self, state: TacticState, strategic_action: str, tactic_family: str
    ) -> float:
        """Evaluate tactical family given strategic action."""
        try:
            encoded_state = self.agent.encode_state(state)
            with torch.no_grad():
                output = self.agent.hierarchical_forward(
                    encoded_state,
                    HierarchyLevel.TACTICAL,
                    strategic_action=strategic_action,
                )
                return output["value"].item()
        except Exception:
            return 0.5  # Default value

    def _construct_action_from_node(self, node: SearchNode) -> Optional[str]:
        """Construct concrete tactic from search node."""
        if node.tactic_family is None:
            return None

        # Map tactic family to specific tactics with intelligent selection (updated for all 16 families)
        family_to_tactics = {
            "apply_family": ["apply", "exact", "refine", "use"],
            "rewrite_family": ["rw", "simp", "conv", "rwa"],
            "intro_family": ["intro", "intros", "rintro"],
            "case_family": ["cases", "rcases", "induction", "split"],
            "calc_family": ["calc", "trans", "symm"],
            "finish_family": ["sorry", "done", "rfl", "trivial"],
            "automation_family": [
                "aesop",
                "tauto",
                "ring",
                "norm_num",
                "linarith",
                "nlinarith",
                "omega",
            ],
            "proof_family": ["have", "show", "suffices", "assert"],
            "structural_family": [
                "constructor",
                "left",
                "right",
                "ext",
                "exfalso",
                "by_contra",
            ],
            "assumption_family": ["assumption", "simp_all", "hint"],
            "advanced_rewrite_family": [
                "simp_rw",
                "rw_mod_cast",
                "simp_intro",
                "field_simp",
            ],
            "induction_family": ["induction'", "cases'", "rcases", "obtain", "choose"],
            "quantifier_family": ["exists", "use!", "existsi", "forall_intro"],
            "conversion_family": ["change", "convert", "congr", "show_term"],
            "goal_management_family": [
                "swap",
                "rotate_left",
                "rotate_right",
                "clear",
                "rename",
            ],
            "specialized_family": [
                "interval_cases",
                "fin_cases",
                "mod_cases",
                "lift",
                "push_neg",
            ],
        }

        # Select best tactic from family based on context (expanded for all families)
        available_tactics = family_to_tactics.get(node.tactic_family, ["sorry"])

        # Enhanced heuristic: choose based on proof state structure
        if node.tactic_family == "apply_family":
            # Prefer 'exact' for simple goals, 'apply' for complex ones
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if "=" in goal_str and len(goal_str) < 50:
                tactic = "exact"
            else:
                tactic = "apply"
        elif node.tactic_family == "rewrite_family":
            # Prefer 'simp' for arithmetic, 'rw' for manual rewriting
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if any(op in goal_str for op in ["+", "*", "-", "/"]):
                tactic = "simp"
            else:
                tactic = "rw"
        elif node.tactic_family == "intro_family":
            # Use 'rintro' for complex patterns, 'intro' otherwise
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if "∃" in goal_str or "∧" in goal_str:
                tactic = "rintro"
            else:
                tactic = "intro"
        elif node.tactic_family == "case_family":
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if "induction" in goal_str.lower():
                tactic = "induction"
            elif "∨" in goal_str:
                tactic = "split"
            else:
                tactic = "cases"
        elif node.tactic_family == "automation_family":
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if any(op in goal_str for op in ["+", "*", "-", "^"]):
                tactic = "ring"
            elif any(op in goal_str for op in ["≤", "≥", "<", ">"]):
                tactic = "linarith"
            else:
                tactic = "aesop"
        elif node.tactic_family == "structural_family":
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if "∃" in goal_str:
                tactic = "constructor"
            elif "∨" in goal_str:
                tactic = "left"  # or "right" based on heuristic
            else:
                tactic = "constructor"
        elif node.tactic_family == "quantifier_family":
            goal_str = str(node.state.pp) if hasattr(node.state, "pp") else ""
            if "∃" in goal_str:
                tactic = "exists"
            else:
                tactic = "use!"
        else:
            tactic = available_tactics[0]

        # Generate parameters using the agent's parameter generator
        parameters = self.agent.generate_tactic_parameters(
            node.state, node.tactic_family
        )

        # Construct tactic string with parameters (expanded for more tactics)
        if tactic == "apply" and parameters:
            if parameters[0].strip():
                return f"apply {parameters[0]}"
            else:
                return "apply h"  # Default hypothesis name
        elif tactic == "exact" and parameters:
            if parameters[0].strip():
                return f"exact {parameters[0]}"
            else:
                return "exact h"
        elif tactic == "rw" and parameters:
            if parameters[0].strip():
                return f"rw [{parameters[0]}]"
            else:
                return "rw [h]"  # Default rewrite rule
        elif tactic == "simp" and parameters:
            if parameters[0].strip():
                return f"simp [{parameters[0]}]"
            else:
                return "simp"
        elif tactic == "intro" and parameters:
            if parameters[0].strip():
                return f"intro {parameters[0]}"
            else:
                return "intro"
        elif tactic == "cases" and parameters:
            if parameters[0].strip():
                return f"cases {parameters[0]}"
            else:
                return "cases h"
        elif tactic == "induction" and parameters:
            if parameters[0].strip():
                return f"induction {parameters[0]}"
            else:
                return "induction n"  # Common induction variable
        elif tactic in ["constructor", "left", "right", "ext", "exfalso", "by_contra"]:
            return tactic  # These typically don't need parameters
        elif tactic in [
            "ring",
            "linarith",
            "norm_num",
            "omega",
            "abel",
            "aesop",
            "tauto",
        ]:
            return tactic  # Automation tactics don't need parameters
        elif tactic in ["assumption", "simp_all", "hint"]:
            return tactic  # Assumption tactics don't need parameters
        elif tactic == "have" and parameters:
            if parameters[0].strip():
                return f"have h : {parameters[0]}"
            else:
                return "have h : _"
        elif tactic == "exists" and parameters:
            if parameters[0].strip():
                return f"exists {parameters[0]}"
            else:
                return "exists _"
        elif tactic == "change" and parameters:
            if parameters[0].strip():
                return f"change {parameters[0]}"
            else:
                return "change _"
        elif tactic in ["swap", "rotate_left", "rotate_right", "clear", "rename"]:
            if tactic == "clear" and parameters:
                return f"clear {parameters[0]}" if parameters[0].strip() else "clear h"
            elif tactic == "rename" and parameters:
                return (
                    f"rename {parameters[0]}"
                    if parameters[0].strip()
                    else "rename h h'"
                )
            else:
                return tactic
        elif tactic in ["interval_cases", "fin_cases", "mod_cases"] and parameters:
            if parameters[0].strip():
                return f"{tactic} {parameters[0]}"
            else:
                return f"{tactic} h"
        else:
            return tactic

    def _prune_beam(self, beam_width: int) -> None:
        """Prune search beam to maintain size limit."""
        # Convert priority queue to list, sort, and rebuild
        nodes = []
        while not self.open_nodes.empty():
            nodes.append(self.open_nodes.get())

        # Keep only top beam_width nodes
        nodes = sorted(nodes, key=lambda x: x.value, reverse=True)[:beam_width]

        # Rebuild priority queue
        self.open_nodes = PriorityQueue()
        for node in nodes:
            self.open_nodes.put(node)
