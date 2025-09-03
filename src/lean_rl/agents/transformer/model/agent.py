"""
Hierarchical Transformer Agent for Lean Theorem Proving.

This module implements the main HierarchicalTransformerAgent that coordinates
all components of the transformer-based hierarchical RL architecture.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F

from lean_dojo import TacticState

from ....environment import StepResult
from ...agents import BaseAgent
from .action import HierarchicalAction
from .hierarchy import (
    HierarchicalPolicyNetwork,
    HierarchyLevel,
    StrategicActions,
    TacticalFamilies,
)
from .parameter_generator import TacticParameterGenerator
from .pointer_network import TacticPointerNetwork, ParameterPointerNetwork
from .search import HierarchicalSearchTree
from .utils import (
    ProofStateTokenizer,
    TacticEncoder,
    StateEncoder,
)


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
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
        )

        self.tactic_pointer = TacticPointerNetwork(
            embedding_dim=d_model, hidden_dim=d_model, n_heads=n_heads
        )

        self.parameter_generator = TacticParameterGenerator(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads
        )

        # Additional parameter pointer network for more sophisticated parameter selection
        self.parameter_pointer = ParameterPointerNetwork(
            vocab_size=vocab_size, embedding_dim=d_model, hidden_dim=d_model
        )

        # Move all sub-models to the correct device
        self.hierarchical_policy.to(self.device)
        self.tactic_pointer.to(self.device)
        self.parameter_generator.to(self.device)
        self.parameter_pointer.to(self.device)

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
        Select an action using a full hierarchical search. This method
        initializes and runs a search, storing detailed information about the
        chosen action for subsequent learning updates.
        """
        self.search_tree = HierarchicalSearchTree(self, state)
        search_result = self.search_tree.search(
            max_time=self.max_search_time, beam_width=self.beam_width
        )

        if search_result:
            action_str, best_node, hierarchical_action = search_result
            if action_str and best_node and hierarchical_action:
                # Store comprehensive information for the learning update
                self._last_action_info = {
                    "log_prob": torch.tensor(best_node.log_prob, device=self.device),
                    "value": torch.tensor(best_node.value, device=self.device),
                    "action_str": action_str,
                    "hierarchical_action": hierarchical_action,
                    "encoded_state": self.encode_state(state),
                }
                return action_str

        # Fallback if search yields no action
        self._last_action_info = {}
        return "sorry"

    def update(self, step_result: StepResult) -> None:
        """
        Update the agent based on step result

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

    def save_model(self, filepath: Optional[str] = None) -> None:
        """Save model state with consistent structure."""
        if filepath is None:
            # Use SCRATCH_SHARED for default save location
            scratch_dir = os.getenv("SCRATCH_SHARED", ".")
            models_dir = Path(scratch_dir) / "saved_models"
            models_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(
                models_dir / f"hierarchical_transformer_{int(time.time())}.pt"
            )

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
        print(f"Model saved to {filepath}")

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

    def get_strategic_action_map(self):
        """Returns the strategic action to index mapping."""
        return self.hierarchical_policy.strategic_policy.action_to_idx

    def evaluate_action(
        self,
        encoded_state: Dict[str, torch.Tensor],
        actions: List[Optional[HierarchicalAction]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a batch of hierarchical actions and return their log probabilities,
        the state value, policy entropies, and the parameter generation loss.
        This is used during training to compute the loss for actions taken in the past.
        """
        batch_size = encoded_state["input_ids"].size(0)
        device = encoded_state["input_ids"].device

        # --- Strategic Level ---
        # This is always computed for all states in the batch.
        strategic_output = self.hierarchical_forward(
            encoded_state, HierarchyLevel.STRATEGIC
        )
        strategic_logits = strategic_output["policy_logits"]
        value = strategic_output["value"]  # Value is estimated at the strategic level.

        strategic_log_probs = F.log_softmax(strategic_logits, dim=-1)
        strategic_probs = F.softmax(strategic_logits, dim=-1)

        strategic_action_map = {
            act: i for i, act in enumerate(StrategicActions.ALL_ACTIONS)
        }
        # Handle None actions for next_state evaluation, default to index 0.
        strategic_action_indices = torch.tensor(
            [
                strategic_action_map.get(a.strategic_action, 0) if a else 0
                for a in actions
            ],
            device=device,
            dtype=torch.long,
        )

        chosen_strategic_log_prob = strategic_log_probs.gather(
            1, strategic_action_indices.unsqueeze(1)
        ).squeeze(1)
        strategic_entropy = -(strategic_probs * strategic_log_probs).sum(-1)

        # --- Tactical and Execution Levels (Grouped by action) ---
        # Initialize tensors to store results for the whole batch.
        total_tactical_log_prob = torch.zeros(batch_size, device=device)
        total_tactical_entropy = torch.zeros(batch_size, device=device)
        total_param_loss = torch.tensor(0.0, device=device)

        # Group actions by (strategic_action, tactic_family) to process them in batches.
        action_groups = {}
        for i, action in enumerate(actions):
            if action is None:  # Skip if no action (e.g., for terminal states)
                continue
            key = (action.strategic_action, action.tactic_family)
            if key not in action_groups:
                action_groups[key] = []
            action_groups[key].append((i, action))  # Store original index and action

        # Process each group
        for (strategic_action, tactic_family), group in action_groups.items():
            indices = torch.tensor(
                [item[0] for item in group], device=device, dtype=torch.long
            )
            group_actions = [item[1] for item in group]

            # --- Tactical Level ---
            # Select the states corresponding to the current group
            group_encoded_state = {k: v[indices] for k, v in encoded_state.items()}

            tactical_output = self.hierarchical_forward(
                group_encoded_state,
                HierarchyLevel.TACTICAL,
                strategic_action=strategic_action,
            )
            tactical_logits = tactical_output["policy_logits"]
            tactical_log_probs = F.log_softmax(tactical_logits, dim=-1)
            tactical_probs = F.softmax(tactical_logits, dim=-1)

            tactic_family_map = {
                fam: i for i, fam in enumerate(TacticalFamilies.ALL_FAMILIES)
            }
            # All actions in this group have the same tactic family
            tactic_family_idx = torch.tensor(
                [tactic_family_map.get(tactic_family, 0)],
                device=device,
                dtype=torch.long,
            ).expand(len(group))

            chosen_tactical_log_prob_group = tactical_log_probs.gather(
                1, tactic_family_idx.unsqueeze(1)
            ).squeeze(1)
            tactical_entropy_group = -(tactical_probs * tactical_log_probs).sum(-1)

            # Use scatter_add_ to place results back in correct batch positions
            total_tactical_log_prob.scatter_add_(
                0, indices, chosen_tactical_log_prob_group
            )
            total_tactical_entropy.scatter_add_(0, indices, tactical_entropy_group)

            # --- Execution (Parameter) Level ---
            if (
                self.parameter_generator.tactic_families.get(tactic_family, {}).get(
                    "max_params", 0
                )
                > 0
            ):
                proof_encoding = tactical_output["representation"]

                param_outputs = self.parameter_generator(
                    proof_state=proof_encoding.unsqueeze(1),
                    tactic_family=tactic_family,
                )

                if (
                    "generated_term" in param_outputs
                    and "logits" in param_outputs["generated_term"]
                ):
                    target_params_str = [
                        " ".join(a.parameters) for a in group_actions if a.parameters
                    ]
                    if target_params_str:
                        target_tokens_list = [
                            torch.tensor(self.tokenizer.encode(s), device=device)
                            for s in target_params_str
                        ]
                        padded_targets = torch.nn.utils.rnn.pad_sequence(
                            target_tokens_list, batch_first=True, padding_value=0
                        )

                        # Ensure batch sizes match for loss calculation
                        if padded_targets.size(0) == param_outputs["generated_term"][
                            "logits"
                        ].size(0):
                            targets = {"target_term": padded_targets}
                            total_param_loss += self.parameter_generator.compute_loss(
                                param_outputs, targets
                            )

        # --- Combine ---
        total_log_prob = chosen_strategic_log_prob + total_tactical_log_prob
        total_entropy = strategic_entropy + total_tactical_entropy

        return total_log_prob, value, total_entropy, total_param_loss
