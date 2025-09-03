"""
Training Module for Hierarchical Transformer Agent.

This module implements comprehensive training strategies for the HierarchicalTransformerAgent,
including curriculum learning, distributed training, and various learning paradigms.
"""

import os
import gc
import json
import random
import time
import logging
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import deque

from ..data.repository import RepoManager
from ..data.replay_buffer import ExperienceReplayBuffer
from lean_dojo import TacticState, Theorem
from lean_dojo.data_extraction.traced_data import TracedRepo, TracedTheorem

from ..model.agent import (
    HierarchicalTransformerAgent,
    HierarchicalAction,
)
from ..model.hierarchy import (
    HierarchyLevel,
)
from ..config import (
    TrainingConfig,
    ExperimentConfig,
    ModelConfig,
    CurriculumConfig,
    DistributedConfig,
)
from ....environment import LeanEnvironment
from .environment_wrapper import LeanEnvWrapper


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    episode: int = 0
    total_steps: int = 0
    episode_rewards: Optional[List[float]] = None
    episode_lengths: Optional[List[int]] = None
    success_rates: Optional[List[float]] = None
    loss_history: Optional[List[float]] = None

    # Hierarchical metrics
    strategic_accuracy: Optional[List[float]] = None
    tactical_accuracy: Optional[List[float]] = None
    parameter_accuracy: Optional[List[float]] = None

    # Search metrics
    search_depths: Optional[List[int]] = None
    search_times: Optional[List[float]] = None
    nodes_expanded: Optional[List[int]] = None

    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []
        if self.success_rates is None:
            self.success_rates = []
        if self.loss_history is None:
            self.loss_history = []
        if self.strategic_accuracy is None:
            self.strategic_accuracy = []
        if self.tactical_accuracy is None:
            self.tactical_accuracy = []
        if self.parameter_accuracy is None:
            self.parameter_accuracy = []
        if self.search_depths is None:
            self.search_depths = []
        if self.search_times is None:
            self.search_times = []
        if self.nodes_expanded is None:
            self.nodes_expanded = []


class CurriculumManager:
    """Manages curriculum learning for theorem difficulty progression."""

    def __init__(self, config: CurriculumConfig, traced_repo: TracedRepo, repo: Any):
        self.config = config
        self.traced_repo = traced_repo
        self.repo = repo  # Store the original repo object
        self.current_stage = 0
        self.stage_success_rates = []
        self.logger = logging.getLogger(__name__)

        # Organize theorems by difficulty
        self._organize_curriculum()

    def _get_theorem_path(self, theorem: Theorem) -> Path:
        """Helper to get the path of a theorem from its file path."""
        # The file_path in a Theorem object is relative to the repo root
        return Path(self.repo.root_dir) / theorem.file_path

    def _get_or_create_theorem_index(self) -> List[Dict[str, str]]:
        """
        Get theorem index from cache or create a new one, ensuring all theorems
        in the index have corresponding `.ast.json` files on disk.
        """
        cache_dir = os.getenv("CACHE_DIR")
        if not cache_dir:
            self.logger.error("CACHE_DIR environment variable not set!")
            return []

        index_path = Path(cache_dir) / "theorem_index.json"

        if index_path.exists():
            self.logger.info(f"Loading theorem index from {index_path}")
            with open(index_path, "r") as f:
                return json.load(f)

        self.logger.info(
            "Creating new theorem index. This is a one-time operation that may take several minutes."
        )
        self.logger.warning(
            "If you see this message, it's recommended to manually delete the old 'theorem_index.json' "
            f"from '{cache_dir}' to ensure a clean, validated index is created."
        )

        theorem_index = []
        # The IR files are expected in `.lake/build/ir` relative to the repo root.
        ir_dir = Path(self.traced_repo.root_dir) / ".lake/build/ir"

        if not ir_dir.exists():
            self.logger.error(
                f"IR directory not found at {ir_dir}. Cannot verify theorems. "
                "Please ensure the repository was built correctly with 'lake build'."
            )
            return []

        for traced_file in self.traced_repo.traced_files:
            try:
                theorems = traced_file.get_traced_theorems()
                for thm in theorems:
                    # The path in the theorem object is relative to the repo root.
                    # We construct the expected path to the .ast.json file.
                    ast_path = ir_dir / thm.theorem.file_path.with_suffix(".ast.json")

                    if thm.theorem.full_name and ast_path.exists():
                        theorem_index.append(
                            {
                                "file_path": str(thm.theorem.file_path),
                                "full_name": thm.theorem.full_name,
                            }
                        )
            except Exception as e:
                self.logger.warning(
                    f"Could not load theorems from {traced_file.path}: {e}"
                )
                continue

        with open(index_path, "w") as f:
            json.dump(theorem_index, f)

        self.logger.info(
            f"Saved a new, validated theorem index to {index_path} with {len(theorem_index)} theorems."
        )
        return theorem_index

    def _organize_curriculum(self):
        """Organize theorems into curriculum stages using the theorem index."""
        theorem_index = self._get_or_create_theorem_index()
        if not theorem_index:
            self.curriculum_stages = []
            return

        # Convert index back to Theorem objects
        all_theorems = [
            Theorem(self.repo, Path(item["file_path"]), item["full_name"])
            for item in theorem_index
            if item["full_name"]  # Ensure full_name is not None or empty
        ]

        # Sort by difficulty heuristics (e.g., by name length as a proxy)
        def difficulty_score(theorem: Theorem):
            return len(theorem.full_name)

        all_theorems.sort(key=difficulty_score)

        # Divide into curriculum stages
        num_stages = self.config.curriculum_stages
        stage_size = len(all_theorems) // num_stages
        self.curriculum_stages = [
            all_theorems[i * stage_size : (i + 1) * stage_size]
            for i in range(num_stages)
        ]

        self.logger.info(
            f"Organized {len(all_theorems)} theorems into {len(self.curriculum_stages)} stages"
        )

    def get_current_theorems(self) -> List[Theorem]:
        """Get theorems for the current curriculum stage."""
        if not self.curriculum_stages:
            # Fallback to a small set of theorems if curriculum setup failed
            self.logger.warning("Curriculum stages are empty, using fallback.")
            return []
        return self.curriculum_stages[
            min(self.current_stage, len(self.curriculum_stages) - 1)
        ]

    def should_advance_stage(self, recent_success_rate: float) -> bool:
        """Check if we should advance to the next curriculum stage."""
        return (
            recent_success_rate >= self.config.difficulty_threshold
            and self.current_stage < len(self.curriculum_stages) - 1
        )

    def advance_stage(self):
        """Advance to the next curriculum stage."""
        if self.current_stage < len(self.curriculum_stages) - 1:
            self.current_stage += 1
            self.logger.info(f"Advanced to curriculum stage {self.current_stage}")


class HierarchicalTransformerTrainer:
    """Main trainer for the hierarchical transformer agent."""

    def __init__(
        self,
        config: ExperimentConfig,
        no_evaluation: bool = False,
        output_dir: Optional[str] = None,
    ):
        self.config = config
        self.no_evaluation = no_evaluation
        self.output_dir = Path(output_dir) if output_dir else None
        self.device = torch.device(
            f"cuda:{config.distributed.local_rank}"
            if torch.cuda.is_available() and config.distributed.use_distributed
            else "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        # Validate configuration
        self._validate_config()

        # Setup logging
        self._setup_logging()

        # Initialize distributed training if needed
        if config.distributed and config.distributed.use_distributed:
            self._setup_distributed()

        # Setup repositories and data
        self._setup_repository()

        # Initialize models and training components
        self._setup_models()
        self._setup_training()

        # Setup evaluation and logging
        self._setup_evaluation()

        # Initialize metrics
        self.metrics = TrainingMetrics()

        # Validate configuration
        self._validate_config()

        # Disable evaluation phase if specified
        self.no_evaluation = no_evaluation

    def _setup_logging(self):
        """Setup logging configuration."""
        if self.output_dir:
            log_dir = self.output_dir / "training_logs"
        else:
            # Fallback to environment variable if output_dir is not provided
            scratch_dir = os.getenv("SCRATCH_SHARED", ".")
            log_dir = Path(scratch_dir) / "training_logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = (
            log_dir / f"training_{self.config.experiment_name}_{int(time.time())}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _setup_distributed(self):
        """Setup distributed training."""
        if self.config.distributed and self.config.distributed.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.config.distributed.rank)

    def _setup_repository(self):
        """Setup Lean repository and trace data."""
        self.logger.info("Setting up repository...")

        self.repo_manager = RepoManager(self.config.repo_url, self.config.repo_commit)
        self.traced_repo = self.repo_manager.get_traced_repo()
        self.repo = self.repo_manager.repo

        # Setup curriculum if enabled
        if self.config.curriculum and self.config.curriculum.use_curriculum:
            self.curriculum_manager = CurriculumManager(
                self.config.curriculum, self.traced_repo, self.repo
            )
        else:
            self.curriculum_manager = None

    def _setup_models(self):
        """Setup neural network models."""
        self.logger.info("Initializing models...")

        model_config = self.config.model
        # Main agent
        self.agent = HierarchicalTransformerAgent(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
            device=str(self.device),
        )

        # Target network for stability (optional)
        self.target_agent = HierarchicalTransformerAgent(
            vocab_size=model_config.vocab_size,
            d_model=model_config.d_model,
            n_heads=model_config.n_heads,
            n_layers=model_config.n_layers,
            dropout=model_config.dropout,
            device=str(self.device),
        )

        # Copy weights to target network
        self._update_target_network(is_hard_update=True)

        # Move models to the correct device
        self._set_agent_device(self.agent)
        self._set_agent_device(self.target_agent)

        # Setup distributed training
        if self.config.distributed and self.config.distributed.use_distributed:
            self.agent = DDP(
                self.agent, device_ids=[self.config.distributed.local_rank]
            )

    def _setup_training(self):
        """Setup training components."""
        self.logger.info("Setting up training components...")

        # Optimizers - get parameters from the agent using the helper method
        agent_parameters = self._get_agent_parameters()

        self.optimizer = torch.optim.AdamW(
            agent_parameters,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=5
        )

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.parameter_loss_fn = nn.CrossEntropyLoss()

        # Experience replay
        if self.config.training.use_experience_replay:
            self.replay_buffer = ExperienceReplayBuffer(
                self.config.training.replay_buffer_size,
                self.device,
                agent=self.agent,  # Pass the agent (or DDP-wrapped agent)
            )

        # Environment
        self.env = LeanEnvWrapper(
            LeanEnvironment(
                repo=self.repo_manager.repo,
                timeout=self.config.training.timeout,
                max_steps=self.config.training.max_steps_per_episode,
                reward_scheme=self.config.training.reward_scheme,
            )
        )

    def _setup_evaluation(self):
        """Setup evaluation and logging, including a persistent evaluation environment."""
        if self.output_dir:
            base_dir = self.output_dir
        else:
            # Fallback to environment variable if output_dir is not provided
            base_dir = Path(os.getenv("SCRATCH_SHARED", "."))

        # Tensorboard writer
        if self.config.distributed.rank == 0:  # Only main process writes logs
            log_dir = (
                base_dir
                / "tensorboard_logs"
                / f"exp_{self.config.experiment_name}_{int(time.time())}"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # Checkpoint directory
        self.checkpoint_dir = (
            base_dir / "checkpoints" / f"exp_{self.config.experiment_name}"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create a persistent environment for evaluation to avoid re-initialization costs
        if not self.no_evaluation:
            self.logger.info("Setting up persistent evaluation environment...")
            self.eval_env = LeanEnvironment(
                repo=self.repo_manager.repo,
                timeout=self.config.training.timeout,
                max_steps=self.config.training.max_steps_per_episode,
                reward_scheme=self.config.training.reward_scheme,
            )
        else:
            self.eval_env = None

    def _get_agent_parameters(self):
        """Get all parameters from agent submodules."""
        agent_module = self.agent.module if isinstance(self.agent, DDP) else self.agent
        params = []
        if hasattr(agent_module, "hierarchical_policy"):
            params.extend(list(agent_module.hierarchical_policy.parameters()))
        if hasattr(agent_module, "tactic_pointer"):
            params.extend(list(agent_module.tactic_pointer.parameters()))
        if hasattr(agent_module, "parameter_generator"):
            params.extend(list(agent_module.parameter_generator.parameters()))
        if hasattr(agent_module, "parameter_pointer"):
            params.extend(list(agent_module.parameter_pointer.parameters()))
        return params

    def _get_agent_state_dict(self, agent_instance):
        """Get state dict from all agent submodules."""
        agent_module = (
            agent_instance.module if isinstance(agent_instance, DDP) else agent_instance
        )
        state_dict = {}
        if hasattr(agent_module, "hierarchical_policy"):
            state_dict["hierarchical_policy"] = (
                agent_module.hierarchical_policy.state_dict()
            )
        if hasattr(agent_module, "tactic_pointer"):
            state_dict["tactic_pointer"] = agent_module.tactic_pointer.state_dict()
        if hasattr(agent_module, "parameter_generator"):
            state_dict["parameter_generator"] = (
                agent_module.parameter_generator.state_dict()
            )
        if hasattr(agent_module, "parameter_pointer"):
            state_dict["parameter_pointer"] = (
                agent_module.parameter_pointer.state_dict()
            )
        return state_dict

    def _load_agent_state_dict(self, agent, state_dict):
        """Load state dict into agent submodules."""
        agent_module = agent.module if isinstance(agent, DDP) else agent
        if (
            hasattr(agent_module, "hierarchical_policy")
            and "hierarchical_policy" in state_dict
        ):
            agent_module.hierarchical_policy.load_state_dict(
                state_dict["hierarchical_policy"]
            )
        if hasattr(agent_module, "tactic_pointer") and "tactic_pointer" in state_dict:
            agent_module.tactic_pointer.load_state_dict(state_dict["tactic_pointer"])
        if (
            hasattr(agent_module, "parameter_generator")
            and "parameter_generator" in state_dict
        ):
            agent_module.parameter_generator.load_state_dict(
                state_dict["parameter_generator"]
            )
        if (
            hasattr(agent_module, "parameter_pointer")
            and "parameter_pointer" in state_dict
        ):
            agent_module.parameter_pointer.load_state_dict(
                state_dict["parameter_pointer"]
            )

    def _set_agent_mode(self, agent, train_mode):
        """Set train/eval mode for agent submodules."""
        agent_module = agent.module if isinstance(agent, DDP) else agent
        if hasattr(agent_module, "hierarchical_policy"):
            agent_module.hierarchical_policy.train(train_mode)
        if hasattr(agent_module, "tactic_pointer"):
            agent_module.tactic_pointer.train(train_mode)
        if hasattr(agent_module, "parameter_generator"):
            agent_module.parameter_generator.train(train_mode)
        if hasattr(agent_module, "parameter_pointer"):
            agent_module.parameter_pointer.train(train_mode)

    def _set_agent_device(self, agent):
        """Move all submodules of an agent to the correct device."""
        agent_module = agent.module if isinstance(agent, DDP) else agent
        if hasattr(agent_module, "hierarchical_policy"):
            agent_module.hierarchical_policy.to(self.device)
        if hasattr(agent_module, "tactic_pointer"):
            agent_module.tactic_pointer.to(self.device)
        if hasattr(agent_module, "parameter_generator"):
            agent_module.parameter_generator.to(self.device)
        if hasattr(agent_module, "parameter_pointer"):
            agent_module.parameter_pointer.to(self.device)

    def _update_target_network(self, is_hard_update: bool = False):
        """Update target network with current agent weights."""
        try:
            target_state_dict = self._get_agent_state_dict(self.target_agent)
            agent_state_dict = self._get_agent_state_dict(self.agent)

            if is_hard_update:
                # Hard update
                self._load_agent_state_dict(self.target_agent, agent_state_dict)
            else:
                # Soft update
                tau = self.config.training.target_update_tau
                updated_state_dict = {}
                for key in target_state_dict:
                    updated_state_dict[key] = (
                        tau * agent_state_dict[key] + (1 - tau) * target_state_dict[key]
                    )
                self._load_agent_state_dict(self.target_agent, updated_state_dict)

        except Exception as e:
            self.logger.warning(f"Could not update target network: {e}")

    def _validate_config(self):
        """Validate configuration to prevent runtime errors."""
        try:
            assert self.config.training.max_episodes > 0
            assert self.config.training.learning_rate > 0
            if self.config.distributed.use_distributed:
                assert self.config.distributed.world_size > 0
        except AssertionError as e:
            self.logger.error(f"Invalid configuration: {e}")
            raise

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")

        episode = 0
        best_success_rate = 0.0
        recent_rewards = deque(maxlen=100)
        recent_successes = deque(maxlen=100)

        while episode < self.config.training.max_episodes:
            if self.curriculum_manager:
                theorems = self.curriculum_manager.get_current_theorems()
                if not theorems:
                    self.logger.warning(
                        f"No theorems in current curriculum stage {self.curriculum_manager.current_stage}. Ending training."
                    )
                    break
                theorem = random.choice(theorems)
            else:
                # Fallback if curriculum is disabled or fails
                all_theorems = self._get_all_theorems()
                if not all_theorems:
                    self.logger.error("No theorems found to train on. Aborting.")
                    break
                theorem = random.choice(all_theorems)

            reward, length, success = self._run_episode(theorem)

            # Logging and checkpointing
            if self.config.distributed.rank == 0:
                recent_rewards.append(reward)
                recent_successes.append(success)

                if episode % self.config.training.log_frequency == 0:
                    self._log_progress(episode, recent_rewards, recent_successes)

                if (
                    not self.no_evaluation
                    and episode > 0
                    and episode % self.config.training.eval_frequency == 0
                ):
                    eval_success_rate = self._evaluate()
                    if eval_success_rate > best_success_rate:
                        best_success_rate = eval_success_rate
                        self._save_checkpoint(episode, "best")

                if episode % self.config.training.save_frequency == 0:
                    self._save_checkpoint(episode, "periodic")

            # Update curriculum
            if self.curriculum_manager and self.config.distributed.rank == 0:
                if len(recent_successes) > 0:
                    avg_success = sum(recent_successes) / len(recent_successes)
                    if self.curriculum_manager.should_advance_stage(avg_success):
                        self.curriculum_manager.advance_stage()

            episode += 1

        self.logger.info("Training completed!")
        if self.config.distributed.rank == 0:
            self._save_checkpoint(episode, "final")

        if self.writer:
            self.writer.close()

    def _run_episode(self, theorem: Theorem) -> Tuple[float, int, bool]:
        """Run a single training episode."""
        agent_module = self.agent.module if isinstance(self.agent, DDP) else self.agent
        try:
            state = self.env.reset(theorem)
            agent_module.reset()

            total_reward = 0.0
            done = False
            steps = 0
            success = False

            while not done and steps < self.config.training.max_steps_per_episode:
                if state is None:
                    break  # End episode if state is None

                # Construct the full hierarchical action
                hierarchical_action = agent_module.construct_full_action(state)
                if hierarchical_action is None:
                    break

                # Get the string representation for the environment
                action_str = str(hierarchical_action)
                next_state, reward, done, result = self.env.step(action_str)

                if result.action_result == "proof_finished":
                    success = True

                if self.replay_buffer:
                    # Store the full action object in the replay buffer
                    self.replay_buffer.push(
                        state, hierarchical_action, reward, next_state, done
                    )

                state = next_state
                total_reward += reward
                steps += 1

                if (
                    self.replay_buffer
                    and steps % self.config.training.train_frequency == 0
                ):
                    self._train_from_replay()

            return total_reward, steps, success

        except Exception as e:
            self.logger.error(
                f"Error during episode with theorem {theorem.full_name}: {e}"
            )
            return 0.0, 0, False

    def _train_from_replay(self):
        """Train the model using experience replay."""
        if (
            not self.replay_buffer
            or len(self.replay_buffer) < self.config.training.replay_batch_size
        ):
            return

        batch = self.replay_buffer.sample(self.config.training.replay_batch_size)
        if batch is None:
            return

        # Compute loss and update model
        loss = self._compute_loss(batch)
        if loss is None:
            return

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self._get_agent_parameters(), self.config.training.gradient_clip_norm
        )
        self.optimizer.step()

        # Update target network
        self._update_target_network()

    def _compute_loss(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """
        Compute a fully hierarchical loss for a batch of experiences using an Actor-Critic style loss.
        This implementation correctly handles the multi-level nature of the agent's actions.
        """
        agent_module = self.agent.module if isinstance(self.agent, DDP) else self.agent
        target_agent_module = (
            self.target_agent.module
            if isinstance(self.target_agent, DDP)
            else self.target_agent
        )

        try:
            rewards = batch["rewards"]
            dones = batch["dones"]
            actions = batch["actions"]
            encoded_states = batch["encoded_states"]
            encoded_next_states = batch["encoded_next_states"]

            if not encoded_states:
                return None

            # --- Get action log-probabilities and state values from the agent ---
            (
                action_log_probs,
                values,
                entropy,
                param_loss,
            ) = agent_module.evaluate_action(encoded_states, actions)
            values = values.squeeze(-1)

            # --- Get values for next states from the target network ---
            next_values = torch.zeros_like(rewards, device=self.device)
            if encoded_next_states:
                with torch.no_grad():
                    next_output = target_agent_module.hierarchical_forward(
                        encoded_next_states, HierarchyLevel.STRATEGIC
                    )
                    next_state_values = next_output["value"].squeeze(-1)

                # Create a mask for non-final states
                non_final_mask = torch.tensor(
                    [s is not None for s in batch["next_states"]],
                    device=self.device,
                    dtype=torch.bool,
                )
                next_values[non_final_mask] = next_state_values

            # --- Compute target Q-values and advantage ---
            target_q_values = rewards + (
                ~dones * self.config.training.gamma * next_values
            )
            advantages = target_q_values - values

            # --- Calculate losses ---
            policy_loss = -(action_log_probs * advantages.detach()).mean()
            value_loss = self.value_loss_fn(values, target_q_values.detach())
            entropy_loss = -self.config.training.entropy_coefficient * entropy.mean()

            # Total Loss
            total_loss = (
                policy_loss
                + self.config.training.value_loss_coefficient * value_loss
                + entropy_loss
                + self.config.training.parameter_loss_coefficient * param_loss
            )

            # Log losses
            if self.writer and self.config.distributed.rank == 0:
                self.writer.add_scalar(
                    "Loss/Policy", policy_loss.item(), self.metrics.total_steps
                )
                self.writer.add_scalar(
                    "Loss/Value", value_loss.item(), self.metrics.total_steps
                )
                self.writer.add_scalar(
                    "Loss/Entropy", entropy_loss.item(), self.metrics.total_steps
                )
                self.writer.add_scalar(
                    "Loss/Parameter", param_loss.item(), self.metrics.total_steps
                )
                self.writer.add_scalar(
                    "Loss/Total", total_loss.item(), self.metrics.total_steps
                )

            return total_loss

        except Exception as e:
            self.logger.error(f"Error during loss computation: {e}", exc_info=True)
            return None

    def _evaluate(self) -> float:
        """
        Evaluate the agent's performance on a set of validation theorems
        using a persistent, pre-initialized environment for efficiency.
        """
        if self.eval_env is None:
            self.logger.warning(
                "Evaluation environment not initialized. Skipping evaluation."
            )
            return 0.0

        self.logger.info("Starting evaluation...")
        self._set_agent_mode(self.agent, train_mode=False)
        agent_module = self.agent.module if isinstance(self.agent, DDP) else self.agent

        if self.curriculum_manager:
            # Get a fixed set of theorems for evaluation from the current stage
            eval_theorems = self.curriculum_manager.get_current_theorems()[
                : self.config.training.eval_episodes
            ]
        else:
            # Fallback to a random sample if curriculum is off
            eval_theorems = random.sample(
                self._get_all_theorems(), self.config.training.eval_episodes
            )

        if not eval_theorems:
            self.logger.warning("No theorems available for evaluation.")
            self._set_agent_mode(self.agent, train_mode=True)
            return 0.0

        proved_count = 0
        for theorem in eval_theorems:
            try:
                state = self.eval_env.reset(theorem)
                agent_module.reset()
                done = False
                steps = 0

                while not done and steps < self.config.training.max_steps_per_episode:
                    if state is None:
                        break
                    action = agent_module.construct_full_action(state)
                    if action is None:
                        break
                    result = self.eval_env.step(str(action))
                    state = result.state
                    done = result.done
                    if result.action_result == "proof_finished":
                        proved_count += 1
                    steps += 1

            except Exception as e:
                self.logger.error(
                    f"Error during evaluation with theorem {theorem.full_name}: {e}"
                )
                continue

        success_rate = (
            proved_count / len(eval_theorems) if len(eval_theorems) > 0 else 0.0
        )
        self.logger.info(f"Evaluation success rate: {success_rate:.3f}")

        self._set_agent_mode(self.agent, train_mode=True)
        return success_rate

    def _log_progress(
        self, episode: int, recent_rewards: deque, recent_successes: deque
    ):
        """Log training progress."""
        if not recent_rewards or not recent_successes:
            return

        avg_reward = sum(recent_rewards) / len(recent_rewards)
        avg_success = sum(recent_successes) / len(recent_successes)

        self.logger.info(
            f"Episode {episode} | Avg Reward: {avg_reward:.3f} | Avg Success: {avg_success:.3f}"
        )

        if self.writer:
            self.writer.add_scalar("Reward/Average", avg_reward, episode)
            self.writer.add_scalar("Success/Average", avg_success, episode)
            self.writer.add_scalar(
                "LearningRate", self.optimizer.param_groups[0]["lr"], episode
            )

    def _save_checkpoint(self, episode: int, checkpoint_type: str):
        """Save model checkpoint."""
        if self.config.distributed.rank != 0:
            return

        checkpoint_path = (
            self.checkpoint_dir / f"checkpoint_{checkpoint_type}_ep{episode}.pt"
        )
        state = {
            "episode": episode,
            "agent_state_dict": self._get_agent_state_dict(self.agent),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": asdict(self.config),
        }
        torch.save(state, checkpoint_path)
        self.logger.info(f"Saved {checkpoint_type} checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint path not found: {checkpoint_path}")
            return

        state = torch.load(checkpoint_path, map_location=self.device)
        self._load_agent_state_dict(self.agent, state["agent_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")

    def _get_all_theorems(self) -> List[Theorem]:
        """Get all theorems from the traced repository."""
        if not hasattr(self, "_all_theorems_cache"):
            all_theorems_info = self._get_or_create_theorem_index()
            self._all_theorems_cache = [
                Theorem(self.repo, Path(item["file_path"]), item["full_name"])
                for item in all_theorems_info
                if item["full_name"]  # Ensure full_name is not None or empty
            ]
        return self._all_theorems_cache

    def _get_or_create_theorem_index(self) -> List[Dict[str, str]]:
        """
        Get theorem index from cache or create a new one, ensuring all theorems
        in the index have corresponding `.ast.json` files on disk.
        """
        cache_dir = os.getenv("CACHE_DIR")
        if not cache_dir:
            self.logger.error("CACHE_DIR environment variable not set!")
            return []

        index_path = Path(cache_dir) / "theorem_index.json"

        if index_path.exists():
            self.logger.info(f"Loading theorem index from {index_path}")
            with open(index_path, "r") as f:
                return json.load(f)

        # If index doesn't exist, create it (this part is also in CurriculumManager)
        self.logger.info(
            "Creating new theorem index. This is a one-time operation that may take several minutes."
        )
        self.logger.warning(
            "If you see this message, it's recommended to manually delete the old 'theorem_index.json' "
            f"from '{cache_dir}' to ensure a clean, validated index is created."
        )
        theorem_index = []
        ir_dir = Path(self.traced_repo.root_dir) / ".lake/build/ir"

        if not ir_dir.exists():
            self.logger.error(
                f"IR directory not found at {ir_dir}. Cannot verify theorems. "
                "Please ensure the repository was built correctly with 'lake build'."
            )
            return []

        for traced_file in self.traced_repo.traced_files:
            try:
                theorems = traced_file.get_traced_theorems()
                for thm in theorems:
                    ast_path = ir_dir / thm.theorem.file_path.with_suffix(".ast.json")
                    if thm.theorem.full_name and ast_path.exists():
                        theorem_index.append(
                            {
                                "file_path": str(thm.theorem.file_path),
                                "full_name": thm.theorem.full_name,
                            }
                        )
            except Exception as e:
                self.logger.warning(
                    f"Could not load theorems from {traced_file.path}: {e}"
                )
                continue

        with open(index_path, "w") as f:
            json.dump(theorem_index, f)

        self.logger.info(
            f"Saved a new, validated theorem index to {index_path} with {len(theorem_index)} theorems."
        )
        return theorem_index

    def _cleanup_memory(self):
        """Clean up memory, especially CUDA cache."""
        try:
            # Get the actual agent (unwrap DDP if needed)
            agent = self.agent.module if isinstance(self.agent, DDP) else self.agent

            # Clear agent's episode data if it exists
            if hasattr(agent, "episode_rewards") and isinstance(
                agent.episode_rewards, list
            ):
                # Keep only recent rewards
                agent.episode_rewards = agent.episode_rewards[-100:]

            if hasattr(agent, "experience_buffer") and isinstance(
                agent.experience_buffer, list
            ):
                # Clear experience buffer periodically to prevent memory buildup
                if len(agent.experience_buffer) > 1000:
                    agent.experience_buffer = agent.experience_buffer[-500:]

            # CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

        except Exception as e:
            self.logger.warning(f"Memory cleanup failed: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train a Hierarchical Transformer Agent for Lean theorem proving."
    )

    # Repository and Experiment Config
    parser.add_argument(
        "--repo-url",
        type=str,
        default="https://github.com/leanprover-community/mathlib4",
        help="URL of the Lean repository to use.",
    )
    parser.add_argument(
        "--repo-commit",
        type=str,
        default="29dcec074de168ac2bf835a77ef68bbe069194c5",
        help="Commit hash of the repository.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=f"training_run_{int(time.time())}",
        help="A name for the training run for logging purposes.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5000,
        help="Maximum number of training episodes.",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=100,
        help="Maximum steps per episode.",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=200,
        help="How often to run evaluation (in episodes).",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=500,
        help="How often to save a periodic checkpoint (in episodes).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base directory to save logs, checkpoints, and other artifacts.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=50,
        help="Number of theorems to use for each evaluation phase.",
    )
    parser.add_argument(
        "--no-evaluation",
        action="store_true",
        help="If set, disables the evaluation phase. Useful for debug runs.",
    )

    args = parser.parse_args()

    # Configuration
    config = ExperimentConfig(
        experiment_name=args.run_name,
        repo_url=args.repo_url,
        repo_commit=args.repo_commit,
        training=TrainingConfig(
            max_episodes=args.max_episodes,
            max_steps_per_episode=args.max_steps_per_episode,
            eval_frequency=args.eval_frequency,
            eval_episodes=args.eval_episodes,
            save_frequency=args.save_frequency,
            use_experience_replay=True,
            parameter_loss_coefficient=0.05,  # Coefficient for parameter loss
        ),
        model=ModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=4,
        ),
        curriculum=CurriculumConfig(
            use_curriculum=True,
            curriculum_stages=10,
            difficulty_threshold=0.6,
        ),
        distributed=DistributedConfig(use_distributed=False),
    )

    # Initialize trainer
    trainer = HierarchicalTransformerTrainer(
        config, no_evaluation=args.no_evaluation, output_dir=args.output_dir
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
