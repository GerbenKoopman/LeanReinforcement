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

from .repository import RepoManager
from lean_dojo import TacticState, Theorem
from lean_dojo.data_extraction.traced_data import TracedRepo, TracedTheorem

from .agent import (
    HierarchicalTransformerAgent,
    HierarchicalAction,
)
from .hierarchy import (
    HierarchyLevel,
)
from .config import (
    TrainingConfig,
    ExperimentConfig,
    ModelConfig,
    CurriculumConfig,
    DistributedConfig,
)
from ...environment import LeanEnvironment


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


class ExperienceReplayBuffer:
    """Enhanced experience replay buffer with proper state encoding."""

    def __init__(self, capacity: int, device: torch.device, agent=None):
        self.capacity = capacity
        self.device = device
        self.buffer = deque(maxlen=capacity)
        self.agent = agent  # Reference to agent for encoding

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def push(
        self,
        state: TacticState,
        action: HierarchicalAction,
        reward: float,
        next_state: Optional[TacticState],
        done: bool,
    ):
        """Add experience to the buffer."""
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        }
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Optional[Dict[str, Any]]:
        """Sample a batch and return properly formatted tensors."""
        if len(self.buffer) < batch_size:
            return None

        experiences = random.sample(self.buffer, batch_size)
        return self._prepare_batch(experiences)

    def _prepare_batch(self, experiences: List[Dict]) -> Dict[str, Any]:
        """Convert experiences to batched tensors."""
        states = [exp["state"] for exp in experiences]
        actions = [exp["action"] for exp in experiences]
        rewards = [exp["reward"] for exp in experiences]
        next_states = [exp["next_state"] for exp in experiences]
        dones = [exp["done"] for exp in experiences]

        # Encode states on-the-fly
        encoded_states = self._encode_state_batch(states)
        encoded_next_states = self._encode_state_batch(
            [s for s in next_states if s is not None]
        )

        # Create batch dictionary
        batch = {
            "states": states,
            "actions": actions,
            "rewards": torch.tensor(rewards, dtype=torch.float32).to(self.device),
            "next_states": next_states,
            "dones": torch.tensor(dones, dtype=torch.bool).to(self.device),
            "encoded_states": encoded_states,
            "encoded_next_states": encoded_next_states,
        }

        return batch

    def _encode_state_batch(
        self, states: List[Optional[TacticState]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Encode a batch of states, handling None values."""
        agent_module = self.agent.module if isinstance(self.agent, DDP) else self.agent
        if not agent_module:
            raise RuntimeError("Agent reference not set in ExperienceReplayBuffer")

        encoded_batch = []
        for state in states:
            if state is not None:
                encoded_batch.append(agent_module.encode_state(state))
        return encoded_batch


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
        """Get theorem index from cache or create it if it doesn't exist."""
        cache_dir = os.getenv("CACHE_DIR")
        if not cache_dir:
            self.logger.error("CACHE_DIR environment variable not set!")
            return []

        index_path = Path(cache_dir) / "theorem_index.json"

        if index_path.exists():
            self.logger.info(f"Loading theorem index from {index_path}")
            with open(index_path, "r") as f:
                return json.load(f)

        self.logger.info("Creating theorem index (one-time operation)...")
        theorem_index = []
        for traced_file in self.traced_repo.traced_files:
            try:
                theorems = traced_file.get_traced_theorems()
                for thm in theorems:
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

        self.logger.info(f"Saved theorem index to {index_path}")
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

    def __init__(self, config: ExperimentConfig):
        self.config = config
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

    def _setup_logging(self):
        """Setup logging configuration."""

        # Get SCRATCH_SHARED from environment
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
        self.env = LeanEnvironment(
            repo=self.repo_manager.repo,
            timeout=self.config.training.timeout,
            max_steps=self.config.training.max_steps_per_episode,
            reward_scheme=self.config.training.reward_scheme,
        )

    def _setup_evaluation(self):
        """Setup evaluation and logging."""

        # Get SCRATCH_SHARED from environment
        scratch_dir = os.getenv("SCRATCH_SHARED", ".")

        # Tensorboard writer
        if self.config.distributed.rank == 0:  # Only main process writes logs
            log_dir = (
                Path(scratch_dir)
                / "tensorboard_logs"
                / f"exp_{self.config.experiment_name}_{int(time.time())}"
            )
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        # Checkpoint directory
        self.checkpoint_dir = (
            Path(scratch_dir) / "checkpoints" / f"exp_{self.config.experiment_name}"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_agent_parameters(self):
        """Get all parameters from agent submodules."""
        parameters = []
        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent

        if hasattr(agent, "hierarchical_policy"):
            parameters.extend(agent.hierarchical_policy.parameters())
        if hasattr(agent, "tactic_pointer"):
            parameters.extend(agent.tactic_pointer.parameters())
        if hasattr(agent, "parameter_generator"):
            parameters.extend(agent.parameter_generator.parameters())
        if hasattr(agent, "parameter_pointer"):
            parameters.extend(agent.parameter_pointer.parameters())
        return parameters

    def _get_agent_state_dict(self):
        """Get state dict from all agent submodules."""
        state_dict = {}
        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent

        if hasattr(agent, "hierarchical_policy"):
            state_dict["hierarchical_policy"] = agent.hierarchical_policy.state_dict()
        if hasattr(agent, "tactic_pointer"):
            state_dict["tactic_pointer"] = agent.tactic_pointer.state_dict()
        if hasattr(agent, "parameter_generator"):
            state_dict["parameter_generator"] = agent.parameter_generator.state_dict()
        if hasattr(agent, "parameter_pointer"):
            state_dict["parameter_pointer"] = agent.parameter_pointer.state_dict()
        return state_dict

    def _get_target_agent_state_dict(self):
        """Get state dict from all target agent submodules."""
        state_dict = {}
        if hasattr(self.target_agent, "hierarchical_policy"):
            state_dict["hierarchical_policy"] = (
                self.target_agent.hierarchical_policy.state_dict()
            )
        if hasattr(self.target_agent, "tactic_pointer"):
            state_dict["tactic_pointer"] = self.target_agent.tactic_pointer.state_dict()
        if hasattr(self.target_agent, "parameter_generator"):
            state_dict["parameter_generator"] = (
                self.target_agent.parameter_generator.state_dict()
            )
        if hasattr(self.target_agent, "parameter_pointer"):
            state_dict["parameter_pointer"] = (
                self.target_agent.parameter_pointer.state_dict()
            )
        return state_dict

    def _load_agent_state_dict(self, agent, state_dict):
        """Load state dict into agent submodules."""
        if (
            hasattr(agent, "hierarchical_policy")
            and "hierarchical_policy" in state_dict
        ):
            agent.hierarchical_policy.load_state_dict(state_dict["hierarchical_policy"])
        if hasattr(agent, "tactic_pointer") and "tactic_pointer" in state_dict:
            agent.tactic_pointer.load_state_dict(state_dict["tactic_pointer"])
        if (
            hasattr(agent, "parameter_generator")
            and "parameter_generator" in state_dict
        ):
            agent.parameter_generator.load_state_dict(state_dict["parameter_generator"])
        if hasattr(agent, "parameter_pointer") and "parameter_pointer" in state_dict:
            agent.parameter_pointer.load_state_dict(state_dict["parameter_pointer"])

    def _set_agent_mode(self, agent, train_mode):
        """Set train/eval mode for agent submodules."""
        if hasattr(agent, "hierarchical_policy"):
            agent.hierarchical_policy.train(train_mode)
        if hasattr(agent, "tactic_pointer"):
            agent.tactic_pointer.train(train_mode)
        if hasattr(agent, "parameter_generator"):
            agent.parameter_generator.train(train_mode)
        if hasattr(agent, "parameter_pointer"):
            agent.parameter_pointer.train(train_mode)

    def _update_target_network(self):
        """Update target network with current agent weights."""
        try:
            agent_state_dict = self._get_agent_state_dict()
            self._load_agent_state_dict(self.target_agent, agent_state_dict)
            self.logger.debug("Target network updated successfully")
        except Exception as e:
            self.logger.warning(f"Target network update failed: {e}")

    def _validate_config(self):
        """Validate configuration to prevent runtime errors."""
        try:
            # Check required fields
            if not hasattr(self.config, "training"):
                raise ValueError("Missing training configuration")

            if not hasattr(self.config, "model"):
                raise ValueError("Missing model configuration")

            # Check distributed configuration
            if hasattr(self.config, "distributed") and self.config.distributed:
                if self.config.distributed.use_distributed:
                    if not hasattr(self.config.distributed, "rank"):
                        self.config.distributed.rank = 0
                        self.logger.warning("No rank specified, defaulting to 0")

                    if not hasattr(self.config.distributed, "world_size"):
                        self.config.distributed.world_size = 1
                        self.logger.warning("No world_size specified, defaulting to 1")
            else:
                # Create default distributed config
                self.config.distributed = DistributedConfig(
                    use_distributed=False, rank=0, world_size=1
                )

            # Validate training parameters
            if self.config.training.max_episodes <= 0:
                raise ValueError("max_episodes must be positive")

            if self.config.training.learning_rate <= 0:
                raise ValueError("learning_rate must be positive")

            # Set default values for missing optional parameters
            if not hasattr(self.config.training, "target_update_frequency"):
                self.config.training.target_update_frequency = 100

            if not hasattr(self.config.training, "replay_start_size"):
                self.config.training.replay_start_size = 1000

            if not hasattr(self.config.training, "gradient_clip_norm"):
                self.config.training.gradient_clip_norm = 0.5

            self.logger.info("Configuration validation passed")

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise

    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")

        episode = 0
        best_success_rate = 0.0
        recent_rewards = deque(maxlen=100)
        recent_successes = deque(maxlen=100)

        while episode < self.config.training.max_episodes:
            # Get theorems for current curriculum stage
            if self.curriculum:
                available_theorems = self.curriculum.get_current_theorems()
            else:
                # Use all available theorems
                available_theorems = self._get_all_theorems()

            if not available_theorems:
                self.logger.warning("No theorems available for training")
                break

            # Sample random theorem
            theorem = random.choice(available_theorems)

            # Run episode
            episode_reward, episode_length, success = self._run_episode(theorem)

            # Update metrics
            self.metrics.episode = episode
            self.metrics.total_steps += episode_length
            if self.metrics.episode_rewards is not None:
                self.metrics.episode_rewards.append(episode_reward)
            if self.metrics.episode_lengths is not None:
                self.metrics.episode_lengths.append(episode_length)

            recent_rewards.append(episode_reward)
            recent_successes.append(1.0 if success else 0.0)

            # Update curriculum if needed
            if self.curriculum and len(recent_successes) >= 50:
                recent_success_rate = float(np.mean(recent_successes))
                if self.curriculum.should_advance_stage(recent_success_rate):
                    self.curriculum.advance_stage()
                    recent_successes.clear()  # Reset for new stage

            # Experience replay training
            if (
                self.config.training.use_experience_replay
                and len(self.replay_buffer) >= self.config.training.replay_start_size
            ):
                self._train_from_replay()

            # Logging
            if episode % 100 == 0:  # Log every 100 episodes
                self._log_progress(episode, recent_rewards, recent_successes)

            # Evaluation
            if episode % self.config.training.eval_frequency == 0:
                success_rate = self._evaluate()
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    self._save_checkpoint(episode, "best")

                # Update learning rate
                self.scheduler.step(success_rate)

            # Checkpointing
            if episode % self.config.training.save_frequency == 0:
                self._save_checkpoint(episode, "regular")

            # Update target network
            if episode % self.config.training.target_update_frequency == 0:
                self._update_target_network()

            # Memory cleanup every 20 episodes to prevent memory leaks
            if episode % 20 == 0:
                self._cleanup_memory()

            episode += 1

        self.logger.info("Training completed!")
        self._save_checkpoint(episode, "final")

        if self.writer:
            self.writer.close()

    def _run_episode(self, theorem) -> Tuple[float, int, bool]:
        """Run a single training episode."""
        try:
            # Get the actual agent (unwrap DDP if needed)
            agent = self.agent.module if isinstance(self.agent, DDP) else self.agent

            # Reset environment
            state = self.env.reset(theorem.theorem)
            agent.reset()

            episode_reward = 0.0
            episode_length = 0
            success = False

            episode_experiences = []

            while True:
                # Select action
                if state is not None:
                    action_str = agent.select_action(state)
                    if action_str is None:
                        break

                    # Take step
                    step_result = self.env.step(action_str)

                    # Update agent
                    agent.update(step_result)

                    # Store experience for replay
                    if self.config.training.use_experience_replay:
                        # Create hierarchical action object (simplified)
                        hierarchical_action = HierarchicalAction(
                            strategic_action="direct_proof",  # Simplified
                            tactic_family="apply_family",  # Simplified
                            specific_tactic=action_str,
                            parameters=[],
                            confidence=0.8,
                        )

                        episode_experiences.append(
                            (
                                state,
                                hierarchical_action,
                                step_result.reward,
                                step_result.state,
                                step_result.done,
                            )
                        )

                    episode_reward += step_result.reward
                    episode_length += 1

                    if step_result.done:
                        success = step_result.action_result == "proof_finished"
                        break

                    state = step_result.state

                    if episode_length >= self.config.training.max_steps_per_episode:
                        break
                else:
                    break

            # Add experiences to replay buffer
            if self.config.training.use_experience_replay:
                for experience in episode_experiences:
                    self.replay_buffer.push(*experience)

            return episode_reward, episode_length, success

        except Exception as e:
            self.logger.warning(f"Episode failed with error: {e}")
            return 0.0, 0, False

    def _train_from_replay(self):
        """Train the model using experience replay."""
        if len(self.replay_buffer) < self.config.training.replay_batch_size:
            return

        # Sample batch
        batch = self.replay_buffer.sample(self.config.training.replay_batch_size)

        # Extract batch components
        actions = batch["actions"]
        rewards_tensor = batch["rewards_tensor"]
        dones_tensor = batch["dones_tensor"]
        encoded_states = batch["encoded_states"]
        strategic_actions = batch["strategic_actions"]
        tactic_families = batch["tactic_families"]

        try:
            # Compute current Q-values for each hierarchy level
            current_q_values = self._compute_hierarchical_q_values(
                encoded_states, strategic_actions, tactic_families
            )

            # Compute target Q-values using target network
            target_q_values = self._compute_target_q_values(
                batch["encoded_next_states"], rewards_tensor, dones_tensor
            )

            # Compute losses for each hierarchy level
            strategic_loss = self._compute_strategic_loss(
                current_q_values["strategic"], target_q_values, strategic_actions
            )

            tactical_loss = self._compute_tactical_loss(
                current_q_values["tactical"], target_q_values, tactic_families
            )

            parameter_loss = self._compute_parameter_loss(
                current_q_values["parameters"], actions
            )

            # Total loss with weights
            total_loss = (
                0.4 * strategic_loss + 0.4 * tactical_loss + 0.2 * parameter_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self._get_agent_parameters(), self.config.training.gradient_clip_norm
            )

            self.optimizer.step()

            # Log loss
            if self.metrics.loss_history is not None:
                self.metrics.loss_history.append(total_loss.item())

        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            # Track training failures
            if not hasattr(self, "training_failures"):
                self.training_failures = 0
            self.training_failures += 1

            # If too many consecutive failures, stop training
            if self.training_failures > 50:
                self.logger.error("Too many training failures, stopping training")
                raise RuntimeError(
                    f"Training failed with {self.training_failures} consecutive failures"
                )

    def _compute_hierarchical_q_values(
        self, encoded_states: List, strategic_actions: List, tactic_families: List
    ) -> Dict[str, torch.Tensor]:
        """Compute Q-values for all hierarchy levels."""
        batch_size = len(encoded_states)

        # Get the actual agent (unwrap DDP if needed)
        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent

        # Initialize outputs
        strategic_q_values = []
        tactical_q_values = []
        parameter_q_values = []

        for i in range(batch_size):
            if encoded_states[i] is None:
                # Use dummy values for None states
                strategic_q_values.append(torch.zeros(1, device=self.device))
                tactical_q_values.append(torch.zeros(1, device=self.device))
                parameter_q_values.append(torch.zeros(1, device=self.device))
                continue

            try:
                # Strategic level
                strategic_output = agent.hierarchical_forward(
                    {k: v.unsqueeze(0) for k, v in encoded_states[i].items()},
                    HierarchyLevel.STRATEGIC,
                )
                strategic_q_values.append(
                    strategic_output.get("value", torch.zeros(1, device=self.device))
                )

                # Tactical level
                tactical_output = agent.hierarchical_forward(
                    {k: v.unsqueeze(0) for k, v in encoded_states[i].items()},
                    HierarchyLevel.TACTICAL,
                    strategic_action=strategic_actions[i],
                )
                tactical_q_values.append(
                    tactical_output.get("value", torch.zeros(1, device=self.device))
                )

                # Parameter level (simplified)
                parameter_q_values.append(torch.zeros(1, device=self.device))

            except Exception as e:
                # Fallback for failed forward passes
                strategic_q_values.append(torch.zeros(1, device=self.device))
                tactical_q_values.append(torch.zeros(1, device=self.device))
                parameter_q_values.append(torch.zeros(1, device=self.device))

        return {
            "strategic": torch.cat(strategic_q_values),
            "tactical": torch.cat(tactical_q_values),
            "parameters": torch.cat(parameter_q_values),
        }

    def _compute_target_q_values(
        self, encoded_next_states: List, rewards: torch.Tensor, dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute target Q-values using target network."""
        batch_size = len(encoded_next_states)
        target_values = []

        with torch.no_grad():
            for i in range(batch_size):
                if encoded_next_states[i] is None or dones[i]:
                    # Terminal state or no next state
                    target_values.append(rewards[i].unsqueeze(0))
                else:
                    try:
                        # Use target network to compute next state value
                        next_output = self.target_agent.hierarchical_forward(
                            {
                                k: v.unsqueeze(0)
                                for k, v in encoded_next_states[i].items()
                            },
                            HierarchyLevel.STRATEGIC,
                        )
                        next_value = next_output.get(
                            "value", torch.zeros(1, device=self.device)
                        )

                        # Q-learning update: r + gamma * max_a Q(s', a)
                        target_value = rewards[i] + 0.99 * next_value.squeeze()
                        target_values.append(target_value.unsqueeze(0))
                    except Exception:
                        # Fallback to reward only
                        target_values.append(rewards[i].unsqueeze(0))

        return torch.cat(target_values)

    def _compute_strategic_loss(
        self, current_q: torch.Tensor, target_q: torch.Tensor, strategic_actions: List
    ) -> torch.Tensor:
        """Compute loss for strategic level."""
        # Simple MSE loss between current and target Q-values
        return F.mse_loss(current_q, target_q.detach())

    def _compute_tactical_loss(
        self, current_q: torch.Tensor, target_q: torch.Tensor, tactic_families: List
    ) -> torch.Tensor:
        """Compute loss for tactical level."""
        # Simple MSE loss between current and target Q-values
        return F.mse_loss(current_q, target_q.detach())

    def _compute_parameter_loss(
        self, current_q: torch.Tensor, actions: List
    ) -> torch.Tensor:
        """Compute loss for parameter generation."""
        # Simplified parameter loss - in practice, would use sequence loss
        dummy_target = torch.zeros_like(current_q)
        return F.mse_loss(current_q, dummy_target)

    def _evaluate(self) -> float:
        """Evaluate the current model."""
        self.logger.info("Running evaluation...")

        # Get evaluation theorems
        if self.curriculum:
            eval_theorems = self.curriculum.get_current_theorems()
        else:
            eval_theorems = self._get_all_theorems()

        if not eval_theorems:
            return 0.0

        # Sample evaluation theorems
        eval_theorems = random.sample(
            eval_theorems, min(self.config.training.eval_episodes, len(eval_theorems))
        )

        successes = 0
        total_reward = 0.0

        # Get the actual agent (unwrap DDP if needed)
        agent = self.agent.module if isinstance(self.agent, DDP) else self.agent

        # Set model to evaluation mode
        self._set_agent_mode(agent, False)  # Set to eval mode

        with torch.no_grad():
            for theorem in eval_theorems:
                try:
                    state = self.env.reset(theorem.theorem)
                    agent.reset()

                    episode_reward = 0.0
                    steps = 0

                    while steps < self.config.training.max_steps_per_episode:
                        if state is not None:
                            action_str = agent.select_action(state)
                            if action_str is None:
                                break

                            step_result = self.env.step(action_str)
                            episode_reward += step_result.reward
                            steps += 1

                            if step_result.done:
                                if step_result.action_result == "proof_finished":
                                    successes += 1
                                break

                            state = step_result.state
                        else:
                            break

                    total_reward += episode_reward

                except Exception as e:
                    self.logger.warning(f"Evaluation episode failed: {e}")
                    continue

        # Set model back to training mode
        self._set_agent_mode(agent, True)  # Set to train mode

        success_rate = successes / len(eval_theorems)
        avg_reward = total_reward / len(eval_theorems)

        self.logger.info(
            f"Evaluation: Success rate: {success_rate:.3f}, Avg reward: {avg_reward:.3f}"
        )

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar(
                "eval/success_rate", success_rate, self.metrics.episode
            )
            self.writer.add_scalar("eval/avg_reward", avg_reward, self.metrics.episode)

        return success_rate

    def _log_progress(
        self, episode: int, recent_rewards: deque, recent_successes: deque
    ):
        """Log training progress."""
        avg_reward = np.mean(recent_rewards) if recent_rewards else 0.0
        success_rate = np.mean(recent_successes) if recent_successes else 0.0

        self.logger.info(
            f"Episode {episode}: Avg reward: {avg_reward:.3f}, "
            f"Success rate: {success_rate:.3f}, "
            f"Curriculum stage: {self.curriculum.current_stage if self.curriculum else 'N/A'}"
        )

        # Tensorboard logging
        if self.writer:
            self.writer.add_scalar("train/avg_reward", avg_reward, episode)
            self.writer.add_scalar("train/success_rate", success_rate, episode)
            self.writer.add_scalar(
                "train/episode_length",
                (
                    np.mean(self.metrics.episode_lengths[-100:])
                    if self.metrics.episode_lengths
                    else 0
                ),
                episode,
            )

            if self.curriculum:
                self.writer.add_scalar(
                    "curriculum/stage", self.curriculum.current_stage, episode
                )

            if self.metrics.loss_history:
                self.writer.add_scalar(
                    "train/loss", np.mean(self.metrics.loss_history[-100:]), episode
                )

    def _save_checkpoint(self, episode: int, checkpoint_type: str):
        """Save model checkpoint to SCRATCH_SHARED."""
        if self.config.distributed.rank != 0:  # Only main process saves
            return

        # Get SCRATCH_SHARED from environment
        scratch_dir = os.getenv("SCRATCH_SHARED", ".")
        checkpoint_dir = (
            Path(scratch_dir) / "checkpoints" / f"exp_{self.config.experiment_name}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{checkpoint_type}_{episode}.pt"

        # Get state dict from potentially wrapped model
        if isinstance(self.agent, DDP):
            agent_state_dict = self.agent.module.state_dict()
        else:
            agent_state_dict = self._get_agent_state_dict()

        checkpoint = {
            "episode": episode,
            "agent_state_dict": agent_state_dict,
            "target_agent_state_dict": self._get_target_agent_state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": asdict(self.config),
            "metrics": asdict(self.metrics),
            "curriculum_stage": self.curriculum.current_stage if self.curriculum else 0,
        }

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        if isinstance(self.agent, DDP):
            self.agent.module.load_state_dict(checkpoint["agent_state_dict"])
        else:
            self._load_agent_state_dict(self.agent, checkpoint["agent_state_dict"])

        self._load_agent_state_dict(
            self.target_agent, checkpoint["target_agent_state_dict"]
        )
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Load metrics
        metrics_dict = checkpoint.get("metrics", {})
        self.metrics = TrainingMetrics(**metrics_dict)

        # Load curriculum state
        if self.curriculum and "curriculum_stage" in checkpoint:
            self.curriculum.current_stage = checkpoint["curriculum_stage"]

        self.logger.info(f"Loaded checkpoint from episode {checkpoint['episode']}")
        return checkpoint["episode"]

    def _get_all_theorems(self) -> List:
        """Get all available theorems from traced repository."""
        all_theorems = []

        for traced_file in self.traced_repo.traced_files:
            try:
                theorems = traced_file.get_traced_theorems()
                all_theorems.extend(theorems)
            except Exception as e:
                self.logger.warning(f"Failed to load theorems from {traced_file}: {e}")
                continue

        return all_theorems

    def _cleanup_memory(self):
        """Clean up memory between episodes to prevent memory leaks."""
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
    # Configuration
    config = ExperimentConfig(
        experiment_name="hierarchical_transformer_training",
        training=TrainingConfig(
            max_episodes=5000,
            eval_frequency=200,
            save_frequency=500,
            use_experience_replay=True,
        ),
        model=ModelConfig(
            d_model=256,  # Smaller for faster training
            n_heads=4,
            n_layers=4,
        ),
        curriculum=CurriculumConfig(
            use_curriculum=True,
        ),
    )

    # Initialize trainer
    trainer = HierarchicalTransformerTrainer(config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
