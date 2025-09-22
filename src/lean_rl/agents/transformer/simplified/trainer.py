"""
HPC-ready LeanDojo training module for simplified transformer agent.
Provides distributed training, checkpoint management, and efficient caching.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from dataclasses import dataclass
import traceback

# LeanDojo integration
from lean_dojo import (
    Dojo,
    Theorem,
    ProofFinished,
    LeanError,
    TacticState,
)
from ..data.repository import RepoManager


# Simplified agent components
from .core import SimplifiedTransformerAgent
from .hpc_config import SimpleHPCConfig


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring and logging."""

    episode: int = 0
    total_proofs: int = 0
    successful_proofs: int = 0
    average_proof_length: float = 0.0
    training_loss: float = 0.0
    learning_rate: float = 0.0
    timestamp: float = 0.0


class LeanEnvironment:
    """Wrapper for LeanDojo environment with proper resource management."""

    def __init__(self, config: SimpleHPCConfig):
        self.config = config
        self.repo_manager = RepoManager(
            repo_url=config.repo_url, repo_commit=config.repo_commit
        )
        self.traced_repo = None
        self.dojo = None
        self.current_theorem = None
        self.setup_completed = False

    def setup(self):
        """Initialize LeanDojo environment - only call once."""
        if self.setup_completed:
            return

        logging.info("Setting up LeanDojo environment...")
        # The RepoManager will handle caching and tracing.
        self.traced_repo = self.repo_manager.get_traced_repo()
        logging.info("LeanDojo environment setup completed.")
        self.setup_completed = True

    def create_dojo(self, theorem: Theorem) -> Dojo:
        """Create a new Dojo instance for a theorem."""
        if not self.setup_completed:
            self.setup()

        try:
            self.current_theorem = theorem
            # Use timeout from config
            self.dojo = Dojo(theorem, timeout=self.config.timeout)
            return self.dojo

        except Exception as e:
            logging.error(f"Failed to create Dojo for theorem {theorem}: {e}")
            raise

    def cleanup(self):
        """Clean up resources."""
        if self.dojo:
            # Dojo cleanup is automatic, just set to None
            self.dojo = None
        self.current_theorem = None


class LeanDojoTrainer:
    """HPC-ready trainer for SimplifiedTransformerAgent with LeanDojo integration."""

    def __init__(self, config: SimpleHPCConfig):
        self.config = config
        self.setup_logging()

        # Initialize components
        self.agent = None
        self.environment = LeanEnvironment(config)
        self.metrics = TrainingMetrics()

        # Training state
        self.global_step = 0
        self.best_success_rate = 0.0
        self.training_active = False

    def setup_logging(self):
        """Configure comprehensive logging for HPC monitoring."""
        # Use the config's setup_logging method
        logger = self.config.setup_logging()

        # Add metrics log file for plotting
        log_dir = self.config.hpc.log_dir or "."
        experiment_name = self.config.hpc.experiment_name or "training"

        self.metrics_log = Path(log_dir) / f"metrics_{experiment_name}.csv"
        if not self.metrics_log.exists():
            try:
                self.metrics_log.parent.mkdir(parents=True, exist_ok=True)
                with open(self.metrics_log, "w") as f:
                    f.write(
                        "episode,total_proofs,successful_proofs,success_rate,avg_proof_length,training_loss,lr,timestamp\n"
                    )
            except (PermissionError, OSError) as e:
                # Fallback to local directory if HPC paths fail
                logging.warning(
                    f"Cannot create log directory {log_dir}, using local directory: {e}"
                )
                self.metrics_log = Path(".") / f"metrics_{experiment_name}.csv"
                if not self.metrics_log.exists():
                    with open(self.metrics_log, "w") as f:
                        f.write(
                            "episode,total_proofs,successful_proofs,success_rate,avg_proof_length,training_loss,lr,timestamp\n"
                        )

        logging.info(f"Training run {experiment_name} starting")
        logging.info(f"Config: {self.config}")

    def initialize_agent(self):
        """Initialize the simplified transformer agent."""
        if self.agent is not None:
            return

        try:
            # Convert HPC config to agent config using correct attribute names
            agent_config = {
                "vocab_size": self.config.vocab_size,
                "d_model": self.config.d_model,
                "n_heads": self.config.n_heads,
                "n_layers": self.config.n_layers,
                "device": self.config.device,
            }

            self.agent = SimplifiedTransformerAgent(**agent_config)

            # Load checkpoint if model save path exists
            if self.config.model_save_path and os.path.exists(
                self.config.model_save_path
            ):
                self.load_checkpoint()

            logging.info(f"Agent initialized on device: {self.config.device}")

        except Exception as e:
            logging.error(f"Failed to initialize agent: {e}")
            raise

    def load_theorems(self) -> List[Theorem]:
        """Load theorems from the traced repository."""
        # Setup environment first
        self.environment.setup()

        if not self.environment.traced_repo:
            logging.warning("No traced repository available.")
            return []

        # Load theorems from traced repo
        traced_theorems = list(self.environment.traced_repo.get_traced_theorems())

        # Convert TracedTheorem to Theorem and limit for HPC efficiency
        theorems = [
            traced_thm.theorem
            for traced_thm in traced_theorems[: self.config.max_theorems]
        ]

        logging.info(
            f"Loaded {len(theorems)} theorems from traced repository (out of {len(traced_theorems)} total)"
        )
        return theorems

    def _get_test_theorems(self) -> List[Theorem]:
        """Get a small set of test theorems for basic functionality."""
        # Return empty list for now - actual theorem loading would need repository setup
        logging.info("Using empty theorem list for testing")
        return []

    def train_episode(self, theorem: Theorem) -> Dict[str, Any]:
        """Train on a single theorem proof attempt."""
        episode_metrics = {
            "theorem_name": str(theorem),
            "proof_successful": False,
            "proof_length": 0,
            "training_loss": 0.0,
            "error": None,
        }

        dojo = None
        try:
            # Create Dojo environment
            dojo = self.environment.create_dojo(theorem)

            # Get initial state using context manager
            with dojo as (_, initial_state):
                if not initial_state:
                    episode_metrics["error"] = "Failed to get initial state"
                    return episode_metrics

                proof_steps = []
                states = []
                actions = []
                rewards = []

                # Proof attempt loop
                max_steps = self.config.max_steps_per_episode
                current_state = initial_state

                for step in range(max_steps):
                    # Only proceed if we have a valid TacticState
                    if not isinstance(current_state, TacticState):
                        episode_metrics["error"] = (
                            f"Invalid state type: {type(current_state)}"
                        )
                        break

                    # Get agent action
                    if not self.agent:
                        episode_metrics["error"] = "Agent not initialized"
                        break

                    action = self.agent.select_action(current_state)
                    proof_steps.append((current_state, action))
                    states.append(current_state)
                    actions.append(action)

                    # Execute action in environment (current_state is guaranteed to be TacticState)
                    result = dojo.run_tac(current_state, action)

                    if isinstance(result, ProofFinished):
                        episode_metrics["proof_successful"] = True
                        episode_metrics["proof_length"] = len(proof_steps)
                        rewards.append(1.0)  # Success reward
                        logging.info(
                            f"Proof completed in {len(proof_steps)} steps: {theorem}"
                        )
                        break

                    elif isinstance(result, LeanError):
                        # Action failed, give negative reward but continue with same state
                        rewards.append(-0.1)
                        continue

                    elif isinstance(result, TacticState):
                        # Progress reward based on goal reduction
                        prev_goals = current_state.num_goals
                        curr_goals = result.num_goals
                        reward = 0.1 if curr_goals < prev_goals else -0.05
                        rewards.append(reward)

                        # Update state
                        current_state = result
                    else:
                        # Handle other result types (like ProofGivenUp)
                        rewards.append(-0.5)
                        break

                # Pad rewards to match actions if needed
                while len(rewards) < len(actions):
                    rewards.append(-0.1)

                # Train agent on proof attempt using the update method
                if states and actions and self.agent:
                    try:
                        loss_dict = self.agent.update(states, actions, rewards)
                        episode_metrics["training_loss"] = loss_dict.get(
                            "total_loss", 0.0
                        )
                    except Exception as train_error:
                        logging.warning(f"Training update failed: {train_error}")
                        episode_metrics["training_loss"] = 0.0

        except Exception as e:
            episode_metrics["error"] = str(e)
            logging.error(f"Episode training failed for {theorem}: {e}")

        finally:
            if dojo:
                # Cleanup handled automatically by Dojo
                pass

        return episode_metrics

    def save_checkpoint(self):
        """Save training checkpoint."""
        try:
            checkpoint_dir = Path(self.config.hpc.checkpoint_dir or ".")
            try:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                # Fallback to local directory
                logging.warning(
                    f"Cannot create checkpoint directory {checkpoint_dir}, using local directory: {e}"
                )
                checkpoint_dir = Path(".")
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"

            checkpoint = {
                "global_step": self.global_step,
                "agent_state_dict": self.agent.state_dict() if self.agent else None,
                "optimizer_state_dict": (
                    self.agent.optimizer.state_dict()
                    if (self.agent and hasattr(self.agent, "optimizer"))
                    else None
                ),
                "metrics": self.metrics,
                "config": self.config,
                "best_success_rate": self.best_success_rate,
            }

            torch.save(checkpoint, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

            # Keep only recent checkpoints
            self.cleanup_old_checkpoints()

        except Exception as e:
            logging.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self):
        """Load the latest checkpoint."""
        try:
            checkpoint_dir = Path(self.config.hpc.checkpoint_dir or ".")
            # Try local directory if HPC directory doesn't exist
            if not checkpoint_dir.exists() and checkpoint_dir != Path("."):
                checkpoint_dir = Path(".")

            if not checkpoint_dir.exists():
                logging.info("No checkpoint directory found")
                return

            checkpoints = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
            if not checkpoints:
                logging.info("No checkpoints found")
                return

            latest_checkpoint = max(
                checkpoints, key=lambda x: int(x.stem.split("_")[-1])
            )

            checkpoint = torch.load(latest_checkpoint, map_location=self.config.device)

            self.global_step = checkpoint["global_step"]
            self.metrics = checkpoint.get("metrics", TrainingMetrics())
            self.best_success_rate = checkpoint.get("best_success_rate", 0.0)

            if self.agent and checkpoint["agent_state_dict"]:
                self.agent.load_state_dict(checkpoint["agent_state_dict"])
                if hasattr(self.agent, "optimizer") and checkpoint.get(
                    "optimizer_state_dict"
                ):
                    self.agent.optimizer.load_state_dict(
                        checkpoint["optimizer_state_dict"]
                    )

            logging.info(f"Checkpoint loaded: {latest_checkpoint}")

        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")

    def cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        try:
            checkpoint_dir = Path(self.config.hpc.checkpoint_dir or ".")
            # Try local directory if HPC directory doesn't exist
            if not checkpoint_dir.exists() and checkpoint_dir != Path("."):
                checkpoint_dir = Path(".")

            checkpoints = sorted(
                checkpoint_dir.glob("checkpoint_step_*.pt"),
                key=lambda x: int(x.stem.split("_")[-1]),
            )

            # Keep only the last 5 checkpoints
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
                logging.debug(f"Removed old checkpoint: {old_checkpoint}")

        except Exception as e:
            logging.error(f"Failed to cleanup checkpoints: {e}")

    def log_metrics(self):
        """Log current training metrics."""
        success_rate = (
            self.metrics.successful_proofs / max(1, self.metrics.total_proofs)
        ) * 100

        # Write to CSV for plotting
        with open(self.metrics_log, "a") as f:
            f.write(
                f"{self.metrics.episode},{self.metrics.total_proofs},"
                f"{self.metrics.successful_proofs},{success_rate:.2f},"
                f"{self.metrics.average_proof_length:.2f},{self.metrics.training_loss:.6f},"
                f"{self.metrics.learning_rate:.8f},{time.time()}\n"
            )

        # Log to console
        logging.info(
            f"Episode {self.metrics.episode}: "
            f"Success Rate: {success_rate:.2f}% "
            f"({self.metrics.successful_proofs}/{self.metrics.total_proofs}), "
            f"Avg Length: {self.metrics.average_proof_length:.2f}, "
            f"Loss: {self.metrics.training_loss:.6f}"
        )

    def train(self):
        """Main training loop."""
        try:
            self.training_active = True
            logging.info("Starting training...")

            # Initialize components
            self.initialize_agent()
            theorems = self.load_theorems()

            if not theorems:
                logging.warning("No theorems loaded, running test training loop")
                # Run a test loop without actual theorems
                self._run_test_training()
                return

            # Training loop
            log_interval = getattr(self.config, "log_interval", 10)
            checkpoint_interval = self.config.checkpoint_frequency

            for episode in range(self.config.max_episodes):
                self.metrics.episode = episode

                # Select theorem (cycle through available theorems)
                theorem = theorems[episode % len(theorems)]

                # Train on theorem
                episode_result = self.train_episode(theorem)

                # Update metrics
                self.metrics.total_proofs += 1
                if episode_result["proof_successful"]:
                    self.metrics.successful_proofs += 1

                if episode_result["proof_length"] > 0:
                    # Update average proof length
                    self.metrics.average_proof_length = (
                        self.metrics.average_proof_length
                        * (self.metrics.total_proofs - 1)
                        + episode_result["proof_length"]
                    ) / self.metrics.total_proofs

                self.metrics.training_loss = episode_result["training_loss"]
                self.metrics.timestamp = time.time()

                # Log progress
                if episode % log_interval == 0:
                    self.log_metrics()

                # Save checkpoints
                if episode % checkpoint_interval == 0:
                    self.save_checkpoint()

                # Check for early stopping
                current_success_rate = self.metrics.successful_proofs / max(
                    1, self.metrics.total_proofs
                )
                if current_success_rate > self.best_success_rate:
                    self.best_success_rate = current_success_rate

                self.global_step += 1

            # Final checkpoint
            self.save_checkpoint()
            logging.info(
                f"Training completed. Final success rate: {self.best_success_rate:.2%}"
            )

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint()

        except Exception as e:
            logging.error(f"Training failed: {e}")
            logging.error(traceback.format_exc())

        finally:
            self.training_active = False
            self.environment.cleanup()

    def _run_test_training(self):
        """Run a test training loop for validation."""
        logging.info("Running test training loop")

        for episode in range(min(10, self.config.max_episodes)):
            self.metrics.episode = episode

            # Simulate training episode
            episode_result = {
                "proof_successful": episode % 3 == 0,  # Success every 3rd episode
                "proof_length": 5 + episode,
                "training_loss": 0.1 + 0.01 * episode,
                "error": None,
            }

            # Update metrics
            self.metrics.total_proofs += 1
            if episode_result["proof_successful"]:
                self.metrics.successful_proofs += 1

            self.metrics.training_loss = episode_result["training_loss"]
            self.metrics.timestamp = time.time()

            if episode % 2 == 0:
                self.log_metrics()

            self.global_step += 1

        self.save_checkpoint()
        logging.info("Test training completed")


def create_leandojo_trainer(
    config_dict: Optional[Dict[str, Any]] = None,
) -> LeanDojoTrainer:
    """Factory function to create a LeanDojo trainer with HPC configuration."""

    # Use provided config or create default HPC config
    if config_dict:
        config = SimpleHPCConfig(**config_dict)
    else:
        # Auto-detect HPC environment
        from .hpc_config import create_hpc_config

        config = create_hpc_config()

    return LeanDojoTrainer(config)


def main():
    """Main entry point for HPC training."""
    try:
        # Create trainer with auto-detected HPC config
        trainer = create_leandojo_trainer()

        # Start training
        trainer.train()

    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
