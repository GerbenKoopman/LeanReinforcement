"""
Main script for agent training and guided-rollout mcts dataset creation. Will
implement tactic generation training in the future.
"""

import argparse
from loguru import logger
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import gc
import os
from pathlib import Path
from dotenv import load_dotenv
import wandb
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from lean_dojo import DojoInitError
from ReProver.common import Corpus, Pos

from src.utilities.dataloader import LeanDataLoader
from src.utilities.gym import LeanDojoEnv
from src.utilities.checkpoint import (
    get_checkpoint_dir,
    save_training_metadata,
    cleanup_old_checkpoints,
)
from src.utilities.analyze_training_data import (
    analyze_value_data,
    print_training_stats,
    save_training_data,
)
from src.agent.runner import AgentRunner
from src.agent.mcts import MCTS_AlphaZero, MCTS_GuidedRollout
from src.agent.transformer import Transformer
from src.agent.value_head import ValueHead

# Load environment variables from .env file
load_dotenv()

# --- Custom Datasets for Training ---


class ValueHeadDataset(Dataset):
    """Dataset for state -> value_target."""

    def __init__(self, data: list):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "state": item["state"],
            "value_target": item["value_target"],
        }


class PolicyHeadDataset(Dataset):
    """Dataset for (state, premises) -> tactic_target."""

    def __init__(self, data: dict):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Training Functions ---


def train_value_head(
    value_head: ValueHead,
    data_buffer: list,
    epochs: int = 1,
    batch_size: int = 32,
    use_wandb: bool = True,
):
    """
    Trains the value head on collected data.
    """
    if not data_buffer:
        logger.warning("Value Head training skipped: No data provided.")
        return

    # Log training data statistics
    value_targets = [item["value_target"] for item in data_buffer]
    avg_target = sum(value_targets) / len(value_targets)
    positive_samples = sum(1 for v in value_targets if v > 0)
    negative_samples = sum(1 for v in value_targets if v < 0)

    logger.info(f"Training Value Head on {len(data_buffer)} samples...")
    logger.info(
        f"  Data distribution: {positive_samples} positive, {negative_samples} negative"
    )
    logger.info(f"  Average target value: {avg_target:.4f}")

    # Log MCTS statistics if available
    avg_mcts = None
    if "mcts_value" in data_buffer[0]:
        mcts_values = [item["mcts_value"] for item in data_buffer]
        avg_mcts = sum(mcts_values) / len(mcts_values)
        logger.info(f"  Average MCTS value estimate: {avg_mcts:.4f}")

    if use_wandb:
        wandb.log(
            {
                "value_head/avg_target": avg_target,
                "value_head/positive_samples": positive_samples,
                "value_head/negative_samples": negative_samples,
                "value_head/avg_mcts_value": avg_mcts,
            }
        )

    value_head.train()  # Set model to training mode

    dataset = ValueHeadDataset(data_buffer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.AdamW(value_head.value_head.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            states = batch["state"]
            value_targets = batch["value_target"].to(
                dtype=torch.float32,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Get features from the frozen encoder
            features = value_head.encode_states(states)

            # Get value prediction from the trainable head
            value_preds = value_head.value_head(features).squeeze()

            loss = loss_fn(value_preds, value_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Value Head Epoch {epoch+1}/{epochs}, Avg. Loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({"value_head/avg_loss": avg_loss})

    value_head.eval()  # Set back to evaluation mode

    # Clear GPU memory after training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --- Checkpoint Management ---


def save_checkpoint(
    value_head: ValueHead,
    epoch: int,
    checkpoint_dir: Path,
    args,
    prefix: str = "value_head",
):
    """
    Save a checkpoint for the value head with metadata.

    Args:
        value_head: The ValueHead model to save
        epoch: Current epoch number
        checkpoint_dir: Directory to save checkpoints
        args: Training arguments
        prefix: Prefix for checkpoint filename
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save the latest checkpoint
    latest_filename = f"{prefix}_latest.pth"
    value_head.save_checkpoint(str(checkpoint_dir), latest_filename)

    # Save epoch-specific checkpoint
    epoch_filename = f"{prefix}_epoch_{epoch}.pth"
    value_head.save_checkpoint(str(checkpoint_dir), epoch_filename)

    # Save training metadata
    metadata = {
        "data_type": args.data_type,
        "mcts_type": args.mcts_type,
        "num_iterations": args.num_iterations,
        "max_steps": args.max_steps,
        "train_epochs": args.train_epochs,
        "use_final_reward": args.use_final_reward,
    }
    save_training_metadata(checkpoint_dir, epoch, metadata)

    # Clean up old checkpoints (keep last 5)
    cleanup_old_checkpoints(checkpoint_dir, prefix, keep_last_n=5)

    logger.info(f"Saved checkpoints: {latest_filename} and {epoch_filename}")


def load_checkpoint(
    value_head: ValueHead, checkpoint_dir: Path, prefix: str = "value_head"
) -> int:
    """
    Load the latest checkpoint if it exists.

    Args:
        value_head: The ValueHead model to load into
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix for checkpoint filename

    Returns:
        The epoch number of the loaded checkpoint, or 0 if no checkpoint found
    """
    latest_filename = f"{prefix}_latest.pth"
    latest_path = checkpoint_dir / latest_filename

    if latest_path.exists():
        try:
            value_head.load_checkpoint(str(checkpoint_dir), latest_filename)

            # Try to determine the epoch from other checkpoints
            epoch_checkpoints = sorted(checkpoint_dir.glob(f"{prefix}_epoch_*.pth"))
            if epoch_checkpoints:
                # Extract epoch number from the last checkpoint
                last_checkpoint = epoch_checkpoints[-1]
                epoch_str = last_checkpoint.stem.split("_")[-1]
                try:
                    return int(epoch_str)
                except ValueError:
                    logger.warning(f"Could not parse epoch from {last_checkpoint}")
                    return 0
            return 0
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {latest_path}: {e}")
            return 0
    else:
        logger.info(f"No checkpoint found at {latest_path}, starting from scratch")
        return 0


# --- Main Loop ---


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )


def process_theorem(
    thm_data,
    corpus,
    dataloader,
    transformer,
    value_head,
    args,
):
    """
    Process a single theorem: initialize env, run agent, collect data.
    """
    theorem = dataloader.extract_theorem(thm_data)
    if not theorem:
        return []

    theorem_pos = Pos(*thm_data["start"])
    if not theorem_pos:
        return []

    try:
        env = LeanDojoEnv(corpus, theorem, theorem_pos)
    except DojoInitError as e:
        logger.error(
            f"Failed to initialize environment for theorem {theorem.full_name}: {e}"
        )
        return []
    except Exception as e:
        logger.error(
            f"Unexpected error initializing environment for theorem {theorem.full_name}: {e}"
        )
        return []

    if args.mcts_type == "alpha_zero":
        mcts_class = MCTS_AlphaZero
        mcts_kwargs = {"value_head": value_head}
    else:
        mcts_class = MCTS_GuidedRollout
        mcts_kwargs = {}

    mcts_kwargs["batch_size"] = args.batch_size

    runner = AgentRunner(
        env=env,
        transformer=transformer,
        mcts_class=mcts_class,
        mcts_kwargs=mcts_kwargs,
        num_iterations=args.num_iterations,
        max_steps=args.max_steps,
    )

    try:
        _, theorem_training_data = runner.run(
            collect_value_data=args.train_value_head,
            use_final_reward=args.use_final_reward,
            use_wandb=args.use_wandb,
        )
        logger.debug(
            f"Collected {len(theorem_training_data)} training samples for theorem: {theorem.full_name}"
        )
        return theorem_training_data
    except Exception as e:
        logger.error(f"Error during proof search for theorem {theorem.full_name}: {e}")
        return []
    finally:
        del runner
        del env
        gc.collect()


def main(args):
    # --- Setup wandb ---
    if args.use_wandb:
        wandb.init(
            entity="gerbennkoopman-university-of-amsterdam",
            project="lean-reinforcement",
            config=args,
        )

    # --- Setup checkpoint directory ---
    checkpoint_dir = get_checkpoint_dir()
    logger.info(f"Using checkpoint directory: {checkpoint_dir}")

    # --- Models ---
    transformer = Transformer()

    value_head = None
    start_epoch = 0
    if args.mcts_type == "alpha_zero" or args.train_value_head:
        value_head = ValueHead(transformer)

        # Load checkpoint if resume flag is set
        if args.resume:
            start_epoch = load_checkpoint(value_head, checkpoint_dir)
            logger.info(f"Resuming training from epoch {start_epoch}")

    log_gpu_memory("After model initialization - ")

    # --- DataLoader ---
    logger.info(f"Loading data from 'leandojo_benchmark_4/{args.data_type}'")
    corpus = Corpus(os.path.join("leandojo_benchmark_4/corpus.jsonl"))
    dataloader = LeanDataLoader(
        corpus, dataset_path="leandojo_benchmark_4", data_type=args.data_type
    )
    dataloader.trace_repo()

    # --- Self-Play and Training Loop ---
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        logger.info(f"Starting Epoch {epoch + 1}/{args.num_epochs}")
        training_data_buffer = []

        # Clear GPU cache at start of each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Shuffle training theorems each epoch
        random.shuffle(dataloader.train_data)

        theorems_to_process = dataloader.train_data[: args.num_theorems]
        logger.info(
            f"Processing {len(theorems_to_process)} theorems with {args.num_workers} workers."
        )

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_theorem,
                    thm_data,
                    corpus,
                    dataloader,
                    transformer,
                    value_head,
                    args,
                )
                for thm_data in theorems_to_process
            ]

            for i, future in enumerate(as_completed(futures)):
                try:
                    data = future.result()
                    if data:
                        training_data_buffer.extend(data)
                except Exception as e:
                    logger.error(f"Error in worker thread: {e}")

                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{len(theorems_to_process)} proofs")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- MODEL TRAINING STEP ---
        if not training_data_buffer:
            logger.warning("No data collected in this epoch. Skipping training.")
            continue

        # --- Analyze collected data ---
        if args.train_value_head:
            stats = analyze_value_data(training_data_buffer)
            print_training_stats(stats)

            # Optionally save training data for offline analysis
            if args.save_training_data:
                data_save_path = (
                    checkpoint_dir / f"training_data_epoch_{epoch + 1}.json"
                )
                save_training_data(training_data_buffer, data_save_path)

        # --- CONDITIONAL TRAINING ---
        if args.train_value_head:
            value_data = [d for d in training_data_buffer if d.get("type") == "value"]
            assert (
                value_head is not None
            ), "ValueHead must be initialized before training"
            train_value_head(
                value_head,
                value_data,
                epochs=args.train_epochs,
                use_wandb=args.use_wandb,
            )

        # --- Save checkpoints after each epoch ---
        if args.train_value_head and value_head is not None and args.save_checkpoints:
            save_checkpoint(value_head, epoch + 1, checkpoint_dir, args)
            logger.info(f"Checkpoint saved for epoch {epoch + 1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCTS-based Training Loop for Lean Prover"
    )
    # --- Data and MCTS Args ---
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["random", "novel_premises"],
        default="novel_premises",
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of self-play/training epochs.",
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=100,
        help="Number of theorems to attempt per epoch.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of MCTS iterations per step (reduced default for memory efficiency).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max steps per proof (reduced default for memory efficiency).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for MCTS search.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers for MCTS simulations.",
    )
    parser.add_argument(
        "--mcts-type",
        type=str,
        choices=["guided_rollout", "alpha_zero"],
        default="guided_rollout",
        help="Which MCTS algorithm to use for self-play.",
    )

    # --- Training Args ---
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Number of training epochs to run on collected data *per* self-play epoch.",
    )
    parser.add_argument(
        "--train-value-head",
        action="store_true",
        help="Train the value head after each epoch.",
    )
    parser.add_argument(
        "--use-final-reward",
        action="store_true",
        default=True,
        help="Whether to use the final reward for training (True) or the MCTS value estimates (False). Default: True.",
    )
    parser.add_argument(
        "--save-training-data",
        action="store_true",
        help="Save raw training data to JSON files for offline analysis.",
    )

    # --- Checkpoint Args ---
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=True,
        help="Save model checkpoints after each epoch (default: True).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint if available.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Override checkpoint directory (defaults to CHECKPOINT_DIR env var or ./checkpoints).",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        default=True,
        help="Use wandb for logging.",
    )

    args = parser.parse_args()

    # Override checkpoint directory if provided
    if args.checkpoint_dir:
        os.environ["CHECKPOINT_DIR"] = args.checkpoint_dir

    main(args)
