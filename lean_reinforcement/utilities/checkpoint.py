"""
Utilities for checkpoint management across the project.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from dotenv import load_dotenv


from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.utilities.config import TrainingConfig

# Load environment variables
load_dotenv()


def save_checkpoint(
    value_head: ValueHead,
    epoch: int,
    checkpoint_dir: Path,
    args: TrainingConfig,
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


def get_checkpoint_dir() -> Path:
    """
    Get the checkpoint directory from environment variable or use default.

    Returns:
        Path object pointing to the checkpoint directory
    """
    checkpoint_dir = os.getenv("CHECKPOINT_DIR")
    if checkpoint_dir:
        return Path(checkpoint_dir)
    else:
        # Fallback to local checkpoints directory
        logger.warning("CHECKPOINT_DIR not set in environment, using ./checkpoints")
        return Path("checkpoints")


def save_training_metadata(
    checkpoint_dir: Path, epoch: int, metadata: Dict[str, Any]
) -> None:
    """
    Save training metadata (hyperparameters, epoch info, etc.) alongside checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoints
        epoch: Current epoch number
        metadata: Dictionary containing training metadata
    """
    metadata_file = checkpoint_dir / "training_metadata.json"

    # Load existing metadata if it exists
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            existing_metadata = json.load(f)
    else:
        existing_metadata = {}

    # Update with new metadata
    existing_metadata.update(
        {"last_epoch": epoch, "last_updated": str(Path.cwd()), **metadata}
    )

    # Save updated metadata
    with open(metadata_file, "w") as f:
        json.dump(existing_metadata, f, indent=2)

    logger.debug(f"Training metadata saved to {metadata_file}")


def cleanup_old_checkpoints(
    checkpoint_dir: Path, prefix: str = "value_head", keep_last_n: int = 5
) -> None:
    """
    Remove old checkpoints, keeping only the most recent N checkpoints.
    Always preserves the '_latest.pth' checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints
        prefix: Prefix to filter checkpoints
        keep_last_n: Number of checkpoints to keep (excluding _latest.pth)
    """
    if not checkpoint_dir.exists():
        return

    # Get all epoch-specific checkpoints (exclude _latest.pth)
    checkpoints = [p for p in checkpoint_dir.glob(f"{prefix}_epoch_*.pth")]

    if len(checkpoints) <= keep_last_n:
        return

    # Sort by modification time
    checkpoints.sort(key=lambda p: p.stat().st_mtime)

    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_last_n]
    for checkpoint in to_remove:
        try:
            checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {checkpoint.name}")
        except Exception as e:
            logger.error(f"Failed to remove checkpoint {checkpoint}: {e}")
