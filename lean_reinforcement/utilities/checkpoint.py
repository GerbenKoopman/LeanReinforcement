"""
Utilities for checkpoint management across the project.
"""

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from dotenv import load_dotenv


from lean_reinforcement.agent.value_head import ValueHead
from lean_reinforcement.utilities.config import TrainingConfig

# Load environment variables
load_dotenv()


def get_next_iteration(base_checkpoint_dir: Path, mcts_type: str) -> int:
    """
    Find the next iteration number for a given mcts_type.

    Scans existing directories matching the pattern {mcts_type}-{N}/
    and returns N+1 for the highest N found, or 1 if none exist.

    Args:
        base_checkpoint_dir: Base directory containing iteration folders
        mcts_type: The MCTS type (e.g., 'alpha_zero' or 'guided_rollout')

    Returns:
        The next iteration number to use
    """
    if not base_checkpoint_dir.exists():
        return 1

    pattern = re.compile(rf"^{re.escape(mcts_type)}-(\d+)$")
    max_iteration = 0

    for item in base_checkpoint_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                iteration = int(match.group(1))
                max_iteration = max(max_iteration, iteration)

    return max_iteration + 1


def get_iteration_checkpoint_dir(
    base_checkpoint_dir: Path, mcts_type: str, resume: bool = False
) -> Path:
    """
    Get the checkpoint directory for a specific mcts_type and iteration.

    If resume=True, returns the latest existing iteration directory.
    If resume=False, creates and returns a new iteration directory.

    Args:
        base_checkpoint_dir: Base directory containing iteration folders
        mcts_type: The MCTS type (e.g., 'alpha_zero' or 'guided_rollout')
        resume: Whether to resume from existing checkpoint or start new

    Returns:
        Path to the iteration-specific checkpoint directory
    """
    if resume:
        latest_iteration = get_next_iteration(base_checkpoint_dir, mcts_type) - 1
        if latest_iteration < 1:
            logger.warning(
                f"No existing checkpoint found for {mcts_type}. Starting new iteration 1."
            )
            latest_iteration = 1
        iteration_dir = base_checkpoint_dir / f"{mcts_type}-{latest_iteration}"
    else:
        next_iteration = get_next_iteration(base_checkpoint_dir, mcts_type)
        iteration_dir = base_checkpoint_dir / f"{mcts_type}-{next_iteration}"

    iteration_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using iteration checkpoint directory: {iteration_dir}")
    return iteration_dir


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
            epoch_checkpoints = list(checkpoint_dir.glob(f"{prefix}_epoch_*.pth"))
            if epoch_checkpoints:
                # Sort numerically by epoch number (not lexicographically)
                def _extract_epoch(p: Path) -> int:
                    try:
                        return int(p.stem.split("_")[-1])
                    except ValueError:
                        return 0

                epoch_checkpoints.sort(key=_extract_epoch)
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
        {"last_epoch": epoch, "last_updated": datetime.now().isoformat(), **metadata}
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
