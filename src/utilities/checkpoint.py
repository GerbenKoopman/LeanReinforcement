"""
Utilities for checkpoint management across the project.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
