"""
Utility functions to analyze and visualize training data collected from MCTS rollouts.
"""

import json
from typing import List, Dict
from pathlib import Path
import numpy as np
from loguru import logger


def analyze_value_data(training_data: List[Dict]) -> Dict:
    """
    Analyze value head training data and compute statistics.

    Args:
        training_data: List of training data dictionaries

    Returns:
        Dictionary containing statistics about the training data
    """
    if not training_data:
        return {}

    # Filter for value data
    value_data = [d for d in training_data if d.get("type") == "value"]

    if not value_data:
        return {}

    # Extract relevant fields
    value_targets = [d["value_target"] for d in value_data]
    final_rewards = [d.get("final_reward", 0) for d in value_data]
    mcts_values = [d.get("mcts_value", None) for d in value_data if "mcts_value" in d]
    visit_counts = [d.get("visit_count", 0) for d in value_data]
    steps = [d.get("step", 0) for d in value_data]

    stats = {
        "total_samples": len(value_data),
        "positive_samples": sum(1 for v in final_rewards if v > 0),
        "negative_samples": sum(1 for v in final_rewards if v < 0),
        "value_target": {
            "mean": float(np.mean(value_targets)),
            "std": float(np.std(value_targets)),
            "min": float(np.min(value_targets)),
            "max": float(np.max(value_targets)),
        },
        "visit_count": {
            "mean": float(np.mean(visit_counts)),
            "std": float(np.std(visit_counts)),
            "min": float(np.min(visit_counts)),
            "max": float(np.max(visit_counts)),
        },
        "step": {
            "mean": float(np.mean(steps)),
            "std": float(np.std(steps)),
            "min": float(np.min(steps)),
            "max": float(np.max(steps)),
        },
    }

    # Add MCTS value statistics if available
    if mcts_values:
        mcts_array = np.array(mcts_values)  # Convert to numpy array for type safety
        stats["mcts_value"] = {
            "mean": float(np.mean(mcts_array)),
            "std": float(np.std(mcts_array)),
            "min": float(np.min(mcts_array)),
            "max": float(np.max(mcts_array)),
            "samples_with_mcts": len(mcts_values),
        }

    return stats


def print_training_stats(stats: Dict):
    """
    Pretty print training data statistics.

    Args:
        stats: Statistics dictionary from analyze_value_data
    """
    if not stats:
        logger.warning("No statistics to print")
        return

    logger.info("=" * 60)
    logger.info("TRAINING DATA STATISTICS")
    logger.info("=" * 60)

    logger.info(f"Total samples: {stats['total_samples']}")
    logger.info(
        f"  Positive (successful proofs): {stats['positive_samples']} "
        f"({100 * stats['positive_samples'] / stats['total_samples']:.1f}%)"
    )
    logger.info(
        f"  Negative (failed proofs): {stats['negative_samples']} "
        f"({100 * stats['negative_samples'] / stats['total_samples']:.1f}%)"
    )

    logger.info("\nValue Targets:")
    vt = stats["value_target"]
    logger.info(
        f"  Mean: {vt['mean']:.4f}, Std: {vt['std']:.4f}, Range: [{vt['min']:.4f}, {vt['max']:.4f}]"
    )

    logger.info("\nVisit Counts:")
    vc = stats["visit_count"]
    logger.info(
        f"  Mean: {vc['mean']:.1f}, Std: {vc['std']:.1f}, Range: [{vc['min']:.0f}, {vc['max']:.0f}]"
    )

    logger.info("\nSteps in Trajectory:")
    st = stats["step"]
    logger.info(
        f"  Mean: {st['mean']:.1f}, Std: {st['std']:.1f}, Range: [{st['min']:.0f}, {st['max']:.0f}]"
    )

    if "mcts_value" in stats:
        logger.info("\nMCTS Value Estimates:")
        mv = stats["mcts_value"]
        logger.info(
            f"  Mean: {mv['mean']:.4f}, Std: {mv['std']:.4f}, Range: [{mv['min']:.4f}, {mv['max']:.4f}]"
        )
        logger.info(f"  Samples with MCTS values: {mv['samples_with_mcts']}")

    logger.info("=" * 60)


def save_training_data(training_data: List[Dict], output_path: Path):
    """
    Save training data to a JSON file.

    Args:
        training_data: List of training data dictionaries
        output_path: Path to save the JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    logger.info(f"Training data saved to {output_path}")


def load_training_data(input_path: Path) -> List[Dict]:
    """
    Load training data from a JSON file.

    Args:
        input_path: Path to the JSON file

    Returns:
        List of training data dictionaries
    """
    with open(input_path, "r") as f:
        training_data = json.load(f)

    logger.info(f"Loaded {len(training_data)} training samples from {input_path}")
    return training_data
