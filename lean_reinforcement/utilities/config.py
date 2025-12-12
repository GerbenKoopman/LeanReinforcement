from dataclasses import dataclass
import argparse
import os
from typing import Optional


@dataclass
class TrainingConfig:
    # Data and MCTS Args
    data_type: str
    num_epochs: int
    num_theorems: int
    num_iterations: int
    max_steps: int
    batch_size: int
    num_workers: int
    mcts_type: str

    # Training Args
    train_epochs: int
    train_value_head: bool
    use_final_reward: bool
    save_training_data: bool

    # Checkpoint Args
    save_checkpoints: bool
    resume: bool
    checkpoint_dir: Optional[str]
    use_wandb: bool


def get_config() -> TrainingConfig:
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
        default=8,
        help="Batch size for MCTS search.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers for processing theorems.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the value head after each epoch.",
    )
    parser.add_argument(
        "--use-final-reward",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the final reward (1.0 or -1.0) for all steps in the proof.",
    )
    parser.add_argument(
        "--save-training-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw training data to JSON files for offline analysis.",
    )

    # --- Checkpoint Args ---
    parser.add_argument(
        "--save-checkpoints",
        action=argparse.BooleanOptionalAction,
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

    return TrainingConfig(
        data_type=args.data_type,
        num_epochs=args.num_epochs,
        num_theorems=args.num_theorems,
        num_iterations=args.num_iterations,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        mcts_type=args.mcts_type,
        train_epochs=args.train_epochs,
        train_value_head=args.train_value_head,
        use_final_reward=args.use_final_reward,
        save_training_data=args.save_training_data,
        save_checkpoints=args.save_checkpoints,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        use_wandb=args.use_wandb,
    )
