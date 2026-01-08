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
    indexed_corpus_path: Optional[str]

    # Training Args
    train_epochs: int
    train_value_head: bool
    use_final_reward: bool
    save_training_data: bool

    # Checkpoint Args
    save_checkpoints: bool
    resume: bool
    use_test_value_head: bool
    checkpoint_dir: Optional[str]
    use_wandb: bool

    # Worker memory management
    max_theorems_per_worker: int = (
        8  # Restart workers after this many theorems to prevent memory leaks
    )

    # Model Args
    model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
    num_tactics_to_expand: int = 32
    max_rollout_depth: int = 30
    max_time: float = 300.0
    env_timeout: int = 100

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        return cls(
            data_type=args.data_type,
            num_epochs=args.num_epochs,
            num_theorems=args.num_theorems,
            num_iterations=args.num_iterations,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mcts_type=args.mcts_type,
            indexed_corpus_path=args.indexed_corpus_path,
            model_name=args.model_name,
            num_tactics_to_expand=args.num_tactics_to_expand,
            max_rollout_depth=args.max_rollout_depth,
            max_time=args.max_time,
            env_timeout=args.env_timeout,
            train_epochs=args.train_epochs,
            train_value_head=args.train_value_head,
            use_final_reward=args.use_final_reward,
            save_training_data=args.save_training_data,
            save_checkpoints=args.save_checkpoints,
            resume=args.resume,
            use_test_value_head=args.use_test_value_head,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.use_wandb,
            max_theorems_per_worker=args.max_theorems_per_worker,
        )


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
        default=100,
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
    parser.add_argument(
        "--indexed-corpus-path",
        type=str,
        default=None,
        help="Path to a pickled IndexedCorpus file. If provided, loads corpus from this file instead of recomputing.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="kaiyuy/leandojo-lean4-tacgen-byt5-small",
        help="HuggingFace model name for tactic generation.",
    )
    parser.add_argument(
        "--num-tactics-to-expand",
        type=int,
        default=32,
        help="Number of tactics to generate/expand at each node.",
    )
    parser.add_argument(
        "--max-rollout-depth",
        type=int,
        default=30,
        help="Max depth for rollout simulations.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=1200.0,
        help="Max time (seconds) for MCTS search.",
    )
    parser.add_argument(
        "--env-timeout",
        type=int,
        default=100,
        help="Max time (seconds) per tactic.",
    )
    parser.add_argument(
        "--max-theorems-per-worker",
        type=int,
        default=8,
        help="Restart workers after processing this many theorems to prevent memory leaks.",
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
        "--use-test-value-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the test value head checkpoint instead of the training one.",
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

    return TrainingConfig.from_args(args)
