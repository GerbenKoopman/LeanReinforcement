from dataclasses import dataclass, field
import argparse
import os
from typing import List, Optional


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
    value_head_batch_size: int
    value_head_hidden_dims: List[int] = field(default_factory=lambda: [256])
    train_value_head: bool = True
    use_final_reward: bool = False
    save_training_data: bool = True
    use_caching: bool = False

    # Reproducibility
    seed: Optional[int] = None

    # Checkpoint Args
    save_checkpoints: bool = True
    resume: bool = False
    use_test_value_head: bool = False
    checkpoint_dir: Optional[str] = None
    use_wandb: bool = True

    # Inference / IPC Args
    inference_timeout: float = 600.0

    # Model Args
    model_name: str = "kaiyuy/leandojo-lean4-tacgen-byt5-small"
    num_tactics_to_expand: int = 32
    max_rollout_depth: int = 30

    # Search mode
    full_search: bool = (
        True  # Run MCTS from root with full budget (allows backtracking)
    )

    # Timeout parameters (all in seconds)
    # Note: These form a hierarchy - each level should be larger than the one below
    # env_timeout < max_time < proof_timeout
    max_time: float = 300.0  # Max time per MCTS search step
    env_timeout: int = 180  # Max time per tactic execution
    proof_timeout: float = 1200.0  # Max time for entire proof search

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
            proof_timeout=args.proof_timeout,
            train_epochs=args.train_epochs,
            value_head_batch_size=args.value_head_batch_size,
            value_head_hidden_dims=args.value_head_hidden_dims,
            train_value_head=args.train_value_head,
            use_final_reward=args.use_final_reward,
            save_training_data=args.save_training_data,
            use_caching=args.use_caching,
            seed=args.seed,
            save_checkpoints=args.save_checkpoints,
            resume=args.resume,
            use_test_value_head=args.use_test_value_head,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.use_wandb,
            inference_timeout=args.inference_timeout,
            full_search=args.full_search,
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
        default=200,
        help="Number of MCTS iterations per step (reduced default for memory efficiency).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10,
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
        help="Number of tactics to expand in MCTS.",
    )
    parser.add_argument(
        "--max-rollout-depth",
        type=int,
        default=30,
        help="Max depth for MCTS rollout.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=300.0,
        help="Max time (seconds) per MCTS search step. Should be > env-timeout.",
    )
    parser.add_argument(
        "--env-timeout",
        type=int,
        default=180,
        help="Max time (seconds) per single tactic execution. Should be < max-time.",
    )
    parser.add_argument(
        "--proof-timeout",
        type=float,
        default=1200.0,
        help="Max time (seconds) for entire proof search per theorem. Should be > max-time.",
    )

    # --- Search mode ---
    parser.add_argument(
        "--full-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run MCTS from root with full budget (enables backtracking). "
        "Disable with --no-full-search for step-by-step commitment.",
    )

    # --- Inference / IPC Args ---
    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=600.0,
        help="Max time (seconds) to wait for inference server responses. Independent of proof timeouts.",
    )

    # --- Training Args ---
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=1,
        help="Number of training epochs to run on collected data *per* self-play epoch.",
    )
    parser.add_argument(
        "--value-head-batch-size",
        type=int,
        default=4,
        help="Batch size for training the value head.",
    )
    parser.add_argument(
        "--value-head-hidden-dims",
        type=int,
        nargs="*",
        default=[256],
        help="Hidden dimensions for the value head MLP. Empty list means direct linear projection. Default: [256].",
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
    parser.add_argument(
        "--use-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache encoded node features whenever possible (memory intensive).",
    )

    # --- Reproducibility ---
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility. Sets Python, NumPy, and PyTorch RNGs.",
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

    # Auto-load GPU params if gpu_params.json exists and user didn't
    # explicitly set batch-size / num-tactics-to-expand on the CLI.
    _apply_gpu_params(args, parser)

    return TrainingConfig.from_args(args)


def _apply_gpu_params(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Override batch_size/num_tactics_to_expand from gpu_params.json when not set explicitly."""
    import json
    from pathlib import Path

    gpu_params_path = Path("gpu_params.json")
    if not gpu_params_path.exists():
        return

    try:
        with open(gpu_params_path) as f:
            gpu_params = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    algo = args.mcts_type
    algo_params = gpu_params.get(algo)
    if algo_params is None:
        return

    # Only apply if user used the default values (didn't pass them explicitly)
    defaults = parser.parse_args([])  # get defaults

    if args.batch_size == defaults.batch_size:
        args.batch_size = algo_params["batch_size"]

    if args.num_tactics_to_expand == defaults.num_tactics_to_expand:
        args.num_tactics_to_expand = algo_params["num_tactics_to_expand"]
