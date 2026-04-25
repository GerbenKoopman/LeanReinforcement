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
    value_head_batch_size: int
    training_mode: str = "value_head"
    value_head_latent_dim: int = 1024
    train_value_head: bool = True
    use_hyperbolic: bool = False  # Use hyperbolic (Poincaré ball) value head
    use_final_reward: bool = False
    save_training_data: bool = True
    use_caching: bool = False
    debugging: bool = False

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
    num_tactics_to_expand: int = 64
    max_rollout_depth: int = 80

    # Search mode
    full_search: bool = (
        True  # Run MCTS from root with full budget (allows backtracking)
    )

    # Hyperbolicity
    curvature: float = 1.0

    # Max MCTS tree nodes — limits per-worker memory.
    # The PUCT-based pruner evicts worst-scored leaves at this limit.
    max_tree_nodes: int = 10000

    # Timeout parameters (all in seconds)
    # Note: These form a hierarchy - each level should be larger than the one below
    # env_timeout < max_time < proof_timeout
    max_time: float = 175  # Max time per MCTS search step
    env_timeout: int = 75  # Max time per tactic execution
    proof_timeout: float = 360  # Max time for entire proof search

    # Log MCTS search trees for qualitative analysis
    log_search_tree: bool = False

    # Hard memory limit (GiB) for each Lean 4 REPL subprocess.
    # Passed as --memory to the Lean runtime; exceeding it produces a
    # catchable DojoCrashError instead of triggering the OOM killer.
    lean_memory_limit_gb: int = 8

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainingConfig":
        return cls(
            data_type=getattr(args, "data_type", "novel_premises"),
            num_epochs=getattr(args, "num_epochs", 3),
            num_theorems=getattr(args, "num_theorems", 128),
            num_iterations=getattr(args, "num_iterations", 1000),
            max_steps=getattr(args, "max_steps", 80),
            batch_size=getattr(args, "batch_size", 1),
            num_workers=getattr(args, "num_workers", 16),
            mcts_type=getattr(args, "mcts_type", "alpha_zero"),
            indexed_corpus_path=getattr(args, "indexed_corpus_path", None),
            model_name=getattr(
                args,
                "model_name",
                "kaiyuy/leandojo-lean4-tacgen-byt5-small",
            ),
            num_tactics_to_expand=getattr(args, "num_tactics_to_expand", 64),
            max_rollout_depth=getattr(args, "max_rollout_depth", 80),
            max_time=getattr(args, "max_time", 175),
            env_timeout=getattr(args, "env_timeout", 75),
            proof_timeout=getattr(args, "proof_timeout", 360),
            training_mode=getattr(args, "training_mode", "value_head"),
            train_epochs=getattr(args, "train_epochs", 50),
            value_head_batch_size=getattr(args, "value_head_batch_size", 4),
            value_head_latent_dim=getattr(args, "value_head_latent_dim", 1024),
            train_value_head=getattr(args, "train_value_head", True),
            use_hyperbolic=getattr(args, "use_hyperbolic", False),
            use_final_reward=getattr(args, "use_final_reward", True),
            save_training_data=getattr(args, "save_training_data", True),
            use_caching=getattr(args, "use_caching", False),
            debugging=getattr(args, "debugging", False),
            seed=getattr(args, "seed", None),
            save_checkpoints=getattr(args, "save_checkpoints", True),
            resume=getattr(args, "resume", False),
            use_test_value_head=getattr(args, "use_test_value_head", False),
            checkpoint_dir=getattr(args, "checkpoint_dir", None),
            use_wandb=getattr(args, "use_wandb", True),
            inference_timeout=getattr(args, "inference_timeout", 600.0),
            curvature=getattr(args, "curvature", 1.0),
            full_search=getattr(args, "full_search", True),
            max_tree_nodes=getattr(args, "max_tree_nodes", 10000),
            lean_memory_limit_gb=getattr(args, "lean_memory_limit_gb", 8),
            log_search_tree=getattr(args, "log_search_tree", False),
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
        default=3,
        help="Number of self-play/training epochs.",
    )
    parser.add_argument(
        "--num-theorems",
        type=int,
        default=128,
        help="Number of theorems to attempt per epoch.",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=1000,
        help="Number of MCTS iterations per step (reduced default for memory efficiency).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=80,
        help="Max steps per proof (reduced default for memory efficiency).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
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
        default="alpha_zero",
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
        default=64,
        help="Number of tactics to expand in MCTS.",
    )
    parser.add_argument(
        "--max-rollout-depth",
        type=int,
        default=80,
        help="Max depth for MCTS rollout.",
    )
    parser.add_argument(
        "--max-tree-nodes",
        type=int,
        default=10000,
        help="Maximum number of nodes kept in the MCTS tree.",
    )
    parser.add_argument(
        "--max-time",
        type=float,
        default=175,
        help="Max time (seconds) per MCTS search step. Should be > env-timeout.",
    )
    parser.add_argument(
        "--env-timeout",
        type=int,
        default=75,
        help="Max time (seconds) per single tactic execution. Should be < max-time.",
    )
    parser.add_argument(
        "--proof-timeout",
        type=float,
        default=360,
        help="Max time (seconds) for entire proof search per theorem. Should be > max-time.",
    )
    parser.add_argument(
        "--lean-memory-limit-gb",
        type=int,
        default=8,
        help="Hard memory limit (GiB) for each Lean 4 REPL subprocess. "
        "Passed as --memory to the Lean runtime. When exceeded, Lean "
        "exits cleanly instead of triggering the OS OOM killer. "
        "Rule of thumb: total_ram / (num_workers + 2). Default: 8.",
    )

    # --- Hyperbolicity ---
    parser.add_argument(
        "--curvature",
        type=float,
        default=1.0,
        help="Curvature of the Poincare Disk, best kept between 0 and 1 (float).",
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
        "--training-mode",
        type=str,
        choices=["value_head", "ppo"],
        default="value_head",
        help="Training mode to use.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=50,
        help="Number of training epochs to run on collected data *per* self-play epoch.",
    )
    parser.add_argument(
        "--value-head-batch-size",
        type=int,
        default=4,
        help="Batch size for training the value head.",
    )
    parser.add_argument(
        "--value-head-latent-dim",
        type=int,
        default=1024,
        help="Hidden dimension for the value head MLP. Default: 1024.",
    )
    parser.add_argument(
        "--train-value-head",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the value head after each epoch.",
    )
    parser.add_argument(
        "--use-hyperbolic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use a hyperbolic (Poincaré ball) value head instead of the default MLP. "
        "Projects encoder features through a learned adapter into the Poincaré ball "
        "before the linear critic.",
    )
    parser.add_argument(
        "--use-final-reward",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    parser.add_argument(
        "--debugging",
        action="store_true",
        default=False,
        help="Enable maximum verbosity for troubleshooting: disable progress dashboard, "
        "set DEBUG-level logs, mirror worker logs to stderr, and emit per-theorem failure reasons.",
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
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use wandb for logging.",
    )
    parser.add_argument(
        "--log-search-tree",
        action="store_true",
        default=False,
        help="Log MCTS search trees for qualitative analysis.",
    )

    args = parser.parse_args()

    # Override checkpoint directory if provided
    if args.checkpoint_dir:
        os.environ["CHECKPOINT_DIR"] = args.checkpoint_dir

    return TrainingConfig.from_args(args)
