#!/usr/bin/env python3
"""
HPC-optimized configuration for the simplified LeanDojo RL agent.
Integrates with SLURM, environment variables, and distributed training.
"""

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging


@dataclass
class HPCConfig:
    """HPC-specific configuration for the simplified agent."""

    # === HPC Environment Variables ===
    scratch_shared: Optional[str] = None
    cache_dir: Optional[str] = None

    # === SLURM Integration ===
    job_id: Optional[str] = None
    task_id: Optional[int] = None
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None

    # === Memory Management ===
    max_memory_gb: float = 32.0
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True

    # === Distributed Training ===
    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0

    # === Output Management ===
    experiment_name: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None

    def __post_init__(self):
        """Auto-configure from environment variables."""
        # HPC directory setup - only use if available
        scratch_env = os.getenv("SCRATCH_SHARED")
        cache_env = os.getenv("CACHE_DIR")

        # Only set HPC paths if they actually exist or we're on HPC
        if scratch_env and (Path(scratch_env).exists() or os.getenv("SLURM_JOB_ID")):
            self.scratch_shared = scratch_env

        if cache_env and (Path(cache_env).parent.exists() or os.getenv("SLURM_JOB_ID")):
            self.cache_dir = cache_env

        # SLURM environment
        self.job_id = os.getenv("SLURM_JOB_ID")
        self.task_id = int(os.getenv("SLURM_PROCID", 0))
        self.num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", 1))
        self.num_gpus = int(os.getenv("SLURM_GPUS", 1))

        # Experiment setup
        if not self.experiment_name:
            timestamp = int(time.time())
            self.experiment_name = f"simplified_training_{timestamp}"

        # Directory setup - only if HPC paths are available
        if self.scratch_shared:
            base_dir = Path(self.scratch_shared)
            self.checkpoint_dir = self.checkpoint_dir or str(base_dir / "checkpoints")
            self.log_dir = self.log_dir or str(base_dir / "training_logs")

        # Create directories - gracefully handle failures
        self._create_directories()

        # Setup HPC environment variables
        self._setup_hpc_environment()

    def _create_directories(self):
        """Create required HPC directories."""
        if self.scratch_shared:
            try:
                base_dir = Path(self.scratch_shared)
                # Only create if we have write permissions
                if base_dir.exists() or self._can_create_directory(base_dir.parent):
                    required_dirs = [
                        "checkpoints",
                        "evaluation_results",
                        "evaluation_plots",
                        "tensorboard_logs",
                        "training_logs",
                        "saved_models",
                        "experiments",
                        "test_reports",
                    ]

                    for dir_name in required_dirs:
                        (base_dir / dir_name).mkdir(parents=True, exist_ok=True)
            except (PermissionError, OSError):
                # Fallback to local directories if HPC paths not accessible
                print(
                    f"Warning: Cannot create HPC directories in {self.scratch_shared}"
                )
                self.scratch_shared = None
                self.checkpoint_dir = None
                self.log_dir = None

    def _can_create_directory(self, parent_path: Path) -> bool:
        """Check if we can create directories in the given path."""
        try:
            if parent_path.exists():
                return os.access(parent_path, os.W_OK)
            else:
                return self._can_create_directory(parent_path.parent)
        except (OSError, PermissionError):
            return False

    def _setup_hpc_environment(self):
        """Setup critical HPC environment variables."""
        # Prevent redundant building/tracing
        os.environ["LEAN_CACHE_ONLY"] = "1"
        os.environ["DISABLE_BUILD_DEPS"] = "1"
        os.environ["LOAD_USED_PACKAGES_ONLY"] = "1"
        os.environ["NO_LAKE_BUILD"] = "1"
        os.environ["SKIP_DEPENDENCIES"] = "1"

        # Set cache directory to local fallback if HPC not available
        if self.cache_dir and Path(self.cache_dir).parent.exists():
            os.environ["LEAN_DOJO_CACHE_DIR"] = self.cache_dir
        else:
            # Use local cache directory that we can actually create
            local_cache = Path.home() / ".cache" / "lean_dojo"
            local_cache.mkdir(parents=True, exist_ok=True)
            os.environ["LEAN_DOJO_CACHE_DIR"] = str(local_cache)

        # Ray configuration for distributed env
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "0"

        # PyTorch optimizations
        if self.use_mixed_precision:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


@dataclass
class SimpleHPCConfig:
    """Enhanced simple configuration with HPC support."""

    # === Model Architecture ===
    vocab_size: int = 10000
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4

    # === Training Parameters ===
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_episodes: int = 200
    max_steps_per_episode: int = 30
    timeout: int = 600
    reward_scheme: str = "dense"

    # === HPC Configuration ===
    hpc: HPCConfig = field(default_factory=HPCConfig)

    # === Device Management ===
    device: Optional[str] = None

    # === Mathlib4 Repository ===
    repo_url: str = "https://github.com/leanprover-community/mathlib4"
    repo_commit: str = "29dcec074de168ac2bf835a77ef68bbe069194c5"

    # === LeanDojo Settings ===
    max_theorems: int = 100
    theorem_difficulty: str = "basic"

    # === Model Persistence ===
    model_save_path: Optional[str] = None
    checkpoint_frequency: int = 100  # Episodes between checkpoints

    def __post_init__(self):
        """Validate and setup configuration."""
        # Prioritize environment variables for repository, like the main agent.
        self.repo_url = os.getenv("LEAN_REPO_URL", self.repo_url)
        self.repo_commit = os.getenv("LEAN_REPO_COMMIT", self.repo_commit)

        # Model validation
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        # Device setup
        if self.device is None:
            import torch

            if torch.cuda.is_available():
                self.device = f"cuda:0"
            else:
                self.device = "cpu"

        # HPC override for repo_url if it's a local path
        if self.hpc.cache_dir:
            cache_dir = Path(self.hpc.cache_dir)
            # This assumes a standard naming convention for the cloned repo.
            repo_name = (
                "leanprover-community-mathlib4-29dcec074de168ac2bf835a77ef68bbe069194c5"
            )
            repo_path = cache_dir / repo_name
            if repo_path.exists():
                self.repo_url = str(repo_path)

        # Model save path with HPC directories
        if self.model_save_path is None:
            if self.hpc.checkpoint_dir:
                self.model_save_path = f"{self.hpc.checkpoint_dir}/simplified_model_{self.hpc.experiment_name}.pt"
            else:
                self.model_save_path = "leandojo_simplified_model.pt"

        # Adjust batch size for GPU memory
        if self.hpc.use_mixed_precision and "cuda" in str(self.device):
            # Can increase batch size with mixed precision
            self.batch_size = min(self.batch_size * 2, 64)

    def get_cache_dir(self) -> str:
        """Get cache directory with validation."""
        cache_dir = self.hpc.cache_dir
        if not cache_dir:
            raise RuntimeError(
                "CACHE_DIR not configured. Set CACHE_DIR environment variable."
            )

        if not Path(cache_dir).exists():
            raise RuntimeError(f"Cache directory does not exist: {cache_dir}")

        return cache_dir

    def setup_logging(self) -> logging.Logger:
        """Setup HPC-compatible logging."""
        logger = logging.getLogger("simplified_trainer")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)

            # File handler for HPC
            file_handler = None
            if self.hpc.log_dir:
                try:
                    log_file = (
                        Path(self.hpc.log_dir) / f"{self.hpc.experiment_name}.log"
                    )
                    log_file.parent.mkdir(parents=True, exist_ok=True)
                    file_handler = logging.FileHandler(log_file)
                    file_handler.setLevel(logging.DEBUG)
                except (OSError, PermissionError):
                    # Skip file logging if can't create log directory
                    file_handler = None

            # Formatter
            formatter = logging.Formatter(
                f"[{self.hpc.job_id}] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            if file_handler:
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            console_handler.setFormatter(formatter)

            logger.addHandler(console_handler)

        return logger

    def save_config(self, path: Optional[str] = None) -> str:
        """Save configuration for reproducibility."""
        if path is None:
            if self.hpc.experiment_name:
                path = (
                    f"{self.hpc.checkpoint_dir}/{self.hpc.experiment_name}_config.json"
                )
            else:
                path = "config.json"

        import json
        from dataclasses import asdict

        config_dict = asdict(self)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

        return path

    @classmethod
    def load_config(cls, path: str) -> "SimpleHPCConfig":
        """Load configuration from file."""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct nested HPCConfig
        hpc_dict = config_dict.pop("hpc", {})
        hpc_config = HPCConfig(**hpc_dict)

        return cls(hpc=hpc_config, **config_dict)


def create_hpc_config(
    d_model: int = 256, max_episodes: int = 200, use_distributed: bool = False, **kwargs
) -> SimpleHPCConfig:
    """Factory function to create HPC-optimized configuration."""

    hpc_config = HPCConfig(use_distributed=use_distributed)

    config = SimpleHPCConfig(
        d_model=d_model, max_episodes=max_episodes, hpc=hpc_config, **kwargs
    )

    return config


# HPC-optimized configurations for different cluster partitions
HPC_A100_CONFIG = SimpleHPCConfig(
    d_model=512,
    n_layers=6,
    batch_size=64,
    max_episodes=1000,
    max_theorems=200,
    hpc=HPCConfig(
        max_memory_gb=64.0, use_mixed_precision=True, gradient_checkpointing=True
    ),
)

HPC_MIG_CONFIG = SimpleHPCConfig(
    d_model=256,
    n_layers=4,
    batch_size=32,
    max_episodes=500,
    max_theorems=100,
    hpc=HPCConfig(
        max_memory_gb=16.0, use_mixed_precision=False, gradient_checkpointing=True
    ),
)
