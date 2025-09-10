"""
Configuration management for LeanReinforcement.

This module provides centralized configuration management with validation
and environment variable support.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EnvironmentType(Enum):
    """Environment types."""

    STANDARD = "standard"
    PERSISTENT = "persistent"
    DISTRIBUTED = "distributed"


@dataclass
class EnvironmentConfig:
    """Configuration for Lean environments."""

    timeout: int = 300
    max_steps: int = 100
    type: EnvironmentType = EnvironmentType.PERSISTENT
    additional_imports: list = field(default_factory=list)

    # Caching configuration
    cache_dir: Optional[Path] = None
    enable_cache: bool = True

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path(os.getenv("CACHE_DIR", "./cache"))


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Basic training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    save_interval: int = 10

    # Model parameters
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    vocab_size: int = 10000

    # Training directories
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("./logs"))

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0

    def __post_init__(self):
        # Ensure directories exist
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    max_theorems: int = 100
    timeout_per_theorem: int = 300
    num_trials: int = 1

    # Analysis options
    analyze_attention: bool = True
    analyze_search_patterns: bool = True
    analyze_failure_modes: bool = True

    # Comparison baselines
    compare_random: bool = True
    compare_mcts: bool = True

    # Output options
    save_detailed_results: bool = True
    generate_visualizations: bool = True
    results_dir: Path = field(default_factory=lambda: Path("./evaluation_results"))


@dataclass
class SystemConfig:
    """System-wide configuration."""

    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Device configuration
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.

    # Resource limits
    memory_limit_gb: Optional[int] = None
    cpu_count: Optional[int] = None

    def __post_init__(self):
        if self.device == "auto":
            import torch

            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Config:
    """Main configuration container."""

    def __init__(
        self,
        environment: Optional[EnvironmentConfig] = None,
        training: Optional[TrainingConfig] = None,
        evaluation: Optional[EvaluationConfig] = None,
        system: Optional[SystemConfig] = None,
    ):
        self.environment = environment or EnvironmentConfig()
        self.training = training or TrainingConfig()
        self.evaluation = evaluation or EvaluationConfig()
        self.system = system or SystemConfig()

    @classmethod
    def from_file(cls, config_path: Path) -> "Config":
        """Load configuration from JSON file."""
        import json

        with open(config_path, "r") as f:
            data = json.load(f)

        return cls(
            environment=EnvironmentConfig(**data.get("environment", {})),
            training=TrainingConfig(**data.get("training", {})),
            evaluation=EvaluationConfig(**data.get("evaluation", {})),
            system=SystemConfig(**data.get("system", {})),
        )

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            environment=EnvironmentConfig(
                timeout=int(os.getenv("LEAN_TIMEOUT", "300")),
                max_steps=int(os.getenv("LEAN_MAX_STEPS", "100")),
                cache_dir=Path(os.getenv("CACHE_DIR", "./cache")),
            ),
            training=TrainingConfig(
                learning_rate=float(os.getenv("LEARNING_RATE", "1e-4")),
                batch_size=int(os.getenv("BATCH_SIZE", "32")),
                output_dir=Path(os.getenv("OUTPUT_DIR", "./outputs")),
            ),
            system=SystemConfig(
                log_level=LogLevel(os.getenv("LOG_LEVEL", "INFO")),
                device=os.getenv("DEVICE", "auto"),
            ),
        )

    def to_file(self, config_path: Path) -> None:
        """Save configuration to JSON file."""
        import json
        from dataclasses import asdict

        data = {
            "environment": asdict(self.environment),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "system": asdict(self.system),
        }

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        data = convert_paths(data)

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)


def get_default_config() -> Config:
    """Get the default configuration."""
    return Config()


def setup_logging(config: SystemConfig) -> None:
    """Setup logging based on configuration."""
    import logging

    logging.basicConfig(
        level=getattr(logging, config.log_level.value),
        format=config.log_format,
    )
