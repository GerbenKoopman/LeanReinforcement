"""
Configuration Management for Hierarchical Transformer Agent.

This module provides configuration management, hyperparameter tuning,
and experimental setup utilities.
"""

import json
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
import logging
import optuna


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    vocab_size: int = 10000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1

    # Hierarchical specific
    strategic_vocab_size: int = 10
    tactical_vocab_size: int = 20
    max_parameter_length: int = 32

    # Attention parameters
    attention_dropout: float = 0.1
    feed_forward_dim: Optional[int] = None  # If None, use 4 * d_model

    def __post_init__(self):
        if self.feed_forward_dim is None:
            self.feed_forward_dim = 4 * self.d_model


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    max_episodes: int = 10000
    max_steps_per_episode: int = 100

    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    lr_scheduler: str = "plateau"  # plateau, cosine, linear
    gradient_clip_norm: float = 1.0

    # Search parameters
    max_search_time: float = 60.0
    beam_width: int = 16

    # Experience replay
    use_experience_replay: bool = True
    replay_buffer_size: int = 100000
    replay_batch_size: int = 64
    replay_start_size: int = 1000

    # Target network updates
    target_update_frequency: int = 1000
    target_update_tau: float = 0.005  # For soft updates

    # Regularization
    use_dropout: bool = True
    use_weight_decay: bool = True
    use_gradient_clipping: bool = True

    # Evaluation
    eval_frequency: int = 500
    eval_episodes: int = 100

    # Checkpointing
    save_frequency: int = 1000
    keep_best_n_checkpoints: int = 5

    # Environment
    timeout: int = 600
    reward_scheme: str = "dense"  # sparse, dense, shaped


@dataclass
class CurriculumConfig:
    """Curriculum learning configuration."""

    use_curriculum: bool = True
    curriculum_stages: int = 5
    difficulty_threshold: float = 0.7

    # Difficulty metrics
    difficulty_metric: str = "proof_length"  # proof_length, file_complexity, manual

    # Stage progression
    min_episodes_per_stage: int = 200
    stage_advancement_patience: int = 5  # Episodes to wait before checking advancement

    # Manual difficulty assignment (if using manual metric)
    file_difficulty_map: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.file_difficulty_map is None:
            self.file_difficulty_map = {
                "Basic.lean": 1,
                "Group/Defs.lean": 2,
                "Ring/Defs.lean": 3,
                "Topology": 4,
                "Analysis": 5,
            }


@dataclass
class DistributedConfig:
    """Distributed training configuration."""

    use_distributed: bool = False
    world_size: int = 1
    rank: int = 0
    backend: str = "nccl"
    init_method: str = "env://"

    # Data parallelism
    use_data_parallel: bool = False
    device_ids: Optional[List[int]] = None

    # Gradient synchronization
    sync_gradients: bool = True
    bucket_size_mb: int = 25


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Experiment metadata
    experiment_name: str = "hierarchical_transformer_experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)

    # Data and environment
    repo_url: str = "https://github.com/leanprover-community/mathlib4"
    repo_commit: str = "29dcec074de168ac2bf835a77ef68bbe069194c5"

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data"

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Hardware
    device: str = "auto"  # auto, cpu, cuda, cuda:0, etc.
    mixed_precision: bool = False
    compile_model: bool = False  # PyTorch 2.0 compile


class ConfigManager:
    """Manages experiment configurations and hyperparameter tuning."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def load_config(self, config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        if config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config_dict = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return self._dict_to_config(config_dict)

    def save_config(self, config: ExperimentConfig, config_path: Union[str, Path]):
        """Save configuration to file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(config)

        if config_path.suffix == ".json":
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)
        elif config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Extract sub-configs
        model_dict = config_dict.pop("model", {})
        training_dict = config_dict.pop("training", {})
        curriculum_dict = config_dict.pop("curriculum", {})
        distributed_dict = config_dict.pop("distributed", {})

        # Create sub-configs
        model_config = ModelConfig(**model_dict)
        training_config = TrainingConfig(**training_dict)
        curriculum_config = CurriculumConfig(**curriculum_dict)
        distributed_config = DistributedConfig(**distributed_dict)

        # Create main config
        experiment_config = ExperimentConfig(
            model=model_config,
            training=training_config,
            curriculum=curriculum_config,
            distributed=distributed_config,
            **config_dict,
        )

        return experiment_config

    def create_default_config(self) -> ExperimentConfig:
        """Create default configuration."""
        return ExperimentConfig()

    def validate_config(self, config: ExperimentConfig) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []

        # Model validation
        if config.model and config.model.d_model % config.model.n_heads != 0:
            warnings.append("d_model should be divisible by n_heads")

        if config.model and config.model.vocab_size <= 0:
            warnings.append("vocab_size should be positive")

        # Training validation
        if config.training and config.training.learning_rate <= 0:
            warnings.append("learning_rate should be positive")

        if config.training and config.training.batch_size <= 0:
            warnings.append("batch_size should be positive")

        if (
            config.training
            and config.training.replay_batch_size > config.training.replay_buffer_size
        ):
            warnings.append("replay_batch_size should not exceed replay_buffer_size")

        # Curriculum validation
        if (
            config.curriculum
            and config.curriculum.use_curriculum
            and config.curriculum.curriculum_stages <= 0
        ):
            warnings.append(
                "curriculum_stages should be positive when using curriculum"
            )

        # Device validation
        if config.device not in ["auto", "cpu"] and not config.device.startswith(
            "cuda"
        ):
            warnings.append(f"Unknown device: {config.device}")

        return warnings

    def get_device(self, config: ExperimentConfig) -> torch.device:
        """Get appropriate device based on config."""
        if config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(config.device)


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""

    def __init__(
        self,
        base_config: ExperimentConfig,
        study_name: str = "hierarchical_transformer_study",
    ):
        self.base_config = base_config
        self.study_name = study_name
        self.logger = logging.getLogger(__name__)

    def create_study(
        self, direction: str = "maximize", storage: Optional[str] = None
    ) -> optuna.Study:
        """Create Optuna study for hyperparameter optimization."""
        study = optuna.create_study(
            direction=direction,
            study_name=self.study_name,
            storage=storage,
            load_if_exists=True,
        )
        return study

    def suggest_config(self, trial: optuna.Trial) -> ExperimentConfig:
        """Suggest hyperparameters for a trial."""
        # Create copy of base config
        config_dict = asdict(self.base_config)

        # Model hyperparameters
        config_dict["model"]["d_model"] = trial.suggest_categorical(
            "d_model", [256, 512, 768, 1024]
        )
        config_dict["model"]["n_heads"] = trial.suggest_categorical(
            "n_heads", [4, 8, 12, 16]
        )
        config_dict["model"]["n_layers"] = trial.suggest_int("n_layers", 3, 8)
        config_dict["model"]["dropout"] = trial.suggest_float("dropout", 0.0, 0.3)

        # Ensure d_model is divisible by n_heads
        d_model = config_dict["model"]["d_model"]
        n_heads = config_dict["model"]["n_heads"]
        if d_model % n_heads != 0:
            # Adjust n_heads to be compatible
            valid_heads = [h for h in [4, 8, 12, 16] if d_model % h == 0]
            if valid_heads:
                config_dict["model"]["n_heads"] = trial.suggest_categorical(
                    f"n_heads_adjusted_{d_model}", valid_heads
                )

        # Training hyperparameters
        config_dict["training"]["learning_rate"] = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )
        config_dict["training"]["weight_decay"] = trial.suggest_float(
            "weight_decay", 1e-3, 1e-1, log=True
        )
        config_dict["training"]["batch_size"] = trial.suggest_categorical(
            "batch_size", [16, 32, 64, 128]
        )
        config_dict["training"]["beam_width"] = trial.suggest_categorical(
            "beam_width", [8, 16, 32, 64]
        )

        # Curriculum hyperparameters
        if config_dict["curriculum"]["use_curriculum"]:
            config_dict["curriculum"]["curriculum_stages"] = trial.suggest_int(
                "curriculum_stages", 3, 7
            )
            config_dict["curriculum"]["difficulty_threshold"] = trial.suggest_float(
                "difficulty_threshold", 0.5, 0.9
            )

        # Recreate config object
        return ConfigManager()._dict_to_config(config_dict)

    def objective_function(
        self, trial: optuna.Trial, train_function, eval_function
    ) -> float:
        """Objective function for optimization."""
        try:
            # Get suggested config
            config = self.suggest_config(trial)

            # Train model
            model, metrics = train_function(config)

            # Evaluate model
            eval_results = eval_function(model, config)

            # Return objective value (e.g., success rate)
            return eval_results.get("success_rate", 0.0)

        except Exception as e:
            self.logger.error(f"Trial failed: {e}")
            return 0.0  # Return worst possible score

    def run_optimization(
        self, train_function, eval_function, n_trials: int = 100
    ) -> optuna.Study:
        """Run hyperparameter optimization."""
        study = self.create_study()

        def objective(trial):
            return self.objective_function(trial, train_function, eval_function)

        study.optimize(objective, n_trials=n_trials)

        self.logger.info(f"Best trial: {study.best_trial.value}")
        self.logger.info(f"Best params: {study.best_trial.params}")

        return study


class ExperimentTracker:
    """Track experiments and their results."""

    def __init__(self, experiments_dir: Optional[Union[str, Path]] = None):
        import os
        
        if experiments_dir is None:
            # Get SCRATCH_SHARED from environment
            scratch_dir = os.getenv('SCRATCH_SHARED', '.')
            experiments_dir = Path(scratch_dir) / "experiments"
        
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_file = self.experiments_dir / "experiments.json"
        self.experiments = self._load_experiments()

        self.logger = logging.getLogger(__name__)

    def _load_experiments(self) -> Dict[str, Any]:
        """Load existing experiments."""
        if self.experiments_file.exists():
            with open(self.experiments_file, "r") as f:
                return json.load(f)
        return {}

    def _save_experiments(self):
        """Save experiments to file."""
        with open(self.experiments_file, "w") as f:
            json.dump(self.experiments, f, indent=2)

    def start_experiment(self, config: ExperimentConfig) -> str:
        """Start a new experiment and return experiment ID."""
        import time
        import uuid

        experiment_id = (
            f"{config.experiment_name}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        )

        experiment_info = {
            "id": experiment_id,
            "name": config.experiment_name,
            "description": config.description,
            "tags": config.tags,
            "config": asdict(config),
            "start_time": time.time(),
            "status": "running",
            "results": {},
        }

        self.experiments[experiment_id] = experiment_info
        self._save_experiments()

        # Create experiment directory
        exp_dir = self.experiments_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        # Save config to experiment directory
        ConfigManager().save_config(config, exp_dir / "config.yaml")

        self.logger.info(f"Started experiment: {experiment_id}")
        return experiment_id

    def update_experiment(self, experiment_id: str, results: Dict[str, Any]):
        """Update experiment with results."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        self.experiments[experiment_id]["results"].update(results)
        self._save_experiments()

    def finish_experiment(self, experiment_id: str, final_results: Dict[str, Any]):
        """Mark experiment as finished."""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")

        import time

        self.experiments[experiment_id]["status"] = "completed"
        self.experiments[experiment_id]["end_time"] = time.time()
        self.experiments[experiment_id]["final_results"] = final_results

        self._save_experiments()

        self.logger.info(f"Finished experiment: {experiment_id}")

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment info."""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self, status: Optional[str] = None, tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        experiments = list(self.experiments.values())

        if status:
            experiments = [exp for exp in experiments if exp.get("status") == status]

        if tag:
            experiments = [exp for exp in experiments if tag in exp.get("tags", [])]

        return experiments

    def get_best_experiment(
        self, metric: str = "success_rate"
    ) -> Optional[Dict[str, Any]]:
        """Get experiment with best performance on given metric."""
        completed_experiments = [
            exp for exp in self.experiments.values() if exp.get("status") == "completed"
        ]

        if not completed_experiments:
            return None

        best_experiment = max(
            completed_experiments,
            key=lambda exp: exp.get("final_results", {}).get(metric, 0.0),
        )

        return best_experiment


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """Create a set of default configurations for different scenarios."""

    configs = {}

    # Quick test config
    configs["quick_test"] = ExperimentConfig(
        experiment_name="quick_test",
        description="Quick test configuration for debugging",
        model=ModelConfig(d_model=256, n_heads=4, n_layers=3),
        training=TrainingConfig(
            max_episodes=100, eval_frequency=50, save_frequency=100
        ),
        curriculum=CurriculumConfig(use_curriculum=False),
    )

    # Small model config
    configs["small"] = ExperimentConfig(
        experiment_name="small_model",
        description="Small model for limited resources",
        model=ModelConfig(d_model=256, n_heads=8, n_layers=4),
        training=TrainingConfig(max_episodes=5000, batch_size=16),
    )

    # Standard config
    configs["standard"] = ExperimentConfig(
        experiment_name="standard_model",
        description="Standard configuration",
        model=ModelConfig(d_model=512, n_heads=8, n_layers=6),
        training=TrainingConfig(max_episodes=10000),
    )

    # Large model config
    configs["large"] = ExperimentConfig(
        experiment_name="large_model",
        description="Large model for maximum performance",
        model=ModelConfig(d_model=768, n_heads=12, n_layers=8),
        training=TrainingConfig(max_episodes=15000, batch_size=64),
    )

    # Distributed config
    configs["distributed"] = ExperimentConfig(
        experiment_name="distributed_training",
        description="Configuration for distributed training",
        model=ModelConfig(d_model=512, n_heads=8, n_layers=6),
        training=TrainingConfig(max_episodes=20000, batch_size=128),
        distributed=DistributedConfig(use_distributed=True, world_size=4),
    )

    return configs


if __name__ == "__main__":
    # Example usage

    # Create default configs
    configs = create_default_configs()

    # Save configs
    config_manager = ConfigManager()
    for name, config in configs.items():
        config_path = f"configs/{name}_config.yaml"
        Path("configs").mkdir(exist_ok=True)
        config_manager.save_config(config, config_path)
        print(f"Saved {name} config to {config_path}")

    # Example of loading and validating
    test_config = config_manager.load_config("configs/standard_config.yaml")
    warnings = config_manager.validate_config(test_config)

    if warnings:
        print("Config warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Config validation passed!")

    print(f"Device: {config_manager.get_device(test_config)}")
