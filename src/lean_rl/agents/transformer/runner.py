"""
Main Runner for Hierarchical Transformer Agent Training and Testing.

This script provides a comprehensive command-line interface for training,
testing, and evaluating the hierarchical transformer agent.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import traceback

# Import our modules
from .training import HierarchicalTransformerTrainer
from .testing import HierarchicalTransformerTester, run_comprehensive_tests
from .evaluation import HierarchicalTransformerEvaluator, run_evaluation
from .config import (
    ConfigManager,
    ExperimentConfig,
    ExperimentTracker,
    HyperparameterTuner,
    create_default_configs,
)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def train_command(args):
    """Execute training command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    try:
        # Load or create config
        config_manager = ConfigManager()

        if args.config:
            config = config_manager.load_config(args.config)
            logger.info(f"Loaded config from {args.config}")
        else:
            # Create default config and override with command line args
            config = ExperimentConfig()

            # Override with command line arguments
            if args.learning_rate:
                config.training.learning_rate = args.learning_rate
            if args.batch_size:
                config.training.batch_size = args.batch_size
            if args.max_episodes:
                config.training.max_episodes = args.max_episodes
            if args.d_model:
                config.model.d_model = args.d_model
            if args.n_heads:
                config.model.n_heads = args.n_heads
            if args.n_layers:
                config.model.n_layers = args.n_layers

            logger.info("Using default config with command line overrides")

        # Set experiment name
        if args.experiment_name:
            config.experiment_name = args.experiment_name

        # Set output directories
        if args.output_dir:
            config.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
            config.log_dir = str(Path(args.output_dir) / "logs")

        # Validate config
        warnings = config_manager.validate_config(config)
        if warnings:
            logger.warning("Config validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")

        # Start experiment tracking
        tracker = ExperimentTracker()
        experiment_id = tracker.start_experiment(config)
        logger.info(f"Started experiment: {experiment_id}")

        # Initialize trainer
        trainer = HierarchicalTransformerTrainer(config)

        # Load checkpoint if specified
        if args.resume_from:
            episode = trainer.load_checkpoint(args.resume_from)
            logger.info(f"Resumed training from episode {episode}")

        # Run training
        trainer.train()

        # Finish experiment
        final_results = {
            "final_episode": trainer.metrics.episode,
            "total_steps": trainer.metrics.total_steps,
            "final_success_rate": (
                trainer.metrics.success_rates[-1]
                if trainer.metrics.success_rates
                else 0.0
            ),
        }
        tracker.finish_experiment(experiment_id, final_results)

        logger.info("Training completed successfully!")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def test_command(args):
    """Execute testing command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting testing...")

    try:
        if args.test_type == "comprehensive":
            results = run_comprehensive_tests(args.model_path)

            print("\n" + "=" * 60)
            print("COMPREHENSIVE TEST RESULTS")
            print("=" * 60)
            print(
                f"Unit Tests: {results.unit_tests_passed}/{results.unit_tests_passed + results.unit_tests_failed}"
            )
            print(f"Success Rate: {results.unit_test_success_rate:.3f}")
            print(
                f"Theorem Proving: {results.theorems_proved}/{results.total_theorems_tested}"
            )
            print(f"Proving Success Rate: {results.success_rate:.3f}")
            print(
                f"Performance: {results.inference_time_ms:.2f}ms inference, {results.memory_usage_mb:.2f}MB memory"
            )

        elif args.test_type == "unit":
            tester = HierarchicalTransformerTester(args.model_path)
            tester._run_unit_tests()
            tester._print_test_summary()

        elif args.test_type == "performance":
            tester = HierarchicalTransformerTester(args.model_path)
            tester._run_performance_benchmarks()
            print(f"Inference time: {tester.results.inference_time_ms:.2f} ms")
            print(f"Memory usage: {tester.results.memory_usage_mb:.2f} MB")
            print(
                f"Search efficiency: {tester.results.search_efficiency:.2f} depth/sec"
            )

        elif args.test_type == "theorems":
            tester = HierarchicalTransformerTester(args.model_path)
            tester._run_theorem_proving_tests()
            print(
                f"Proved {tester.results.theorems_proved}/{tester.results.total_theorems_tested} theorems"
            )
            print(f"Success rate: {tester.results.success_rate:.3f}")

        logger.info("Testing completed successfully!")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def evaluate_command(args):
    """Execute evaluation command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation...")

    try:
        from .evaluation import EvaluationConfig

        # Create evaluation config
        eval_config = EvaluationConfig(
            max_steps_per_theorem=args.max_steps,
            timeout_per_theorem=args.timeout,
            results_dir=args.output_dir or "evaluation_results",
            generate_visualizations=not args.no_plots,
            save_detailed_results=True,
        )

        # Run evaluation
        results = run_evaluation(args.model_path, eval_config)

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Theorems evaluated: {results.total_theorems}")
        print(f"Theorems proved: {results.theorems_proved}")
        print(f"Success rate: {results.success_rate:.3f}")
        print(f"Average proof length: {results.average_proof_length:.1f} steps")
        print(f"Average search time: {results.average_search_time:.2f} seconds")

        if results.baseline_results:
            print(f"\nBaseline comparisons:")
            for baseline, metrics in results.baseline_results.items():
                print(f"  {baseline}: {metrics['success_rate']:.3f} success rate")

        print(f"\nDetailed results saved to: {eval_config.results_dir}")

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def hyperparameter_tune_command(args):
    """Execute hyperparameter tuning command."""
    logger = logging.getLogger(__name__)
    logger.info("Starting hyperparameter tuning...")

    try:
        # Load base config
        config_manager = ConfigManager()
        if args.config:
            base_config = config_manager.load_config(args.config)
        else:
            base_config = ExperimentConfig()

        # Initialize tuner
        tuner = HyperparameterTuner(
            base_config, study_name=args.study_name or "hierarchical_transformer_study"
        )

        # Define training and evaluation functions
        def train_function(config):
            trainer = HierarchicalTransformerTrainer(config)
            trainer.train()
            return trainer.agent, trainer.metrics

        def eval_function(model, config):
            from .evaluation import EvaluationConfig

            eval_config = EvaluationConfig(max_steps_per_theorem=50)  # Quick evaluation
            evaluator = HierarchicalTransformerEvaluator(model, eval_config)
            results = evaluator.evaluate()
            return {"success_rate": results.success_rate}

        # Run optimization
        study = tuner.run_optimization(
            train_function, eval_function, n_trials=args.n_trials
        )

        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING RESULTS")
        print("=" * 60)
        print(f"Best trial value: {study.best_trial.value:.3f}")
        print(f"Best parameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")

        # Save best config - create config from best parameters
        config_manager = ConfigManager()
        best_config = config_manager.create_default_config()

        # Apply best parameters to config
        best_params = study.best_trial.params
        if "d_model" in best_params:
            best_config.model.d_model = best_params["d_model"]
        if "n_heads" in best_params:
            best_config.model.n_heads = best_params["n_heads"]
        if "n_layers" in best_params:
            best_config.model.n_layers = best_params["n_layers"]
        if "dropout" in best_params:
            best_config.model.dropout = best_params["dropout"]
        if "learning_rate" in best_params:
            best_config.training.learning_rate = best_params["learning_rate"]
        if "weight_decay" in best_params:
            best_config.training.weight_decay = best_params["weight_decay"]
        if "batch_size" in best_params:
            best_config.training.batch_size = best_params["batch_size"]
        if "beam_width" in best_params:
            best_config.training.beam_width = best_params["beam_width"]
        output_path = f"best_config_{study.study_name}.yaml"
        config_manager.save_config(best_config, output_path)
        print(f"\nBest configuration saved to: {output_path}")

        logger.info("Hyperparameter tuning completed successfully!")

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def config_command(args):
    """Execute config-related commands."""
    logger = logging.getLogger(__name__)

    try:
        config_manager = ConfigManager()

        if args.config_action == "create-defaults":
            # Create default configs
            configs = create_default_configs()

            output_dir = Path(args.output_dir or "configs")
            output_dir.mkdir(exist_ok=True)

            for name, config in configs.items():
                config_path = output_dir / f"{name}_config.yaml"
                config_manager.save_config(config, config_path)
                print(f"Created {name} config: {config_path}")

        elif args.config_action == "validate":
            if not args.config:
                print("Error: --config required for validation")
                sys.exit(1)

            config = config_manager.load_config(args.config)
            warnings = config_manager.validate_config(config)

            if warnings:
                print("Configuration warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            else:
                print("Configuration is valid!")

        elif args.config_action == "convert":
            if not args.config:
                print("Error: --config required for conversion")
                sys.exit(1)

            config = config_manager.load_config(args.config)

            # Determine output format
            input_path = Path(args.config)
            if args.output_file:
                output_path = Path(args.output_file)
            else:
                if input_path.suffix == ".yaml":
                    output_path = input_path.with_suffix(".json")
                else:
                    output_path = input_path.with_suffix(".yaml")

            config_manager.save_config(config, output_path)
            print(f"Converted {input_path} -> {output_path}")

        logger.info("Config command completed successfully!")

    except Exception as e:
        logger.error(f"Config command failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def list_experiments_command(args):
    """List experiments."""
    tracker = ExperimentTracker()
    experiments = tracker.list_experiments(status=args.status, tag=args.tag)

    if not experiments:
        print("No experiments found.")
        return

    print(f"\nFound {len(experiments)} experiments:")
    print("-" * 80)

    for exp in experiments:
        status = exp.get("status", "unknown")
        start_time = exp.get("start_time", 0)

        print(f"ID: {exp['id']}")
        print(f"Name: {exp['name']}")
        print(f"Status: {status}")
        print(f"Description: {exp.get('description', 'N/A')}")
        print(f"Tags: {', '.join(exp.get('tags', []))}")

        if "final_results" in exp:
            results = exp["final_results"]
            success_rate = results.get("final_success_rate", "N/A")
            print(f"Final Success Rate: {success_rate}")

        print("-" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Hierarchical Transformer Agent for Lean Theorem Proving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python -m lean_rl.agents.transformer.runner train --experiment-name my_experiment

  # Train with custom config
  python -m lean_rl.agents.transformer.runner train --config my_config.yaml

  # Resume training from checkpoint
  python -m lean_rl.agents.transformer.runner train --resume-from checkpoint.pt

  # Run comprehensive tests
  python -m lean_rl.agents.transformer.runner test --model-path model.pt --test-type comprehensive

  # Evaluate trained model
  python -m lean_rl.agents.transformer.runner evaluate --model-path model.pt --output-dir results

  # Hyperparameter tuning
  python -m lean_rl.agents.transformer.runner tune --n-trials 50

  # Create default configs
  python -m lean_rl.agents.transformer.runner config create-defaults
        """,
    )

    # Global arguments
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument("--log-file", type=str, help="Log file path")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the agent")
    train_parser.add_argument("--config", type=str, help="Configuration file path")
    train_parser.add_argument("--experiment-name", type=str, help="Experiment name")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    train_parser.add_argument("--resume-from", type=str, help="Resume from checkpoint")

    # Training hyperparameters
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--max-episodes", type=int, help="Maximum episodes")
    train_parser.add_argument("--d-model", type=int, help="Model dimension")
    train_parser.add_argument("--n-heads", type=int, help="Number of attention heads")
    train_parser.add_argument("--n-layers", type=int, help="Number of layers")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the agent")
    test_parser.add_argument("--model-path", type=str, help="Path to trained model")
    test_parser.add_argument(
        "--test-type",
        choices=["comprehensive", "unit", "performance", "theorems"],
        default="comprehensive",
        help="Type of test to run",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the agent")
    eval_parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
    )
    eval_parser.add_argument(
        "--output-dir", type=str, help="Output directory for results"
    )
    eval_parser.add_argument(
        "--max-steps", type=int, default=100, help="Max steps per theorem"
    )
    eval_parser.add_argument(
        "--timeout", type=int, default=300, help="Timeout per theorem (seconds)"
    )
    eval_parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot generation"
    )

    # Hyperparameter tuning command
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning")
    tune_parser.add_argument("--config", type=str, help="Base configuration file")
    tune_parser.add_argument("--study-name", type=str, help="Optuna study name")
    tune_parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of trials"
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration utilities")
    config_parser.add_argument(
        "config_action",
        choices=["create-defaults", "validate", "convert"],
        help="Configuration action",
    )
    config_parser.add_argument("--config", type=str, help="Configuration file")
    config_parser.add_argument("--output-dir", type=str, help="Output directory")
    config_parser.add_argument(
        "--output-file", type=str, help="Output file for conversion"
    )

    # List experiments command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--status", type=str, help="Filter by status")
    list_parser.add_argument("--tag", type=str, help="Filter by tag")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    if args.command == "train":
        train_command(args)
    elif args.command == "test":
        test_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "tune":
        hyperparameter_tune_command(args)
    elif args.command == "config":
        config_command(args)
    elif args.command == "list":
        list_experiments_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
