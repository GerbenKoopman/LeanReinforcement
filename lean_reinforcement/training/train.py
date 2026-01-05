"""
Main script for agent training and guided-rollout mcts dataset creation. Will
implement tactic generation training in the future.
"""

import sys
from lean_reinforcement.utilities.config import get_config
from lean_reinforcement.training.trainer import Trainer
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    try:
        args = get_config()
        trainer = Trainer(args)
        trainer.train()
        sys.exit(0)  # Explicit success exit
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)  # Non-zero exit code to signal failure
