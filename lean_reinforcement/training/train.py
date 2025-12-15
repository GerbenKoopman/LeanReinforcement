"""
Main script for agent training and guided-rollout mcts dataset creation. Will
implement tactic generation training in the future.
"""

from lean_reinforcement.utilities.config import get_config
from lean_reinforcement.training.trainer import Trainer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if __name__ == "__main__":
    args = get_config()
    trainer = Trainer(args)
    trainer.train()
