#!/usr/bin/env python3
"""
Main script to run training for the SimpleTransformerAgent.
"""
import logging
import argparse

from ....lean_rl.environment import LeanEnvironment
from ....lean_rl.agents.simple_transformer.agent import SimpleTransformerAgent
from ....lean_rl.agents.simple_transformer.config import SimpleTransformerConfig
from ....lean_rl.agents.simple_transformer.training import train


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Train a simple transformer agent for Lean."
    )
    # Add arguments for config overrides if needed
    args = parser.parse_args()

    config = SimpleTransformerConfig()

    # Initialize agent
    agent = SimpleTransformerAgent(config)

    # Get traced repo
    traced_repo = agent.repo_manager.get_traced_repo()
    theorems = traced_repo.get_traced_theorems()

    # For debugging, let's use a small subset of theorems
    theorems_subset = traced_repo.get_traced_theorems()[:10]

    # Initialize environment
    env = LeanEnvironment(
        traced_repo=traced_repo,
        max_steps=config.max_steps_per_episode,
        timeout=config.timeout,
    )

    # Start training
    train(agent, env, theorems_subset, config.num_epochs)


if __name__ == "__main__":
    main()
