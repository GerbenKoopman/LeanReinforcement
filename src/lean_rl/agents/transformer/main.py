#!/usr/bin/env python3
"""
Main script to run training for the TransformerAgent.
"""
import logging
import argparse


from ...environment import LeanEnvironment
from ...agents.transformer.model.agent import TransformerAgent
from ...agents.transformer.config import TransformerConfig
from ...agents.transformer.training import train
from ...agents.transformer.data.repository import RepoManager


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train a transformer agent for Lean.")
    # Add arguments for config overrides if needed
    args = parser.parse_args()

    config = TransformerConfig()

    # Initialize agent
    agent = TransformerAgent(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        device=config.device,
    )

    # Get traced repo
    repo_manager = RepoManager(config.repo_url, config.repo_commit)
    traced_repo = repo_manager.get_traced_repo()
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
