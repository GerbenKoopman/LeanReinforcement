#!/usr/bin/env python3
"""
Main script to run training for the TransformerAgent.
"""
import os
import sys
import logging
from pathlib import Path
from lean_dojo import LeanGitRepo
from lean_dojo.data_extraction.trace import is_available_in_cache, get_traced_repo_path
from lean_dojo.data_extraction.traced_data import TracedRepo

from .config import TransformerConfig
from .model.agent import TransformerAgent

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    """
    A simple entry point for local debugging that respects the cache-only philosophy.
    This script will fail if a pre-traced repository is not found.
    """
    config = TransformerConfig()

    # 1. Define the repository based on the configuration
    repo = LeanGitRepo(config.repo_url, config.repo_commit)

    # 2. Enforce cache-only behavior: Check for cache and fail if it's not there.
    if not is_available_in_cache(repo):
        logging.error(f"Repository not found in cache: {repo.url} @ {repo.commit}")
        logging.error("This script requires a pre-traced repository.")
        logging.error(
            "Please run the 'trace_repo.py' script first to generate the cache."
        )
        sys.exit(1)

    logging.info("Found repository in cache. Loading traced data...")

    # 3. Load the traced repository directly from the known cache path.
    # We no longer use RepoManager's get_traced_repo() to avoid its trace-fallback logic.
    try:
        traced_repo_path = get_traced_repo_path(repo)
        traced_repo = TracedRepo.load_from_disk(traced_repo_path)
        logging.info(f"Successfully loaded traced repository from: {traced_repo_path}")
    except Exception as e:
        logging.error(
            f"Failed to load traced repository from cache: {e}", exc_info=True
        )
        sys.exit(1)

    # 4. Load theorems and proceed with a small, well-defined task for debugging.
    theorems = list(traced_repo.get_traced_theorems())
    if not theorems:
        logging.error("No theorems found in the traced repository.")
        sys.exit(1)

    # Use a small subset for the debugging/demonstration task
    theorems_subset = theorems[:10]
    logging.info(f"Using a subset of {len(theorems_subset)} theorems for this run.")

    # ... (rest of the original logic for agent initialization, etc.) ...
    logging.info("Initializing agent...")
    agent = TransformerAgent(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        device="cpu",
    )
    logging.info("Agent initialized. Main script finished.")


if __name__ == "__main__":
    main()
