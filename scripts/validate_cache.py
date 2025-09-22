"""
This script validates the integrity of a cached and traced Lean repository.

It initializes a RepoManager, loads the traced repository, and iterates through
all traced theorems to ensure they are readable and not corrupted.
"""

import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Add the source directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.lean_rl.agents.transformer.data.repository import RepoManager
from src.lean_rl.agents.transformer.simplified.hpc_config import create_hpc_config


def validate_repository_cache():
    """
    Validates the traced repository cache by attempting to load all theorems.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    try:
        logger.info("Loading HPC configuration...")
        config = create_hpc_config()

        logger.info(f"Repo URL: {config.repo_url}")
        logger.info(f"Repo Commit: {config.repo_commit}")

        repo_manager = RepoManager(
            repo_url=config.repo_url, repo_commit=config.repo_commit
        )

        logger.info("Accessing the traced repository...")
        traced_repo = repo_manager.get_traced_repo()

        if not traced_repo:
            logger.error("Failed to load or trace the repository.")
            return

        logger.info("Successfully loaded traced repository. Now validating theorems...")
        theorems = list(traced_repo.get_traced_theorems())

        corrupted_theorems = []

        for theorem in tqdm(theorems, desc="Validating theorems"):
            try:
                # Accessing properties of the theorem can trigger lazy loading
                _ = theorem.file_path
                _ = theorem.theorem
            except Exception as e:
                logger.error(f"Corruption detected in theorem: {str(theorem.theorem)}")
                logger.error(f"Error: {e}")
                corrupted_theorems.append(str(theorem.theorem))

        if corrupted_theorems:
            logger.warning(
                f"Validation finished. Found {len(corrupted_theorems)} corrupted theorems."
            )
            logger.warning("Corrupted theorems: " + ", ".join(corrupted_theorems))
        else:
            logger.info(
                f"Validation successful. All {len(theorems)} theorems are intact."
            )

    except Exception as e:
        logger.fatal(
            f"An unexpected error occurred during validation: {e}", exc_info=True
        )


if __name__ == "__main__":
    validate_repository_cache()
