"""
This module handles the management of Lean repositories for the Transformer agent.

It provides a centralized way to initialize, cache, and load LeanGitRepo objects,
avoiding redundant tracing and building operations. The design is inspired by
repository handling in LeanDojo, ReProver, and LeanAgent.
"""

import logging
from typing import Optional

from lean_dojo import LeanGitRepo, TracedRepo, get_traced_repo_path
from lean_dojo.data_extraction.trace import trace as trace_repo

logger = logging.getLogger(__name__)


class RepoManager:
    """
    Manages the LeanGitRepo and its traced versions.

    This class ensures that a repository is initialized only once and that
    pre-traced versions are loaded from cache whenever possible.
    """

    def __init__(self, repo_url: str, repo_commit: str, build_deps: bool = False):
        """
        Initializes the RepoManager.

        Args:
            repo_url (str): The URL of the Git repository.
            repo_commit (str): The commit hash of the repository.
            build_deps (bool): Whether to build dependencies when tracing.
                               Defaults to False as per HPC guide recommendations.
        """
        self.repo_url = repo_url
        self.repo_commit = repo_commit
        self.build_deps = build_deps
        self._repo: Optional[LeanGitRepo] = None
        self._traced_repo: Optional[TracedRepo] = None
        self._traced_repo_build_deps: Optional[bool] = None

    @property
    def repo(self) -> LeanGitRepo:
        """
        Returns the LeanGitRepo object, initializing it if necessary.
        """
        if self._repo is None:
            logger.info(
                f"Initializing LeanGitRepo for {self.repo_url} at commit {self.repo_commit}"
            )
            self._repo = LeanGitRepo(self.repo_url, self.repo_commit)
        return self._repo

    def get_traced_repo(self, build_deps: Optional[bool] = None) -> TracedRepo:
        """
        Loads or creates a traced version of the repository.

        This function first checks if a pre-traced version of the repository
        exists in the cache. If so, it loads it. Otherwise, it traces the
        repository and saves it to the cache.

        Args:
            build_deps (bool, optional): Override the default build_deps setting.
                                         If None, uses the value from the constructor.

        Returns:
            TracedRepo: The traced repository object.

        Raises:
            RuntimeError: If the repository cannot be loaded or traced.
        """
        current_build_deps = build_deps if build_deps is not None else self.build_deps

        if (
            self._traced_repo is not None
            and self._traced_repo_build_deps == current_build_deps
        ):
            logger.debug("Returning cached TracedRepo object.")
            return self._traced_repo

        traced_repo_path = get_traced_repo_path(
            self.repo, build_deps=current_build_deps
        )

        if traced_repo_path.exists():
            logger.info(
                f"Attempting to lazily load traced repo from {traced_repo_path}"
            )
            try:
                # This method is significantly faster as it lazy-loads the data.
                repo_to_return = TracedRepo.from_traced_files(traced_repo_path)
                logger.info("Successfully loaded traced repo using lazy loading.")
            except Exception as e:
                logger.error(f"Failed to load traced repo with from_traced_files: {e}")
                logger.info("Falling back to older, slower load_from_disk method...")
                try:
                    repo_to_return = TracedRepo.load_from_disk(traced_repo_path)
                    logger.info("Successfully loaded traced repo with fallback method.")
                except Exception as e_fallback:
                    logger.error(f"Fallback loading method also failed: {e_fallback}")
                    # If loading from cache fails, fall through to tracing
                    repo_to_return = None
        else:
            repo_to_return = None

        if repo_to_return is None:
            logger.warning(
                f"Traced repo not found in cache or failed to load. Tracing repository: {self.repo.url}"
            )
            try:
                repo_to_return = trace_repo(self.repo, build_deps=current_build_deps)
                logger.info("Successfully traced repository.")
            except Exception as final_e:
                raise RuntimeError(
                    f"Fatal: Could not load or trace repository. Final error: {final_e}"
                ) from final_e

        if repo_to_return is None:
            raise RuntimeError("Fatal: Repository tracing returned None unexpectedly.")

        self._traced_repo = repo_to_return
        self._traced_repo_build_deps = current_build_deps
        return self._traced_repo
