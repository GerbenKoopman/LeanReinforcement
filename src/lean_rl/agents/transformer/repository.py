"""
This module handles the management of Lean repositories for the Transformer agent.

It provides a centralized way to initialize, cache, and load LeanGitRepo objects,
avoiding redundant tracing and building operations. The design is inspired by
repository handling in LeanDojo, ReProver, and LeanAgent.
"""

import os
import logging
from pathlib import Path
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

        try:
            if traced_repo_path.exists():
                logger.info(
                    f"Attempting to load traced repo from cache: {traced_repo_path} with build_deps={current_build_deps}"
                )
                self._traced_repo = TracedRepo.load_from_disk(traced_repo_path)
                self._traced_repo_build_deps = current_build_deps
                logger.info("Successfully loaded traced repo from cache.")
                return self._traced_repo
            else:
                logger.warning(
                    f"Traced repo not found in cache. Tracing repository: {self.repo.url}"
                )
                self._traced_repo = trace_repo(self.repo, build_deps=current_build_deps)
                self._traced_repo_build_deps = current_build_deps
                logger.info("Successfully traced repository.")
                return self._traced_repo
        except Exception as e:
            logger.error(f"Failed to load or trace repository: {e}")
            logger.warning("Falling back to tracing without cache.")
            self._traced_repo = trace_repo(self.repo, build_deps=current_build_deps)
            self._traced_repo_build_deps = current_build_deps
            return self._traced_repo
