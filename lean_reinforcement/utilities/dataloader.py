"""
Data loader for LeanDojo traced repositories and theorems.
"""

import os
import json
import shutil
from typing import List, Optional
from pathlib import Path
from loguru import logger

from lean_dojo import LeanGitRepo, TracedRepo, trace, Theorem
from ReProver.common import Corpus, Pos

from lean_reinforcement.utilities.types import TheoremData
from lean_reinforcement.utilities.validation import is_list_of_theorem_data


class LeanDataLoader:

    def __init__(
        self,
        corpus: Optional[Corpus] = None,
        dataset_path: str = "leandojo_benchmark_4",
        data_type: str = "novel_premises",
        load_splits: bool = True,
    ):
        self.corpus = corpus
        self.dataset_path = dataset_path
        self.data_type = data_type

        if load_splits:
            self.train_data = self._load_split("train")
            self.test_data = self._load_split("test")
            self.val_data = self._load_split("val")
        else:
            self.train_data = []
            self.test_data = []
            self.val_data = []

    def _load_split(self, split: str) -> List[TheoremData]:
        """
        Loads a specific split of the dataset (train, test, or val).
        """
        file_path = os.path.join(self.dataset_path, self.data_type, f"{split}.json")
        with open(file_path, "r") as f:
            data = json.load(f)
            if is_list_of_theorem_data(data):
                return data
            raise ValueError(f"Invalid data format in {file_path}")

    def extract_theorem(self, data: TheoremData) -> Optional[Theorem]:
        url = data["url"]
        commit = data["commit"]
        file_path = data["file_path"]
        full_name = data["full_name"]

        if url is None or commit is None or file_path is None or full_name is None:
            return None

        repo = LeanGitRepo(url, commit)
        theorem = Theorem(repo, Path(file_path), full_name)

        return theorem

    def extract_tactics(self, data: dict) -> List[str]:
        traced_tactics = data["traced_tactics"]
        tactics_list = [verbose_tactic["tactic"] for verbose_tactic in traced_tactics]

        return tactics_list

    def trace_repo(
        self,
        url: str = "https://github.com/leanprover-community/mathlib4",
        commit: str = "29dcec074de168ac2bf835a77ef68bbe069194c5",
        max_retries: int = 2,
    ) -> TracedRepo:
        """
        Traces a Lean Repository using the LeanDojo library.
        Handles corrupted cache by clearing and retrying.
        """
        from git.exc import InvalidGitRepositoryError

        repo = LeanGitRepo(url, commit)

        for attempt in range(max_retries):
            try:
                traced_repo = trace(repo)
                return traced_repo
            except InvalidGitRepositoryError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Corrupted LeanDojo cache detected (attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    cache_paths: List[Path] = []
                    cache_env = os.environ.get("LEAN_DOJO_CACHE_DIR") or os.environ.get(
                        "CACHE_DIR"
                    )
                    if cache_env:
                        cache_paths.append(Path(cache_env))
                    else:
                        cache_paths.append(Path.home() / ".cache" / "lean_dojo")

                    cache_paths.append(Path(str(e)))
                    self._clear_cache_paths(cache_paths)
                    logger.info("Retrying trace operation...")
                else:
                    logger.error(
                        f"Failed to trace repository after {max_retries} attempts. "
                        f"Cache may be corrupted at: {e}"
                    )
                    raise

        raise RuntimeError("trace_repo failed to return or raise")

    def _clear_cache_paths(self, paths: List[Path]) -> None:
        for path in paths:
            if path.exists():
                logger.info(f"Clearing LeanDojo cache at {path}")
                shutil.rmtree(path, ignore_errors=True)

    def get_premises(self, theorem: Theorem, theorem_pos: Pos) -> List[str]:
        """Retrieve all accessible premises given a theorem."""
        if self.corpus is None:
            raise ValueError("Corpus not set. Cannot retrieve premises.")
        return [
            str(p)
            for p in self.corpus.get_accessible_premises(
                str(theorem.file_path), theorem_pos
            )
        ]
