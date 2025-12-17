"""
Data loader for LeanDojo traced repositories and theorems.
"""

import os
import json
from typing import List, Optional
from pathlib import Path

from lean_dojo import LeanGitRepo, TracedRepo, trace, Theorem
from ReProver.common import Corpus, Pos

from lean_reinforcement.utilities.types import TheoremData
from lean_reinforcement.utilities.validation import is_list_of_theorem_data


class LeanDataLoader:

    def __init__(
        self,
        corpus: Corpus,
        dataset_path: str = "leandojo_benchmark_4",
        data_type: str = "novel_premises",
    ):
        self.corpus = corpus
        self.dataset_path = dataset_path
        self.data_type = data_type

        self.train_data = self._load_split("train")
        self.test_data = self._load_split("test")
        self.val_data = self._load_split("val")

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
    ) -> TracedRepo:
        """
        Traces a Lean Repository using the LeanDojo library.
        """

        repo = LeanGitRepo(url, commit)

        traced_repo = trace(repo)

        return traced_repo

    def get_premises(self, theorem: Theorem, theorem_pos: Pos) -> List[str]:
        """Retrieve all accessible premises given a theorem."""
        return [
            str(p)
            for p in self.corpus.get_accessible_premises(
                str(theorem.file_path), theorem_pos
            )
        ]
