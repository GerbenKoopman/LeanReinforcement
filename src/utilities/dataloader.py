"""
Data loader for LeanDojo traced repositories and theorems.
"""

import os
import json
from typing import List, Optional

from lean_dojo import LeanGitRepo, TracedRepo, trace, Theorem
from ReProver.common import Corpus, Pos


class DataLoader:
    def __init__(
        self,
        dataset_path: str = "leandojo_benchmark_4",
        data_type: str = "novel_premises",
        jsonl_path: Optional[str] = None,
    ):
        if jsonl_path is None:
            jsonl_path = os.path.join(dataset_path, "corpus.jsonl")
        self.corpus = Corpus(jsonl_path)
        self.dataset_path = dataset_path
        self.data_type = data_type

        self.train_data = self._load_split("train")
        self.test_data = self._load_split("test")
        self.val_data = self._load_split("val")

    # TODO: Verify that output is indeed technically a dict. See
    # structure_example.json for reference
    def _load_split(self, split: str) -> List[dict]:
        """
        Loads a specific split of the dataset (train, test, or val).
        """
        file_path = os.path.join(self.dataset_path, self.data_type, f"{split}.json")
        with open(file_path, "r") as f:
            return json.load(f)

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
