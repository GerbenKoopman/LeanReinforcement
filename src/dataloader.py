"""
Data loader for LeanDojo traced repositories and theorems.
"""

from typing import List, Tuple
from lean_dojo import LeanGitRepo, TracedRepo, TracedTheorem, trace

from lean_dojo import Theorem
from ReProver.common import Corpus, Pos


class DataLoader:
    def __init__(self, jsonl_path: str = "corpus.jsonl"):
        self.corpus = Corpus(jsonl_path)

    def trace_repo(
        self,
        url: str = "https://github.com/leanprover-community/mathlib4",
        commit: str = "29dcec074de168ac2bf835a77ef68bbe069194c5",
    ):
        """
        Traces a Lean Repository using the LeanDojo library.
        """

        repo = LeanGitRepo(url, commit)

        traced_repo = trace(repo)

        return traced_repo

    def load_theorems(
        self,
        repo: TracedRepo,
    ) -> Tuple[List[TracedTheorem], List[TracedTheorem]]:
        """
        Loads theorems and seperates them into tactic proof theorems and statement theorems.
        """

        traced_theorems = repo.get_traced_theorems()
        tactic_theorems = []
        statement_theorems = []

        for theorem in traced_theorems:
            if theorem.get_tactic_proof() is not None:
                tactic_theorems.append(theorem)
            else:
                statement_theorems.append(theorem)

        return tactic_theorems, statement_theorems

    def get_premises(self, theorem: Theorem, theorem_pos: Pos) -> List[str]:
        """Retrieve all accessible premises given a theorem."""
        return [
            str(p)
            for p in self.corpus.get_accessible_premises(
                str(theorem.file_path), theorem_pos
            )
        ]
