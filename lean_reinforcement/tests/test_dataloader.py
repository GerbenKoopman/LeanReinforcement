import unittest
import os
from unittest.mock import patch, mock_open, MagicMock

from lean_dojo import Theorem
from ReProver.common import Pos

from lean_reinforcement.utilities.dataloader import LeanDataLoader


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "test_dataset"
        self.data_type = "test_type"
        self.jsonl_path = os.path.join(self.dataset_path, "corpus.jsonl")

        # Mock the file system
        self.mock_fs = {
            self.jsonl_path: '{"key": "value"}',
            os.path.join(
                self.dataset_path, self.data_type, "train.json"
            ): '[{"id": 1}]',
            os.path.join(self.dataset_path, self.data_type, "test.json"): '[{"id": 2}]',
            os.path.join(self.dataset_path, self.data_type, "val.json"): '[{"id": 3}]',
        }

    @patch("builtins.open", new_callable=mock_open)
    def test_initialization(self, mock_file):
        # Arrange
        mock_file.side_effect = lambda path, *args, **kwargs: mock_open(
            read_data=self.mock_fs.get(path, "")
        ).return_value

        mock_corpus = MagicMock()

        # Act
        loader = LeanDataLoader(
            corpus=mock_corpus, dataset_path=self.dataset_path, data_type=self.data_type
        )

        # Assert
        self.assertEqual(loader.corpus, mock_corpus)
        self.assertEqual(loader.train_data, [{"id": 1}])
        self.assertEqual(loader.test_data, [{"id": 2}])
        self.assertEqual(loader.val_data, [{"id": 3}])

    @patch("lean_reinforcement.utilities.dataloader.LeanGitRepo")
    @patch("lean_reinforcement.utilities.dataloader.Theorem")
    def test_extract_theorem(self, MockTheorem, MockLeanGitRepo):
        # Arrange
        mock_corpus = MagicMock()
        loader = LeanDataLoader(corpus=mock_corpus)
        data = {
            "url": "test_url",
            "commit": "test_commit",
            "file_path": "test_file.lean",
            "full_name": "test_theorem",
        }

        # Act
        loader.extract_theorem(data)

        # Assert
        MockLeanGitRepo.assert_called_once_with("test_url", "test_commit")
        MockTheorem.assert_called_once_with(
            MockLeanGitRepo.return_value, "test_file.lean", "test_theorem"
        )

    def test_extract_tactics(self):
        # Arrange
        mock_corpus = MagicMock()
        loader = LeanDataLoader(corpus=mock_corpus)
        data = {
            "traced_tactics": [
                {"tactic": "tactic1"},
                {"tactic": "tactic2"},
            ]
        }

        # Act
        tactics = loader.extract_tactics(data)

        # Assert
        self.assertEqual(tactics, ["tactic1", "tactic2"])

    @patch("lean_reinforcement.utilities.dataloader.trace")
    @patch("lean_reinforcement.utilities.dataloader.LeanGitRepo")
    def test_trace_repo(self, MockLeanGitRepo, mock_trace):
        # Arrange
        mock_corpus = MagicMock()
        loader = LeanDataLoader(corpus=mock_corpus)
        url = "test_url"
        commit = "test_commit"

        # Act
        loader.trace_repo(url, commit)

        # Assert
        MockLeanGitRepo.assert_called_once_with(url, commit)
        mock_trace.assert_called_once_with(MockLeanGitRepo.return_value)

    def test_get_premises(self):
        # Arrange
        mock_corpus = MagicMock()
        mock_corpus.get_accessible_premises.return_value = ["p1", "p2"]
        loader = LeanDataLoader(corpus=mock_corpus)

        mock_theorem = MagicMock(spec=Theorem)
        mock_theorem.file_path = "path/to/file.lean"
        theorem_pos = Pos(1, 0)

        # Act
        premises = loader.get_premises(mock_theorem, theorem_pos)

        # Assert
        self.assertEqual(premises, ["p1", "p2"])
        mock_corpus.get_accessible_premises.assert_called_once_with(
            "path/to/file.lean", theorem_pos
        )


if __name__ == "__main__":
    unittest.main()
