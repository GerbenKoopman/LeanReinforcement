import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from lean_dojo import TacticState

from lean_reinforcement.agent.mcts.base_mcts import Node
from lean_reinforcement.utilities.tree_analysis import (
    analyze_search_tree_directory,
    analyze_search_tree_file,
    get_embedding_radius_vs_depth,
    get_gromov_hyperbolicity_and_map,
    get_mcts_shape_analysis,
    load_search_tree_root,
    plot_2d_embeddings,
)


class TestTreeAnalysis(unittest.TestCase):
    @staticmethod
    def _state(pp: str) -> Mock:
        state = Mock(spec=TacticState)
        state.pp = pp
        return state

    def test_shape_analysis_on_live_tree(self) -> None:
        root = Node(self._state("root"))
        child_a = Node(self._state("child_a"), parent=root, action="a")
        child_b = Node(self._state("child_b"), parent=root, action="b")
        root.children = [child_a, child_b]
        root.visit_count = 10
        child_a.visit_count = 4
        child_b.visit_count = 6

        stats = get_mcts_shape_analysis(root)

        self.assertEqual(stats["max_depth"], 1)
        self.assertEqual(stats["total_nodes"], 3)
        self.assertEqual(stats["max_branching_factor"], 2)
        self.assertAlmostEqual(stats["avg_visitation"], 20 / 3)

    def test_embedding_and_hyperbolicity_handle_tensor_features(self) -> None:
        root = Node(self._state("root"))
        child = Node(self._state("child"), parent=root, action="a")
        root.children = [child]
        root.encoder_features = torch.tensor([3.0, 4.0])
        child.encoder_features = torch.tensor([0.0, 5.0])

        radius_stats = get_embedding_radius_vs_depth(root)
        hyperbolic_stats = get_gromov_hyperbolicity_and_map(root)

        self.assertIn(0, radius_stats)
        self.assertEqual(radius_stats[0], 5.0)
        self.assertIn("gromov_hyperbolicity", hyperbolic_stats)
        self.assertIn("mAP_spearman", hyperbolic_stats)

    def test_json_search_tree_analysis(self) -> None:
        payload = {
            "root": "root",
            "nodes": {
                "root": {
                    "state": "root",
                    "action": None,
                    "visit_count": 10,
                    "max_value": 0.5,
                    "prior_p": 0.0,
                    "depth": 0,
                    "encoder_features": [3.0, 4.0],
                    "analysis_embedding": [0.5, 1.5],
                    "children": ["child"],
                },
                "child": {
                    "state": "child",
                    "action": "a",
                    "visit_count": 6,
                    "max_value": 0.3,
                    "prior_p": 0.0,
                    "depth": 1,
                    "encoder_features": [0.0, 5.0],
                    "analysis_embedding": [1.0, 2.0],
                    "children": [],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tree_dir = Path(tmpdir) / "search_trees"
            tree_dir.mkdir(parents=True, exist_ok=True)
            tree_file = tree_dir / "search_tree_test.json"
            with open(tree_file, "w") as handle:
                json.dump(payload, handle)

            file_report = analyze_search_tree_file(tree_file)
            dir_report = analyze_search_tree_directory(tree_dir)
            root = load_search_tree_root(tree_file)
            plot_path = Path(tmpdir) / "embedding_plot.png"
            written_plot = (
                plot_2d_embeddings(root, plot_path) if root is not None else None
            )

            self.assertIsNotNone(written_plot)
            self.assertTrue(plot_path.exists())

        self.assertEqual(file_report["shape"]["total_nodes"], 2)
        self.assertIn(0, file_report["radius_vs_depth"])
        self.assertEqual(dir_report["tree_count"], 1)
        self.assertEqual(dir_report["trees"][0]["shape"]["total_nodes"], 2)
