from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from loguru import logger


@dataclass(slots=True, eq=False)
class AnalysisNode:
    """Lightweight node used when loading serialized search trees."""

    state: Any = None
    action: Optional[str] = None
    visit_count: int = 0
    depth: int = 0
    encoder_features: Any = None
    analysis_embedding: Any = None
    children: list["AnalysisNode"] = field(default_factory=list)
    parents: list[tuple["AnalysisNode", Optional[str]]] = field(default_factory=list)


def _iter_children(node: Any) -> list[Any]:
    children = getattr(node, "children", None)
    if not children:
        return []
    return list(children)


def _to_tensor(features: Any) -> Optional[torch.Tensor]:
    raw = getattr(features, "tensor", features)
    if isinstance(raw, torch.Tensor):
        return raw
    try:
        tensor = torch.as_tensor(raw)
    except Exception:
        return None
    return tensor if isinstance(tensor, torch.Tensor) else None


def _tensor_to_jsonable(features: Any) -> Any:
    tensor = _to_tensor(features)
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def _node_embedding(node: Any) -> Optional[torch.Tensor]:
    embedding = _to_tensor(getattr(node, "analysis_embedding", None))
    if embedding is not None:
        return embedding
    return _to_tensor(getattr(node, "encoder_features", None))


def _walk_tree(root: Any):
    queue = deque([(root, 0)])
    visited: set[int] = set()

    while queue:
        node, depth = queue.popleft()
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)
        yield node, depth

        for child in _iter_children(node):
            queue.append((child, depth + 1))


def _ancestor_distances(node: Any) -> dict[int, int]:
    queue = deque([(node, 0)])
    distances: dict[int, int] = {}
    visited: set[int] = set()

    while queue:
        current, distance = queue.popleft()
        node_id = id(current)
        if node_id in visited:
            continue
        visited.add(node_id)

        previous = distances.get(node_id)
        if previous is None or distance < previous:
            distances[node_id] = distance

        parents = list(getattr(current, "parents", None) or [])
        if not parents:
            parent = getattr(current, "parent", None)
            if parent is not None:
                parents = [(parent, None)]

        for parent, _ in parents:
            if parent is not None:
                queue.append((parent, distance + 1))

    return distances


def _tree_distance(node_a: Any, node_b: Any) -> float:
    if node_a is node_b:
        return 0.0

    distances_a = _ancestor_distances(node_a)
    distances_b = _ancestor_distances(node_b)
    common_ancestors = set(distances_a).intersection(distances_b)
    if common_ancestors:
        return float(
            min(
                distances_a[node_id] + distances_b[node_id]
                for node_id in common_ancestors
            )
        )

    depth_a = getattr(node_a, "depth", 0)
    depth_b = getattr(node_b, "depth", 0)
    return float(abs(depth_a - depth_b))


def _rankdata(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(float)

    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]

    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and np.isclose(
            sorted_values[end], sorted_values[start]
        ):
            end += 1

        average_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = average_rank
        start = end

    return ranks


def _spearmanr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    if values_a.size < 2 or values_b.size < 2:
        return 0.0

    ranks_a = _rankdata(values_a.astype(float))
    ranks_b = _rankdata(values_b.astype(float))

    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()

    denominator = np.linalg.norm(ranks_a) * np.linalg.norm(ranks_b)
    if denominator == 0:
        return 0.0

    return float(np.dot(ranks_a, ranks_b) / denominator)


def _load_serialized_tree(tree_payload: Dict[str, Any]) -> Optional[AnalysisNode]:
    nodes_payload = tree_payload.get("nodes", {})
    if not nodes_payload:
        return None

    nodes: dict[str, AnalysisNode] = {}
    for node_key, node_data in nodes_payload.items():
        nodes[node_key] = AnalysisNode(
            state=node_data.get("state"),
            action=node_data.get("action"),
            visit_count=int(node_data.get("visit_count", 0) or 0),
            depth=int(node_data.get("depth", 0) or 0),
            encoder_features=_to_tensor(node_data.get("encoder_features")),
            analysis_embedding=_to_tensor(node_data.get("analysis_embedding")),
        )

    for node_key, node_data in nodes_payload.items():
        parent = nodes[node_key]
        for child_key in node_data.get("children", []) or []:
            child = nodes.get(child_key)
            if child is None:
                continue
            parent.children.append(child)
            child.parents.append((parent, child.action))

    root_key = tree_payload.get("root")
    if root_key in nodes:
        return nodes[root_key]

    return next(iter(nodes.values()), None)


def get_mcts_shape_analysis(root) -> Dict[str, float]:
    """
    Computes summary statistics for MCTS tree morphology.
    Returns average branching factor, maximum depth reached, and node visitation distribution stats.
    """
    if root is None:
        return {}

    depths = []
    branching_factors = []
    visitations = []

    for node, depth in _walk_tree(root):
        depths.append(depth)
        visitations.append(int(getattr(node, "visit_count", 0) or 0))

        children = _iter_children(node)
        if children:
            branching_factors.append(len(children))

    return {
        "max_depth": max(depths) if depths else 0,
        "avg_depth": sum(depths) / len(depths) if depths else 0,
        "avg_branching_factor": (
            sum(branching_factors) / len(branching_factors) if branching_factors else 0
        ),
        "max_branching_factor": max(branching_factors) if branching_factors else 0,
        "total_nodes": len(depths),
        "avg_visitation": sum(visitations) / len(visitations) if visitations else 0,
        "max_visitation": max(visitations) if visitations else 0,
    }


def get_embedding_radius_vs_depth(root) -> Dict[int, float]:
    """
    Tracks the average Euclidean norm (which correlates with hyperbolic radius)
    of encoder features against the MCTS node depth.
    """
    if root is None:
        return {}

    radius_by_depth = defaultdict(list)

    for node, depth in _walk_tree(root):
        features = _node_embedding(node)
        if features is not None:
            # Compute radius (L2 norm of the features)
            if features.ndim == 0:
                radius = float(features.abs().item())
            else:
                radius = features.norm(dim=-1).mean().item()
            radius_by_depth[depth].append(radius)

    avg_radius_by_depth = {
        depth: sum(radii) / len(radii) for depth, radii in radius_by_depth.items()
    }

    return dict(sorted(avg_radius_by_depth.items()))


def get_gromov_hyperbolicity_and_map(root, sample_size=50) -> Dict[str, float]:
    """
    Measures graph frustration and Gromov-hyperbolicity by sampling nodes from the MCTS tree.
    Approximates shortest path distances and compares them with embedding distances.
    """
    if root is None:
        return {}

    # Collect nodes that have features
    valid_nodes = []
    for node, depth in _walk_tree(root):
        features = _node_embedding(node)
        if features is not None:
            valid_nodes.append(node)

    if len(valid_nodes) < 4:
        return {"gromov_hyperbolicity": 0.0, "mAP_spearman": 0.0}

    # Sample nodes
    sampled = random.sample(valid_nodes, min(sample_size, len(valid_nodes)))
    n = len(sampled)

    tree_dists = np.zeros((n, n))
    emb_dists = np.zeros((n, n))

    for i in range(n):
        f_i = _node_embedding(sampled[i])
        if f_i is None:
            continue
        for j in range(i + 1, n):
            f_j = _node_embedding(sampled[j])
            if f_j is None:
                continue
            # Using Euclidean distance of the embeddings as a proxy for the latent space structure
            if f_i.ndim == 0 or f_j.ndim == 0:
                emb_dists[i, j] = emb_dists[j, i] = float(torch.norm(f_i - f_j).item())
            else:
                emb_dists[i, j] = emb_dists[j, i] = float(
                    torch.norm(f_i - f_j, dim=-1).mean().item()
                )

            tree_dists[i, j] = tree_dists[j, i] = _tree_distance(sampled[i], sampled[j])

    # Flatten upper triangles
    tree_flat = tree_dists[np.triu_indices(n, 1)]
    emb_flat = emb_dists[np.triu_indices(n, 1)]

    correlation = _spearmanr(tree_flat, emb_flat)

    # Gromov hyperbolicity (delta) calculation
    # delta = max(0.5 * (d(x,w) + d(y,z) - max(d(x,y)+d(z,w), d(x,z)+d(y,w))))
    max_delta = 0.0
    for _ in range(min(500, n**4)):
        try:
            x, y, z, w = random.sample(range(n), 4)
            d_xw_yz = emb_dists[x, w] + emb_dists[y, z]
            d_xy_zw = emb_dists[x, y] + emb_dists[z, w]
            d_xz_yw = emb_dists[x, z] + emb_dists[y, w]

            S = sorted([d_xw_yz, d_xy_zw, d_xz_yw])
            delta = 0.5 * (S[2] - S[1])
            max_delta = max(max_delta, delta)
        except ValueError:
            break

    return {
        "mAP_spearman": float(correlation) if not np.isnan(correlation) else 0.0,
        "gromov_hyperbolicity": float(max_delta),
    }


def plot_2d_embeddings(
    root: Any, output_path: Path, color_by: str = "depth"
) -> Optional[Path]:
    nodes: list[Any] = []
    coords: list[np.ndarray] = []
    depths: list[int] = []
    visits: list[int] = []

    for node, depth in _walk_tree(root):
        embedding = _node_embedding(node)
        if embedding is None:
            continue
        row = embedding.detach().cpu().reshape(-1).numpy().astype(float)
        if row.shape[0] != 2:
            continue
        nodes.append(node)
        coords.append(row)
        depths.append(depth)
        visits.append(int(getattr(node, "visit_count", 0) or 0))

    if not coords:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    points = np.asarray(coords, dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if color_by == "visits":
        colors = visits
        color_label = "visit count"
    else:
        colors = depths
        color_label = "depth"

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        cmap="viridis",
        s=28,
        alpha=0.85,
        edgecolors="none",
    )
    ax.set_title("2D learned value-head embedding")
    ax.set_xlabel("latent dim 1")
    ax.set_ylabel("latent dim 2")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(color_label)

    for node, (x, y) in zip(nodes[:20], points[:20]):
        label = getattr(node, "state", None)
        label_text = getattr(label, "pp", None) or str(label)
        if label_text:
            ax.annotate(str(label_text)[:60], (x, y), fontsize=7, alpha=0.65)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def analyze_search_tree_file(search_tree_file: Path) -> Dict[str, Any]:
    with open(search_tree_file, "r") as handle:
        tree_payload = json.load(handle)

    root = _load_serialized_tree(tree_payload)
    if root is None:
        return {
            "file": str(search_tree_file),
            "shape": {},
            "radius_vs_depth": {},
            "mAP_spearman": 0.0,
            "gromov_hyperbolicity": 0.0,
        }

    return {
        "file": str(search_tree_file),
        "shape": get_mcts_shape_analysis(root),
        "radius_vs_depth": get_embedding_radius_vs_depth(root),
        **get_gromov_hyperbolicity_and_map(root),
    }


def analyze_search_tree_directory(search_tree_dir: Path) -> Dict[str, Any]:
    tree_files = sorted(search_tree_dir.glob("search_tree_*.json"))
    if not tree_files:
        tree_files = sorted(search_tree_dir.rglob("search_tree_*.json"))
    if not tree_files and search_tree_dir.parent is not None:
        tree_files = sorted(search_tree_dir.parent.rglob("search_tree_*.json"))
    if not tree_files:
        return {
            "search_tree_dir": str(search_tree_dir),
            "tree_count": 0,
            "tree_files": [],
            "trees": [],
            "aggregate": {},
        }

    tree_reports = [analyze_search_tree_file(tree_file) for tree_file in tree_files]

    aggregate: Dict[str, Any] = {}
    numeric_values: dict[str, list[float]] = defaultdict(list)
    radius_by_depth: dict[str, list[float]] = defaultdict(list)

    for report in tree_reports:
        for key, value in report.get("shape", {}).items():
            numeric_values[f"shape.{key}"].append(float(value))

        for key in ("mAP_spearman", "gromov_hyperbolicity"):
            numeric_values[key].append(float(report.get(key, 0.0)))

        for depth, radius in report.get("radius_vs_depth", {}).items():
            radius_by_depth[str(depth)].append(float(radius))

    for key, values in numeric_values.items():
        if values:
            aggregate[key] = {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    if radius_by_depth:
        aggregate["radius_vs_depth"] = {
            depth: {
                "mean": float(np.mean(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for depth, values in sorted(
                radius_by_depth.items(), key=lambda item: int(item[0])
            )
        }

    return {
        "search_tree_dir": str(search_tree_dir),
        "tree_count": len(tree_reports),
        "tree_files": [str(tree_file) for tree_file in tree_files],
        "trees": tree_reports,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze serialized MCTS search trees emitted during training."
    )
    parser.add_argument(
        "--search-tree-dir",
        type=Path,
        required=True,
        help="Directory containing search_tree_*.json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the aggregated analysis JSON report.",
    )
    parser.add_argument(
        "--plot-2d",
        action="store_true",
        default=False,
        help="Also save a 2D plot from the learned value-head embeddings.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Where to save the optional 2D plot.",
    )
    args = parser.parse_args()

    report = analyze_search_tree_directory(args.search_tree_dir)

    output_path = args.output or (
        args.search_tree_dir.parent / "search_tree_analysis.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    if args.plot_2d:
        tree_files = sorted(args.search_tree_dir.glob("search_tree_*.json"))
        if tree_files:
            root = load_search_tree_root(tree_files[0])
            if root is not None:
                plot_path = args.plot_output or (
                    args.search_tree_dir.parent / "search_tree_embeddings_2d.png"
                )
                written = plot_2d_embeddings(root, plot_path)
                if written is not None:
                    logger.info(f"Saved 2D embedding plot to {written}")

    logger.info(f"Saved search tree analysis to {output_path}")


def load_search_tree_root(search_tree_file: Path) -> Optional[AnalysisNode]:
    with open(search_tree_file, "r") as handle:
        tree_payload = json.load(handle)
    return _load_serialized_tree(tree_payload)


if __name__ == "__main__":
    main()
