from __future__ import annotations

import argparse
import json
import math
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
    max_value: float = 0.0
    depth: int = 0
    encoder_features: Any = None
    analysis_embedding: Any = None
    terminal_type: Optional[str] = (
        None  # 'proof_finished', 'error', 'given_up', or None
    )
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
            max_value=float(node_data.get("max_value", 0.0) or 0.0),
            depth=int(node_data.get("depth", 0) or 0),
            encoder_features=_to_tensor(node_data.get("encoder_features")),
            analysis_embedding=_to_tensor(node_data.get("analysis_embedding")),
            terminal_type=node_data.get("terminal_type"),
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


def _poincare_dist_from_tangent(
    v1: torch.Tensor, v2: torch.Tensor, c: float = 1.0
) -> float:
    """Poincaré geodesic distance between two points given as tangent vectors.

    The ``analysis_embedding`` stored on nodes is ``logmap_0(h)`` — a tangent
    vector at the origin.  This function recovers the manifold points via
    ``h = expmap_0(v)`` and then applies the standard Poincaré ball formula::

        d_c(h1, h2) = (1/sqrt(c)) * acosh(1 + 2c*||h1-h2||^2
                                              / ((1-c*||h1||^2)*(1-c*||h2||^2)))
    """
    sqrt_c = math.sqrt(c)
    v1_flat = v1.reshape(-1).float()
    v2_flat = v2.reshape(-1).float()

    # Recover manifold points: h = tanh(sqrt(c)*||v||) / (sqrt(c)*||v||) * v
    norm1 = v1_flat.norm().clamp(min=1e-8)
    norm2 = v2_flat.norm().clamp(min=1e-8)
    h1 = torch.tanh(sqrt_c * norm1) / (sqrt_c * norm1) * v1_flat
    h2 = torch.tanh(sqrt_c * norm2) / (sqrt_c * norm2) * v2_flat

    # Clamp squared norms strictly inside the ball to avoid acosh of <1
    c_norm1_sq = (c * h1.norm() ** 2).clamp(max=1.0 - 1e-6)
    c_norm2_sq = (c * h2.norm() ** 2).clamp(max=1.0 - 1e-6)

    diff_sq = (h1 - h2).norm() ** 2
    inner = 1.0 + 2.0 * c * diff_sq / ((1.0 - c_norm1_sq) * (1.0 - c_norm2_sq))
    inner = inner.clamp(min=1.0 + 1e-8)
    return float((torch.acosh(inner) / sqrt_c).item())


def _tangent_to_disk(v: torch.Tensor, c: float = 1.0) -> np.ndarray:
    """Map a 2D tangent vector (logmap_0 output) to Poincaré disk coordinates.

    Returns a 2-element float array representing the point on the disk.
    """
    sqrt_c = math.sqrt(c)
    v_flat = v.reshape(-1).float()
    norm = v_flat.norm().clamp(min=1e-8)
    h = torch.tanh(sqrt_c * norm) / (sqrt_c * norm) * v_flat
    result: np.ndarray = h.detach().cpu().numpy().astype(float)
    return result


def get_gromov_hyperbolicity_and_map(
    root,
    sample_size: int = 50,
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> Dict[str, float]:
    """
    Measures graph frustration and Gromov-hyperbolicity by sampling nodes from
    the MCTS tree.  Approximates shortest path distances and compares them with
    embedding distances.

    When ``is_hyperbolic=True`` the Poincaré geodesic distance is used instead
    of Euclidean distance so that the δ-hyperbolicity estimate is meaningful for
    Poincaré-ball value heads.
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

            if is_hyperbolic:
                dist = _poincare_dist_from_tangent(f_i, f_j, curvature)
            elif f_i.ndim == 0 or f_j.ndim == 0:
                dist = float(torch.norm(f_i - f_j).item())
            else:
                dist = float(torch.norm(f_i - f_j, dim=-1).mean().item())

            emb_dists[i, j] = emb_dists[j, i] = dist
            tree_dists[i, j] = tree_dists[j, i] = _tree_distance(sampled[i], sampled[j])

    # Flatten upper triangles
    tree_flat = tree_dists[np.triu_indices(n, 1)]
    emb_flat = emb_dists[np.triu_indices(n, 1)]

    correlation = _spearmanr(tree_flat, emb_flat)

    # Gromov δ-hyperbolicity: δ = 0.5 * (S[2] - S[1]) for the sorted triple
    # of summed pairwise distances over all quadruples of sampled nodes.
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


def _node_outcome(node: Any) -> str:
    """Classify a node as 'proof', 'error', or 'unknown' from serialized fields.

    Uses the ``terminal_type`` field emitted by ``_serialize_node``, falling
    back to ``max_value`` heuristics for older tree files that lack it.
    """
    terminal_type = getattr(node, "terminal_type", None)
    if terminal_type == "proof_finished":
        return "proof"
    if terminal_type in ("error", "given_up"):
        return "error"
    # Fallback: inspect the serialized state string and max_value
    state = getattr(node, "state", None)
    if state is None:
        # Terminal node — no pp string stored; infer from max_value
        max_val = float(getattr(node, "max_value", 0.0) or 0.0)
        return "proof" if max_val >= 1.0 else "error"
    max_val = float(getattr(node, "max_value", 0.0) or 0.0)
    if max_val >= 1.0:
        return "proof"
    if max_val <= -0.9:
        return "error"
    return "unknown"


def plot_2d_embeddings(
    root: Any,
    output_path: Path,
    color_by: str = "depth",
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
    draw_edges: bool = False,
    draw_entailment_cones: bool = False,
) -> Optional[Path]:
    """Plot 2-D value-head embeddings.

    Args:
        root: Root node of the MCTS tree (in-memory or deserialized).
        output_path: Where to write the PNG.
        color_by: ``'depth'``, ``'visits'``, or ``'outcome'``.
        is_hyperbolic: When True, transforms tangent-space coordinates to
            Poincaré disk coordinates and draws the disk boundary.
        curvature: Curvature parameter *c* (> 0) used by the Poincaré ball.
        draw_edges: When True, draw parent→child arrows between embedded nodes.
        draw_entailment_cones: When True, draw the half-angle sector of the
            entailment cone centred at each non-leaf node (the angle is defined
            by the maximum angular spread of the node's children in the
            embedding space).
    """
    nodes: list[Any] = []
    coords: list[np.ndarray] = []
    depths: list[int] = []
    visits: list[int] = []
    outcomes: list[str] = []
    # Map node identity → (index, coord) for edge drawing
    node_to_idx: Dict[int, int] = {}

    for node, depth in _walk_tree(root):
        embedding = _node_embedding(node)
        if embedding is None:
            continue
        raw = embedding.detach().cpu().reshape(-1).numpy().astype(float)
        if raw.shape[0] != 2:
            continue
        if is_hyperbolic:
            row = _tangent_to_disk(torch.as_tensor(raw), curvature)
        else:
            row = raw
        node_to_idx[id(node)] = len(nodes)
        nodes.append(node)
        coords.append(row)
        depths.append(depth)
        visits.append(int(getattr(node, "visit_count", 0) or 0))
        outcomes.append(_node_outcome(node))

    if not coords:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    points = np.asarray(coords, dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # ── Disk boundary ─────────────────────────────────────────────────────────
    if is_hyperbolic:
        radius = 1.0 / math.sqrt(curvature)
        disk_circle = plt.Circle(
            (0, 0), radius, color="black", fill=False, linewidth=1.5, linestyle="--"
        )
        ax.add_patch(disk_circle)
        ax.set_xlim(-radius * 1.05, radius * 1.05)
        ax.set_ylim(-radius * 1.05, radius * 1.05)
        ax.set_aspect("equal")

    # ── Tree edges ────────────────────────────────────────────────────────────
    if draw_edges:
        for node in nodes:
            parent_idx = node_to_idx.get(id(node))
            if parent_idx is None:
                continue
            px, py = points[parent_idx]
            for child in _iter_children(node):
                child_idx = node_to_idx.get(id(child))
                if child_idx is None:
                    continue
                cx, cy = points[child_idx]
                ax.annotate(
                    "",
                    xy=(cx, cy),
                    xytext=(px, py),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        color="steelblue",
                        alpha=0.35,
                        linewidth=0.7,
                        shrinkA=3,
                        shrinkB=3,
                    ),
                )

    # ── Entailment cones ──────────────────────────────────────────────────────
    if draw_entailment_cones:
        import matplotlib.patches as pat

        for node, pt in zip(nodes, points):
            children_pts = []
            for child in _iter_children(node):
                cidx = node_to_idx.get(id(child))
                if cidx is not None:
                    children_pts.append(points[cidx])
            if len(children_pts) < 2:
                continue
            # Direction vectors from parent to each child in embedding space
            vecs = [cp - pt for cp in children_pts]
            angles = [math.degrees(math.atan2(v[1], v[0])) for v in vecs]
            min_angle = min(angles)
            max_angle = max(angles)
            span = max_angle - min_angle
            if span > 180:
                # Wrap around case — take the complement
                min_angle, max_angle = max_angle, min_angle + 360
                span = 360 - span
            # Draw a wedge sector showing the angular spread of children
            mean_dist = float(np.mean([float(np.linalg.norm(v)) for v in vecs]))
            wedge = pat.Wedge(
                center=(pt[0], pt[1]),
                r=mean_dist * 0.8,
                theta1=min_angle,
                theta2=max_angle,
                facecolor="gold",
                alpha=0.15,
                edgecolor="darkorange",
                linewidth=0.6,
            )
            ax.add_patch(wedge)

    # ── Scatter ───────────────────────────────────────────────────────────────
    if color_by == "outcome":
        _outcome_colors = {"proof": "#2ca02c", "error": "#d62728", "unknown": "#aec7e8"}
        scatter_colors = [_outcome_colors.get(o, "#aec7e8") for o in outcomes]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=scatter_colors,
            s=28,
            alpha=0.85,
            edgecolors="none",
        )
        legend_patches = [
            mpatches.Patch(color="#2ca02c", label="proof"),
            mpatches.Patch(color="#d62728", label="error"),
            mpatches.Patch(color="#aec7e8", label="unknown"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    elif color_by == "visits":
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=visits,
            cmap="viridis",
            s=28,
            alpha=0.85,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax).set_label("visit count")
    else:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=depths,
            cmap="viridis",
            s=28,
            alpha=0.85,
            edgecolors="none",
        )
        fig.colorbar(scatter, ax=ax).set_label("depth")

    title = (
        "2D Poincaré disk embedding"
        if is_hyperbolic
        else "2D learned value-head embedding"
    )
    ax.set_title(title)
    ax.set_xlabel("latent dim 1")
    ax.set_ylabel("latent dim 2")

    for node, (x, y) in zip(nodes[:20], points[:20]):
        label = getattr(node, "state", None)
        label_text = getattr(label, "pp", None) or str(label)
        if label_text:
            ax.annotate(str(label_text)[:60], (x, y), fontsize=7, alpha=0.65)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def analyze_search_tree_file(
    search_tree_file: Path,
    is_hyperbolic: Optional[bool] = None,
    curvature: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyse one serialised MCTS search-tree file.

    ``is_hyperbolic`` and ``curvature`` are auto-detected from the file's
    ``metadata`` block when present; explicit arguments take precedence.
    """
    with open(search_tree_file, "r") as handle:
        tree_payload = json.load(handle)

    # Read geometry metadata emitted by MCTS_AlphaZero._save_search_tree
    meta = tree_payload.get("metadata", {})
    _is_hyp = (
        is_hyperbolic
        if is_hyperbolic is not None
        else bool(meta.get("is_hyperbolic", False))
    )
    _curv = curvature if curvature is not None else float(meta.get("curvature", 1.0))

    root = _load_serialized_tree(tree_payload)
    if root is None:
        return {
            "file": str(search_tree_file),
            "is_hyperbolic": _is_hyp,
            "curvature": _curv,
            "shape": {},
            "radius_vs_depth": {},
            "mAP_spearman": 0.0,
            "gromov_hyperbolicity": 0.0,
        }

    return {
        "file": str(search_tree_file),
        "is_hyperbolic": _is_hyp,
        "curvature": _curv,
        "shape": get_mcts_shape_analysis(root),
        "radius_vs_depth": get_embedding_radius_vs_depth(root),
        **get_gromov_hyperbolicity_and_map(
            root, is_hyperbolic=_is_hyp, curvature=_curv
        ),
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
    parser.add_argument(
        "--color-by",
        choices=["depth", "visits", "outcome"],
        default="outcome",
        help="How to colour nodes in the 2D embedding plot (default: outcome).",
    )
    parser.add_argument(
        "--draw-edges",
        action="store_true",
        default=False,
        help="Draw parent→child arrows in the 2D embedding plot.",
    )
    parser.add_argument(
        "--draw-entailment-cones",
        action="store_true",
        default=False,
        help="Draw entailment cone wedge sectors around each non-leaf node.",
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
            # Detect geometry from first tree file's metadata
            with open(tree_files[0]) as _f:
                _meta_payload = json.load(_f)
            _meta = _meta_payload.get("metadata", {})
            _is_hyp = bool(_meta.get("is_hyperbolic", False))
            _curv = float(_meta.get("curvature", 1.0))

            root = _load_serialized_tree(_meta_payload)
            if root is not None:
                plot_path = args.plot_output or (
                    args.search_tree_dir.parent / "search_tree_embeddings_2d.png"
                )
                written = plot_2d_embeddings(
                    root,
                    plot_path,
                    color_by=args.color_by,
                    is_hyperbolic=_is_hyp,
                    curvature=_curv,
                    draw_edges=args.draw_edges,
                    draw_entailment_cones=args.draw_entailment_cones,
                )
                if written is not None:
                    logger.info(f"Saved 2D embedding plot to {written}")

    logger.info(f"Saved search tree analysis to {output_path}")


def load_search_tree_root(search_tree_file: Path) -> Optional[AnalysisNode]:
    with open(search_tree_file, "r") as handle:
        tree_payload = json.load(handle)
    return _load_serialized_tree(tree_payload)


if __name__ == "__main__":
    main()
