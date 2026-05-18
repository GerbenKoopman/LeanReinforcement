from __future__ import annotations

# flake8: noqa

import argparse
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, cast

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
        disk_circle = mpatches.Circle(
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


# ---------------------------------------------------------------------------
# Experiment 3 — value calibration curves
# ---------------------------------------------------------------------------

_N_CALIB_BINS: int = 10


def get_value_calibration_data(root: Any) -> Dict[str, Any]:
    """Compute calibration data comparing predicted max_value to node outcome.

    Buckets predicted values in [0, 1] into ``_N_CALIB_BINS`` equal-width bins
    and computes the fraction of nodes in each bin that were eventually proven
    (i.e. reached a ``proof_finished`` terminal).  Returns bin centres, mean
    predicted values per bin, fraction proven per bin, and the Expected
    Calibration Error (ECE).
    """
    predictions: list[float] = []
    labels: list[int] = []  # 1 = proven, 0 = not

    for node, _ in _walk_tree(root):
        pred = float(getattr(node, "max_value", 0.0) or 0.0)
        # Clamp to [0, 1] — value heads are trained with MSE against {0, 1}
        pred = max(0.0, min(1.0, (pred + 1.0) / 2.0))
        label = 1 if _node_outcome(node) == "proof" else 0
        predictions.append(pred)
        labels.append(label)

    if not predictions:
        return {"bins": [], "mean_pred": [], "fraction_proven": [], "ece": 0.0}

    preds = np.array(predictions)
    lbls = np.array(labels, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, _N_CALIB_BINS + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mean_pred: list[float] = []
    frac_proven: list[float] = []
    bin_counts: list[int] = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if not mask.any():
            mean_pred.append(float(0.5 * (lo + hi)))
            frac_proven.append(0.0)
            bin_counts.append(0)
        else:
            mean_pred.append(float(preds[mask].mean()))
            frac_proven.append(float(lbls[mask].mean()))
            bin_counts.append(int(mask.sum()))

    total = len(predictions)
    ece = float(
        sum(
            count / total * abs(mp - fp)
            for count, mp, fp in zip(bin_counts, mean_pred, frac_proven)
        )
    )

    return {
        "bins": bin_centres.tolist(),
        "mean_pred": mean_pred,
        "fraction_proven": frac_proven,
        "bin_counts": bin_counts,
        "ece": ece,
    }


def plot_calibration_curve(root: Any, output_path: Path) -> Optional[Path]:
    """Plot the reliability diagram (calibration curve) for one search tree."""
    data = get_value_calibration_data(root)
    if not data["bins"]:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    bins = data["bins"]
    mean_pred = data["mean_pred"]
    frac_proven = data["fraction_proven"]
    ece = data["ece"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="perfect calibration")
    ax.bar(
        bins,
        frac_proven,
        width=1.0 / _N_CALIB_BINS,
        align="center",
        alpha=0.6,
        color="steelblue",
        label="fraction proven",
    )
    ax.plot(mean_pred, frac_proven, "ro-", markersize=5, label="mean predicted value")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted value (normalised)")
    ax.set_ylabel("fraction of nodes proven")
    ax.set_title(f"Value calibration curve  (ECE = {ece:.3f})")
    ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Experiment 4 — Poincaré radius vs node outcome
# ---------------------------------------------------------------------------


def get_radius_by_outcome(
    root: Any,
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> Dict[str, list]:
    """Collect Poincaré radii (or L2 norms for Euclidean) grouped by outcome.

    Returns a dict with keys ``'proof'``, ``'error'``, ``'unknown'``, each
    containing a list of float radii for all nodes in that outcome category
    that have a 2-D embedding.
    """
    radii_by_outcome: Dict[str, list] = {"proof": [], "error": [], "unknown": []}

    for node, _ in _walk_tree(root):
        embedding = _node_embedding(node)
        if embedding is None:
            continue
        raw = embedding.detach().cpu().reshape(-1).numpy().astype(float)
        if raw.shape[0] != 2:
            continue

        if is_hyperbolic:
            disk_pt = _tangent_to_disk(torch.as_tensor(raw), curvature)
            radius = float(np.linalg.norm(disk_pt))
        else:
            radius = float(np.linalg.norm(raw))

        outcome = _node_outcome(node)
        radii_by_outcome[outcome].append(radius)

    return radii_by_outcome


def plot_radius_by_outcome(
    root: Any,
    output_path: Path,
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> Optional[Path]:
    """Violin/box plot of embedding radii stratified by node outcome."""
    data = get_radius_by_outcome(root, is_hyperbolic=is_hyperbolic, curvature=curvature)
    # Keep only outcomes that have data
    outcomes_present = [k for k in ("proof", "error", "unknown") if data[k]]
    if not outcomes_present:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    outcome_colors = {"proof": "#2ca02c", "error": "#d62728", "unknown": "#aec7e8"}
    parts = ax.violinplot(
        [data[k] for k in outcomes_present],
        positions=range(len(outcomes_present)),
        showmedians=True,
        showextrema=True,
    )
    bodies: Any = parts["bodies"]  # type: ignore[index]
    for pc, outcome in zip(bodies, outcomes_present):
        pc.set_facecolor(outcome_colors[outcome])
        pc.set_alpha(0.6)

    ax.set_xticks(range(len(outcomes_present)))
    ax.set_xticklabels(outcomes_present)
    ax.set_xlabel("node outcome")
    ylabel = "Poincaré radius" if is_hyperbolic else "L2 norm (embedding)"
    ax.set_ylabel(ylabel)
    title = (
        f"Poincaré radius by outcome  (c={curvature})"
        if is_hyperbolic
        else "Embedding norm by outcome"
    )
    ax.set_title(title)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Experiment 6 — geodesic proof-path visualisation
# ---------------------------------------------------------------------------


def _geodesic_arc_points(
    p1: np.ndarray,
    p2: np.ndarray,
    c: float = 1.0,
    n: int = 80,
) -> np.ndarray:
    """Return ``n`` points along the Poincaré-disk geodesic from *p1* to *p2*.

    In the Poincaré disk with curvature parameter *c* (boundary radius
    ``R = 1/√c``) geodesics are arcs of circles orthogonal to the boundary
    circle, or diameters when *p1*, *p2*, and the origin are collinear.
    Falls back to a straight line for the Euclidean / degenerate case.
    """
    R = 1.0 / math.sqrt(c)
    m = (p1 + p2) / 2.0
    d = p2 - p1
    # Perpendicular direction (left-normal of d)
    n_vec = np.array([-d[1], d[0]], dtype=float)
    mn = float(m @ n_vec)
    if abs(mn) < 1e-9:
        # Collinear with origin — straight line is the geodesic
        return np.linspace(p1, p2, n)
    # Center of the circle orthogonal to the boundary disk
    # Derivation: |O|² = |O - p1|² + R²  →  t = (|d/2|² + R² - |m|²) / (2·(m·n_vec))
    t = (float(d @ d) / 4.0 + R**2 - float(m @ m)) / (2.0 * mn)
    center = m + t * n_vec
    r = float(np.linalg.norm(p1 - center))
    theta1 = math.atan2(float(p1[1] - center[1]), float(p1[0] - center[0]))
    theta2 = math.atan2(float(p2[1] - center[1]), float(p2[0] - center[0]))
    # Choose the shorter arc
    diff = (theta2 - theta1 + math.pi) % (2 * math.pi) - math.pi
    thetas = np.linspace(0.0, diff, n)
    xs = center[0] + r * np.cos(theta1 + thetas)
    ys = center[1] + r * np.sin(theta1 + thetas)
    return np.column_stack([xs, ys])


def _find_proof_paths(root: AnalysisNode) -> list[list[AnalysisNode]]:
    """DFS to collect all root→proof_finished paths in the search tree."""
    paths: list[list[AnalysisNode]] = []

    def _dfs(node: AnalysisNode, path: list[AnalysisNode]) -> None:
        path = path + [node]
        if node.terminal_type == "proof_finished":
            paths.append(path)
            return
        for child in node.children:
            _dfs(child, path)

    _dfs(root, [])
    return paths


def plot_geodesic_proof_paths(
    root: AnalysisNode,
    output_path: Path,
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> Optional[Path]:
    """Draw geodesic arcs along all root→proof_finished paths on the Poincaré disk.

    Each proof path is rendered in a distinct colour as a chain of geodesic
    segments between consecutive nodes with 2D embeddings.  Circle markers
    indicate the root end of each path; star markers indicate the proof node.
    Returns ``None`` when no proof paths exist or no nodes have 2D embeddings.
    """
    paths = _find_proof_paths(root)
    if not paths:
        logger.info("No proof-finished paths found; skipping geodesic path plot.")
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(6, 6))
    if is_hyperbolic:
        R = 1.0 / math.sqrt(curvature)
        disk = mpatches.Circle((0, 0), R, fill=False, color="black", linewidth=1.5)
        ax.add_patch(disk)
        ax.set_xlim(-R * 1.05, R * 1.05)
        ax.set_ylim(-R * 1.05, R * 1.05)
    else:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

    cmap = plt.cm.get_cmap("tab10", max(len(paths), 1))
    any_plotted = False
    for path_idx, path in enumerate(paths):
        color = cmap(path_idx % 10)
        coords: list[np.ndarray] = []
        for node in path:
            if (
                node.analysis_embedding is not None
                and len(node.analysis_embedding) == 2
            ):
                raw = node.analysis_embedding
                if is_hyperbolic:
                    pt = _tangent_to_disk(torch.as_tensor(raw), curvature)
                else:
                    pt = np.array(raw, dtype=float)
                coords.append(pt)
        if len(coords) < 2:
            continue
        any_plotted = True
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            if is_hyperbolic:
                arc = _geodesic_arc_points(p1, p2, c=curvature, n=80)
            else:
                arc = np.array([p1, p2])
            ax.plot(arc[:, 0], arc[:, 1], color=color, lw=1.2, alpha=0.75)
        ax.scatter(
            [coords[0][0]], [coords[0][1]], color=color, s=40, marker="o", zorder=5
        )
        ax.scatter(
            [coords[-1][0]], [coords[-1][1]], color=color, s=80, marker="*", zorder=5
        )

    if not any_plotted:
        plt.close(fig)
        logger.info("No 2D embeddings on proof paths; skipping geodesic path plot.")
        return None

    geometry_label = "Poincaré disk" if is_hyperbolic else "Euclidean plane"
    ax.set_title(f"Geodesic proof paths — {geometry_label}")
    ax.set_aspect("equal")
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Experiment 7 — entailment cone angle distribution
# ---------------------------------------------------------------------------


def get_entailment_cone_angles(
    root: AnalysisNode,
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> list[float]:
    """Compute cone half-angles for every parent→child pair with 2D embeddings.

    For each pair the angle is measured between the *parent→child* direction
    and the *parent→origin* direction (i.e. the inward direction in hyperbolic
    space).  A small angle means the child lies 'in front of' its parent
    relative to the origin — deeper in the Poincaré hierarchy.

    Returns a list of angles in degrees.
    """
    angles: list[float] = []

    def _recurse(node: AnalysisNode) -> None:
        if node.analysis_embedding is None or len(node.analysis_embedding) != 2:
            for child in node.children:
                _recurse(child)
            return

        raw_parent = node.analysis_embedding
        if is_hyperbolic:
            p_parent = _tangent_to_disk(torch.as_tensor(raw_parent), curvature)
        else:
            p_parent = np.array(raw_parent, dtype=float)

        parent_norm = float(np.linalg.norm(p_parent))
        if parent_norm < 1e-9:
            for child in node.children:
                _recurse(child)
            return

        toward_origin = -p_parent  # direction from parent toward the origin
        for child in node.children:
            if child.analysis_embedding is None or len(child.analysis_embedding) != 2:
                _recurse(child)
                continue
            raw_child = child.analysis_embedding
            if is_hyperbolic:
                p_child = _tangent_to_disk(torch.as_tensor(raw_child), curvature)
            else:
                p_child = np.array(raw_child, dtype=float)

            direction = p_child - p_parent
            dir_norm = float(np.linalg.norm(direction))
            if dir_norm < 1e-9:
                _recurse(child)
                continue
            cos_theta = float(np.dot(direction, toward_origin)) / (
                dir_norm * parent_norm
            )
            cos_theta = max(-1.0, min(1.0, cos_theta))
            angles.append(math.degrees(math.acos(cos_theta)))
            _recurse(child)

    _recurse(root)
    return angles


def plot_entailment_cone_distribution(
    root: AnalysisNode,
    output_path: Path,
    is_hyperbolic: bool = False,
    curvature: float = 1.0,
) -> Optional[Path]:
    """Histogram of parent→child entailment cone half-angles.

    In a well-structured hyperbolic hierarchy the distribution should be
    concentrated near 0° (children deeper than parents) rather than uniform
    across 0–180°.  Returns ``None`` when no angle data is available.
    """
    angles = get_entailment_cone_angles(
        root, is_hyperbolic=is_hyperbolic, curvature=curvature
    )
    if not angles:
        logger.info("No entailment cone angle data; skipping distribution plot.")
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(angles, bins=36, range=(0.0, 180.0), edgecolor="white", color="steelblue")
    ax.axvline(
        float(np.mean(angles)),
        color="red",
        linestyle="--",
        label=f"Mean {float(np.mean(angles)):.1f}°",
    )
    ax.set_xlabel("Cone half-angle (degrees)")
    ax.set_ylabel("Count")
    geometry_label = "Poincaré disk" if is_hyperbolic else "Euclidean plane"
    ax.set_title(f"Entailment cone angle distribution — {geometry_label}")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _extract_epoch_from_filename(path: Path, prefix: str) -> Optional[int]:
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix) :]
    if not suffix.isdigit():
        return None
    return int(suffix)


def _load_json(path: Path) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def get_run_artifact_stats(run_dir: Path) -> Dict[str, Any]:
    """Collect simple per-run stats from theorem and training-data artifacts.

    This is a lightweight fallback when ``search_trees/`` is missing due to
    interrupted runs (e.g. OOM). It summarizes theorem success by epoch and key
    aggregates from ``training_data_epoch_*.json``.
    """
    theorem_files = sorted(run_dir.glob("theorem_results_epoch_*.json"))
    training_files = sorted(run_dir.glob("training_data_epoch_*.json"))

    theorem_by_epoch: list[Dict[str, Any]] = []
    for file in theorem_files:
        payload = _load_json(file)
        epoch = _extract_epoch_from_filename(file, "theorem_results_epoch_")
        if epoch is None:
            epoch = int(payload.get("epoch", 0) or 0)
        theorem_by_epoch.append(
            {
                "epoch": epoch,
                "total": int(payload.get("total", 0) or 0),
                "proved": int(payload.get("proved", 0) or 0),
                "failed": int(payload.get("failed", 0) or 0),
                "success_rate": float(payload.get("success_rate", 0.0) or 0.0),
            }
        )
    theorem_by_epoch.sort(key=lambda item: int(item["epoch"]))

    value_targets: list[float] = []
    mcts_values: list[float] = []
    visit_counts: list[int] = []
    training_rows = 0
    unique_states: set[str] = set()
    for file in training_files:
        rows = _load_json(file)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            training_rows += 1
            state = row.get("state")
            if isinstance(state, str):
                unique_states.add(state)
            if isinstance(row.get("value_target"), (int, float)):
                value_targets.append(float(row["value_target"]))
            if isinstance(row.get("mcts_value"), (int, float)):
                mcts_values.append(float(row["mcts_value"]))
            if isinstance(row.get("visit_count"), int):
                visit_counts.append(int(row["visit_count"]))

    def _summary(values: list[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        arr = np.asarray(values, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "run_dir": str(run_dir),
        "theorem_epoch_count": len(theorem_by_epoch),
        "theorem_by_epoch": theorem_by_epoch,
        "training_epoch_count": len(training_files),
        "training_rows": training_rows,
        "unique_states": len(unique_states),
        "value_target": _summary(value_targets),
        "mcts_value": _summary(mcts_values),
        "visit_count": _summary([float(v) for v in visit_counts]),
    }


def plot_theorem_success_curve(run_dir: Path, output_path: Path) -> Optional[Path]:
    """Plot theorem success-rate trajectory across epochs for one run."""
    theorem_files = sorted(run_dir.glob("theorem_results_epoch_*.json"))
    if not theorem_files:
        return None

    epochs: list[int] = []
    success: list[float] = []
    proved: list[int] = []
    totals: list[int] = []

    for file in theorem_files:
        payload = _load_json(file)
        epoch = _extract_epoch_from_filename(file, "theorem_results_epoch_")
        if epoch is None:
            epoch = int(payload.get("epoch", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        proved_count = int(payload.get("proved", 0) or 0)
        sr = float(payload.get("success_rate", 0.0) or 0.0)
        if sr <= 1.0:
            sr *= 100.0
        epochs.append(epoch)
        success.append(sr)
        proved.append(proved_count)
        totals.append(total)

    if not epochs:
        return None

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    ax1.plot(epochs, success, "o-", color="tab:blue", label="success rate (%)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("success rate (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, proved, "s--", color="tab:green", label="proved")
    ax2.plot(epochs, totals, "x:", color="tab:red", label="total")
    ax2.set_ylabel("count", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best", fontsize=8)
    ax1.set_title("Theorem outcomes across epochs")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _compute_2d_projection(points: np.ndarray) -> np.ndarray:
    """Simple PCA-to-2D projection without external dependencies."""
    if points.ndim != 2:
        raise ValueError("points must be a 2-D array")
    n, d = points.shape
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if d == 2:
        return points.astype(float)
    centred = points - np.mean(points, axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:2].T
    projected: np.ndarray = centred @ components
    return projected


def _collect_state_records(run_dir: Path) -> list[tuple[str, float]]:
    records: list[tuple[str, float]] = []
    for file in sorted(run_dir.glob("training_data_epoch_*.json")):
        rows = _load_json(file)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            state = row.get("state")
            if not isinstance(state, str) or not state.strip():
                continue
            target = row.get("value_target", 0.0)
            target_f = float(target) if isinstance(target, (int, float)) else 0.0
            records.append((state, target_f))
    return records


def plot_checkpoint_state_embeddings(
    run_dir: Path,
    output_path: Path,
    sample_size: int = 600,
) -> Optional[Path]:
    """Plot theorem-state embeddings from the saved value-head checkpoint.

    Uses ``training_data_epoch_*.json`` states and the run's latest checkpoint.
    For hyperbolic heads with 2-D latent vectors, points are mapped to the
    Poincaré disk; otherwise a simple PCA-to-2D projection is used.
    """
    config_file = run_dir / "training_config.json"
    if not config_file.exists():
        logger.warning(f"No training config found at {config_file}; skipping plot")
        return None

    records = _collect_state_records(run_dir)
    if not records:
        logger.warning(f"No training_data_epoch_*.json states found under {run_dir}")
        return None

    unique: dict[str, float] = {}
    for state, target in records:
        if state not in unique:
            unique[state] = target
    state_target_pairs = list(unique.items())
    if len(state_target_pairs) > sample_size:
        state_target_pairs = random.sample(state_target_pairs, sample_size)
    states = [state for state, _ in state_target_pairs]
    targets = np.asarray([target for _, target in state_target_pairs], dtype=float)

    cfg = _load_json(config_file)
    model_name = str(cfg.get("model_name", "kaiyuy/leandojo-lean4-tacgen-byt5-small"))
    latent_dim = int(cfg.get("value_head_latent_dim", 1024) or 1024)
    hidden_layers = int(cfg.get("value_head_hidden_layers", 1) or 1)
    use_hyperbolic = bool(cfg.get("use_hyperbolic", False))
    curvature = float(cfg.get("curvature", 1.0) or 1.0)
    mcts_type = str(cfg.get("mcts_type", "alpha_zero"))

    from lean_reinforcement.agent.transformer import Transformer
    from lean_reinforcement.agent.value_head.hyperbolic_value_head import (
        HyperbolicValueHead,
    )
    from lean_reinforcement.agent.value_head.value_head import ValueHead
    from lean_reinforcement.utilities.checkpoint import load_checkpoint

    transformer = Transformer(model_name=model_name)
    if use_hyperbolic:
        value_head: Any = HyperbolicValueHead(
            transformer,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
            curvature=curvature,
        )
    else:
        value_head = ValueHead(
            transformer,
            latent_dim=latent_dim,
            hidden_layers=hidden_layers,
        )
    load_checkpoint(value_head, run_dir, prefix=f"value_head_{mcts_type}")
    # Keep extraction on CPU to avoid GPU OOM on local/dev machines.
    value_head.to("cpu")

    def _encode_states_cpu(batch_states: list[str]) -> torch.Tensor:
        tokenized = value_head.tokenizer(
            batch_states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        hidden_state = cast(
            torch.Tensor,
            value_head.encoder(tokenized.input_ids).last_hidden_state,
        )
        lens = tokenized.attention_mask.sum(dim=1)
        attn_mask = tokenized.attention_mask.unsqueeze(2)
        features = (hidden_state * attn_mask).sum(dim=1) / lens.unsqueeze(1)
        return cast(torch.Tensor, features.detach())

    latents_batches: list[np.ndarray] = []
    batch_size = 16
    for start in range(0, len(states), batch_size):
        batch = states[start : start + batch_size]
        features = _encode_states_cpu(batch)
        latent = (
            value_head.latent_from_features(features)
            .detach()
            .cpu()
            .numpy()
            .astype(float)
        )
        latents_batches.append(latent)
    latents = np.concatenate(latents_batches, axis=0) if latents_batches else None
    if latents is None or latents.size == 0:
        return None

    if use_hyperbolic and latents.shape[1] == 2:
        coords = np.asarray(
            [_tangent_to_disk(torch.as_tensor(row), curvature) for row in latents],
            dtype=float,
        )
        in_disk = True
    else:
        coords = _compute_2d_projection(latents)
        in_disk = False

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=targets,
        cmap="coolwarm",
        alpha=0.8,
        s=20,
        edgecolors="none",
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("value_target")

    if in_disk:
        radius = 1.0 / math.sqrt(curvature)
        disk = mpatches.Circle((0, 0), radius, fill=False, color="black", linewidth=1.3)
        ax.add_patch(disk)
        ax.set_xlim(-radius * 1.05, radius * 1.05)
        ax.set_ylim(-radius * 1.05, radius * 1.05)
        ax.set_aspect("equal")
        title = "Hyperbolic theorem-state embeddings " f"(Poincaré disk, c={curvature})"
    else:
        title = "Euclidean theorem-state embeddings (2D projection)"

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    fig.savefig(output_path, dpi=180)
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
        "calibration": get_value_calibration_data(root),
        "radius_by_outcome": {
            k: v
            for k, v in get_radius_by_outcome(
                root, is_hyperbolic=_is_hyp, curvature=_curv
            ).items()
        },
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
        description=("Analyze serialized MCTS search trees emitted during training.")
    )
    parser.add_argument(
        "--search-tree-dir",
        type=Path,
        required=False,
        default=None,
        help="Directory containing search_tree_*.json files.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing theorem_results_epoch_*.json / "
            "training_data_epoch_*.json. Used as fallback when search trees "
            "are missing."
        ),
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
        help=("How to colour nodes in the 2D embedding plot " "(default: outcome)."),
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
    parser.add_argument(
        "--plot-calibration",
        action="store_true",
        default=False,
        help="Save a value calibration curve (reliability diagram).",
    )
    parser.add_argument(
        "--plot-radius-by-outcome",
        action="store_true",
        default=False,
        help="Save a violin plot of embedding radii stratified by node outcome.",
    )
    parser.add_argument(
        "--plot-geodesic-paths",
        action="store_true",
        default=False,
        help="Save a plot of geodesic arcs along root→proof paths on the Poincaré disk.",
    )
    parser.add_argument(
        "--plot-entailment-cone-dist",
        action="store_true",
        default=False,
        help="Save a histogram of parent→child entailment cone half-angles.",
    )
    parser.add_argument(
        "--plot-success-curve",
        action="store_true",
        default=False,
        help="Save theorem success-rate-vs-epoch plot from theorem_results_epoch_*.json.",
    )
    parser.add_argument(
        "--plot-state-embeddings",
        action="store_true",
        default=False,
        help="Encode states from training_data_epoch_*.json and plot learned embeddings.",
    )
    parser.add_argument(
        "--embedding-sample-size",
        type=int,
        default=600,
        help="Max number of unique states to encode for state embedding plot.",
    )
    args = parser.parse_args()

    if args.search_tree_dir is None and args.run_dir is None:
        parser.error("Provide --search-tree-dir or --run-dir")

    search_tree_dir = args.search_tree_dir
    if search_tree_dir is None and args.run_dir is not None:
        search_tree_dir = args.run_dir / "search_trees"

    if search_tree_dir is None:
        parser.error("Could not determine search tree directory")

    report = analyze_search_tree_directory(search_tree_dir)

    if args.run_dir is not None:
        report["run_artifacts"] = get_run_artifact_stats(args.run_dir)

    default_parent = (
        args.run_dir if args.run_dir is not None else search_tree_dir.parent
    )
    output_path = args.output or (default_parent / "search_tree_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    if args.plot_2d:
        tree_files = sorted(search_tree_dir.glob("search_tree_*.json"))
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
                    search_tree_dir.parent / "search_tree_embeddings_2d.png"
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

    if args.plot_calibration:
        tree_files = sorted(search_tree_dir.glob("search_tree_*.json"))
        if tree_files:
            with open(tree_files[0]) as _f:
                _calib_payload = json.load(_f)
            root = _load_serialized_tree(_calib_payload)
            if root is not None:
                calib_path = search_tree_dir.parent / "value_calibration_curve.png"
                written = plot_calibration_curve(root, calib_path)
                if written is not None:
                    logger.info(f"Saved calibration curve to {written}")

    if args.plot_radius_by_outcome:
        tree_files = sorted(search_tree_dir.glob("search_tree_*.json"))
        if tree_files:
            with open(tree_files[0]) as _f:
                _rbo_payload = json.load(_f)
            _meta = _rbo_payload.get("metadata", {})
            _is_hyp = bool(_meta.get("is_hyperbolic", False))
            _curv = float(_meta.get("curvature", 1.0))
            root = _load_serialized_tree(_rbo_payload)
            if root is not None:
                rbo_path = search_tree_dir.parent / "radius_by_outcome.png"
                written = plot_radius_by_outcome(
                    root, rbo_path, is_hyperbolic=_is_hyp, curvature=_curv
                )
                if written is not None:
                    logger.info(f"Saved radius-by-outcome plot to {written}")

    if args.plot_geodesic_paths:
        tree_files = sorted(search_tree_dir.glob("search_tree_*.json"))
        if tree_files:
            with open(tree_files[0]) as _f:
                _gp_payload = json.load(_f)
            _meta = _gp_payload.get("metadata", {})
            _is_hyp = bool(_meta.get("is_hyperbolic", False))
            _curv = float(_meta.get("curvature", 1.0))
            root = _load_serialized_tree(_gp_payload)
            if root is not None:
                gp_path = search_tree_dir.parent / "geodesic_proof_paths.png"
                written = plot_geodesic_proof_paths(
                    root, gp_path, is_hyperbolic=_is_hyp, curvature=_curv
                )
                if written is not None:
                    logger.info(f"Saved geodesic proof path plot to {written}")

    if args.plot_entailment_cone_dist:
        tree_files = sorted(search_tree_dir.glob("search_tree_*.json"))
        if tree_files:
            with open(tree_files[0]) as _f:
                _ec_payload = json.load(_f)
            _meta = _ec_payload.get("metadata", {})
            _is_hyp = bool(_meta.get("is_hyperbolic", False))
            _curv = float(_meta.get("curvature", 1.0))
            root = _load_serialized_tree(_ec_payload)
            if root is not None:
                ec_path = search_tree_dir.parent / "entailment_cone_distribution.png"
                written = plot_entailment_cone_distribution(
                    root, ec_path, is_hyperbolic=_is_hyp, curvature=_curv
                )
                if written is not None:
                    logger.info(f"Saved entailment cone distribution plot to {written}")

    if args.plot_success_curve and args.run_dir is not None:
        success_path = args.run_dir / "theorem_success_curve.png"
        written = plot_theorem_success_curve(args.run_dir, success_path)
        if written is not None:
            logger.info(f"Saved theorem success curve to {written}")

    if args.plot_state_embeddings and args.run_dir is not None:
        emb_path = args.run_dir / "state_embeddings_2d.png"
        written = plot_checkpoint_state_embeddings(
            args.run_dir,
            emb_path,
            sample_size=max(50, int(args.embedding_sample_size)),
        )
        if written is not None:
            logger.info(f"Saved state embedding plot to {written}")

    logger.info(f"Saved search tree analysis to {output_path}")


def load_search_tree_root(search_tree_file: Path) -> Optional[AnalysisNode]:
    with open(search_tree_file, "r") as handle:
        tree_payload = json.load(handle)
    return _load_serialized_tree(tree_payload)


if __name__ == "__main__":
    main()
