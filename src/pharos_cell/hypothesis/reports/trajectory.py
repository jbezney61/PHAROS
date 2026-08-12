#!/usr/bin/env python
"""
make_explicit_trajectory_report.py

Generate trajectory figures for the explicit --2drug-pair path saved by
positive_control_2drug.py.

Expected input directory:
    run_dir/
        trajectory_embeddings/
            explicit_pair_trajectory.npz
            explicit_pair_trajectory_metadata.json

Outputs:
    run_dir/trajectory_report_projection/  (default; projected PCA--PLS-DA space)
    run_dir/trajectory_report/  (for --embedding-space full)
    run_dir/trajectory_report_compare/  (for --embedding-space both)
        summary.md
        tables/
            trajectory_metrics_by_batch.tsv
            umap_grid_search_results.tsv
        figures/
            with_target/
                01_target_aligned_trajectory.png
                02_distance_to_target.png
                03_distance_from_wt.png
                04_stepwise_gain_waterfall.png
                05_umap_grid_search.png
                06_angle_to_target.png
            without_target/
                matching no-target versions of the same six figure types
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


STATE_COLORS = {
    "start": "#4d4d4d",
    "drug1": "#2f6fed",
    "drug2": "#f28e2b",
    "target": "#2ca25f",
}

STATE_MARKERS = {
    "start": "o",
    "drug1": "s",
    "drug2": "^",
    "target": "D",
}

PUBLICATION_RC = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "figure.dpi": 120,
    "savefig.dpi": 320,
}

np = None
pd = None
plt = None
PCA = None
silhouette_score = None
UMAP = None


def ensure_report_dependencies() -> None:
    """Import plotting/data dependencies lazily so --help works in a bare shell."""
    global np, pd, plt, PCA, silhouette_score, UMAP
    if np is not None and pd is not None and plt is not None and PCA is not None:
        return

    import numpy as _np
    import pandas as _pd
    import matplotlib.pyplot as _plt
    from sklearn.decomposition import PCA as _PCA
    from sklearn.metrics import silhouette_score as _silhouette_score

    try:
        from umap import UMAP as _UMAP
    except Exception:
        _UMAP = None

    np = _np
    pd = _pd
    plt = _plt
    PCA = _PCA
    silhouette_score = _silhouette_score
    UMAP = _UMAP
    plt.rcParams.update(PUBLICATION_RC)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def polish_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.1, length=4)


def make_legend_opaque(ax: Any, **kwargs: Any) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    if legend is None:
        return
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([70])


def load_trajectory(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cache_path = run_dir / "trajectory_embeddings" / "explicit_pair_trajectory.npz"
    metadata_path = run_dir / "trajectory_embeddings" / "explicit_pair_trajectory_metadata.json"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing explicit trajectory cache: {cache_path}. "
            "Run positive_control_2drug_analysis.py with --save-trajectory-embeddings."
        )

    z = np.load(cache_path, allow_pickle=True)
    data = {
        "start": z["start_embeddings"].astype(np.float32, copy=False),
        "drug1": z["drug1_embeddings"].astype(np.float32, copy=False),
        "drug2": z["drug2_embeddings"].astype(np.float32, copy=False),
        "target": z["target_embeddings"].astype(np.float32, copy=False),
        "batch_index": z["batch_index"].astype(int) if "batch_index" in z else np.arange(z["start_embeddings"].shape[0]),
        "seed": z["seed"].astype(int) if "seed" in z else np.arange(z["start_embeddings"].shape[0]),
    }
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    return data, metadata


def resolve_projection_cache(run_dir: Path, projection_cache: Optional[str | Path]) -> Path:
    if projection_cache is not None:
        path = Path(projection_cache)
        if not path.exists():
            raise FileNotFoundError(f"Projection cache was provided but does not exist: {path}")
        return path

    config_path = run_dir / "positive_control_config.used.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text())
            candidate = payload.get("paths", {}).get("projection_cache")
            if candidate and Path(candidate).exists():
                return Path(candidate)
            candidate = payload.get("config", {}).get("projection_cache_path")
            if candidate and Path(candidate).exists():
                return Path(candidate)
        except Exception:
            pass

    candidates = [
        run_dir / "cache" / "projection.npz",
        run_dir / "projection.npz",
    ]
    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "A projected trajectory report needs the fitted projection cache, but none was found. "
        "Pass --projection-cache, or rerun positive_control_2drug_analysis.py with projection enabled "
        "so it writes <run-dir>/cache/projection.npz."
    )


def load_projection(projection_cache: Path) -> Any:
    from ...projections import LinearProjection

    return LinearProjection.load(projection_cache)


def transform_trajectory_data(data: Dict[str, Any], projection: Any) -> Dict[str, Any]:
    projected = {
        key: projection.transform(data[key]).astype(np.float32, copy=False)
        for key in ["start", "drug1", "drug2", "target"]
    }
    projected["batch_index"] = data["batch_index"]
    projected["seed"] = data["seed"]
    return projected


def projection_summary(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return "full"
    method = metadata.get("method", "projection")
    n_components = metadata.get("n_components", "unknown")
    rotation_hash = metadata.get("rotation_hash", "")
    suffix = f", hash {rotation_hash}" if rotation_hash else ""
    return f"{method}, K={n_components}{suffix}"


def state_label(metadata: Dict[str, Any], key: str) -> str:
    first = str(metadata.get("first_drug", "drug 1"))
    second = str(metadata.get("second_drug", "drug 2"))
    if key == "start":
        return "WT"
    if key == "drug1":
        return f"WT + {first}"
    if key == "drug2":
        return f"WT + {first} + {second}"
    if key == "target":
        return "Target"
    return key


def short_state_label(metadata: Dict[str, Any], key: str) -> str:
    first = str(metadata.get("first_drug", "drug 1"))
    second = str(metadata.get("second_drug", "drug 2"))
    if key == "start":
        return "WT"
    if key == "drug1":
        return first
    if key == "drug2":
        return f"{first} + {second}"
    if key == "target":
        return "Target"
    return key


def visible_state_keys(include_target: bool) -> List[str]:
    keys = ["start", "drug1", "drug2"]
    if include_target:
        keys.append("target")
    return keys


def flatten_state(arr: Any) -> Any:
    return arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2])


def batch_centroids(arr: Any) -> Any:
    return arr.mean(axis=1)


def unit_vector(vec: Any, fallback: Optional[Any] = None) -> Any:
    norm = float(np.linalg.norm(vec))
    if norm > 1e-12:
        return vec / norm
    if fallback is not None:
        return unit_vector(fallback)
    out = np.zeros_like(vec)
    out[0] = 1.0
    return out


def target_aligned_basis(data: Dict[str, Any]) -> Tuple[Any, Any]:
    start_all = flatten_state(data["start"])
    target_all = flatten_state(data["target"])
    drug2_all = flatten_state(data["drug2"])
    origin = start_all.mean(axis=0)
    target_mean = target_all.mean(axis=0)
    axis1 = target_mean - origin

    pooled = np.vstack([flatten_state(data[k]) for k in ["start", "drug1", "drug2", "target"]])
    centered = pooled - origin
    if float(np.linalg.norm(axis1)) <= 1e-12:
        pca = PCA(n_components=2, random_state=0)
        pca.fit(centered)
        axis1 = pca.components_[0]
    axis1 = unit_vector(axis1)

    residual = centered - np.outer(centered @ axis1, axis1)
    residual = residual - residual.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(residual, full_matrices=False)
        axis2 = vh[0]
    except Exception:
        axis2 = np.zeros_like(axis1)
        axis2[1 if axis2.shape[0] > 1 else 0] = 1.0
    axis2 = axis2 - axis1 * float(np.dot(axis2, axis1))
    axis2 = unit_vector(axis2)

    drug2_residual = drug2_all.mean(axis=0) - origin
    drug2_residual = drug2_residual - axis1 * float(np.dot(drug2_residual, axis1))
    if float(np.dot(axis2, drug2_residual)) < 0:
        axis2 = -axis2

    basis = np.column_stack([axis1, axis2])
    return origin, basis


def project_points(x: Any, origin: Any, basis: Any) -> Any:
    return (x - origin) @ basis


def sample_rows(x: Any, max_rows: int, seed: int) -> Any:
    if max_rows <= 0 or x.shape[0] <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    return x[np.sort(idx)]


def add_mean_arrow(ax: Any, p0: Any, p1: Any, color: str, *, linestyle: str = "-") -> None:
    ax.annotate(
        "",
        xy=(float(p1[0]), float(p1[1])),
        xytext=(float(p0[0]), float(p0[1])),
        arrowprops={
            "arrowstyle": "->",
            "lw": 2.5,
            "color": color,
            "linestyle": linestyle,
            "shrinkA": 0,
            "shrinkB": 0,
        },
    )


def plot_target_aligned_trajectory(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    fig_path: Path,
    *,
    include_target: bool,
    max_cells_per_group: int,
    seed: int,
) -> None:
    origin, basis = target_aligned_basis(data)
    keys = visible_state_keys(include_target)

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for i, key in enumerate(keys):
        x = sample_rows(flatten_state(data[key]), max_cells_per_group, seed + i)
        xy = project_points(x, origin, basis)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=18,
            alpha=0.58 if key != "target" else 0.52,
            color=STATE_COLORS[key],
            label=state_label(metadata, key),
            linewidths=0,
        )

    centroids = {key: project_points(batch_centroids(data[key]), origin, basis) for key in keys}
    n_batches = data["start"].shape[0]
    for batch_i in range(n_batches):
        path = np.vstack([centroids["start"][batch_i], centroids["drug1"][batch_i], centroids["drug2"][batch_i]])
        ax.plot(path[:, 0], path[:, 1], color="#222222", alpha=0.35, lw=1.3)
        if include_target:
            target_path = np.vstack([centroids["drug2"][batch_i], centroids["target"][batch_i]])
            ax.plot(target_path[:, 0], target_path[:, 1], color=STATE_COLORS["target"], alpha=0.38, lw=1.2, ls="--")

    mean_centroids = {key: centroids[key].mean(axis=0) for key in keys}
    add_mean_arrow(ax, mean_centroids["start"], mean_centroids["drug1"], STATE_COLORS["drug1"])
    add_mean_arrow(ax, mean_centroids["drug1"], mean_centroids["drug2"], STATE_COLORS["drug2"])
    if include_target:
        add_mean_arrow(ax, mean_centroids["drug2"], mean_centroids["target"], STATE_COLORS["target"], linestyle="--")

    for key in keys:
        xy = mean_centroids[key]
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=165,
            marker=STATE_MARKERS[key],
            color=STATE_COLORS[key],
            edgecolor="white",
            linewidth=1.4,
            zorder=5,
        )

    ax.axhline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.axvline(0, color="#dddddd", lw=0.8, zorder=0)
    ax.set_xlabel("WT-to-target centroid axis")
    ax.set_ylabel("Largest residual axis")
    ax.set_title("Explicit 2-drug trajectory in target-aligned space")
    polish_axes(ax)
    make_legend_opaque(ax, loc="best")
    savefig(fig_path)


def angle_degrees(a: Any, b: Any) -> float:
    an = float(np.linalg.norm(a))
    bn = float(np.linalg.norm(b))
    if an <= 1e-12 or bn <= 1e-12:
        return math.nan
    cos = float(np.dot(a, b) / (an * bn))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def compute_metrics(data: Dict[str, Any]) -> Any:
    start_c = batch_centroids(data["start"])
    drug1_c = batch_centroids(data["drug1"])
    drug2_c = batch_centroids(data["drug2"])
    target_c = batch_centroids(data["target"])
    rows: List[Dict[str, Any]] = []

    for batch_i in range(start_c.shape[0]):
        wt = start_c[batch_i]
        d1 = drug1_c[batch_i]
        d2 = drug2_c[batch_i]
        tgt = target_c[batch_i]

        dist_wt_target = float(np.linalg.norm(wt - tgt))
        dist_d1_target = float(np.linalg.norm(d1 - tgt))
        dist_d2_target = float(np.linalg.norm(d2 - tgt))
        step1 = float(np.linalg.norm(d1 - wt))
        step2 = float(np.linalg.norm(d2 - d1))
        cumulative = float(np.linalg.norm(d2 - wt))
        target_from_wt = dist_wt_target
        gain1 = dist_wt_target - dist_d1_target
        gain2 = dist_d1_target - dist_d2_target

        rows.append(
            {
                "batch_index": int(batch_i),
                "dist_wt_to_target": dist_wt_target,
                "dist_drug1_to_target": dist_d1_target,
                "dist_drug2_to_target": dist_d2_target,
                "dist_wt_from_wt": 0.0,
                "dist_drug1_from_wt": step1,
                "dist_drug2_from_wt": cumulative,
                "dist_target_from_wt": target_from_wt,
                "step1_length": step1,
                "step2_length": step2,
                "path_length": step1 + step2,
                "cumulative_displacement": cumulative,
                "drug1_gain_to_target": gain1,
                "drug2_gain_to_target": gain2,
                "total_gain_to_target": dist_wt_target - dist_d2_target,
                "remaining_target_gap": dist_d2_target,
                "step1_angle_to_target_deg": angle_degrees(d1 - wt, tgt - wt),
                "step2_angle_to_remaining_target_deg": angle_degrees(d2 - d1, tgt - d1),
                "total_angle_to_target_deg": angle_degrees(d2 - wt, tgt - wt),
            }
        )

    return pd.DataFrame(rows)


def plot_line_metric(
    metrics: Any,
    metadata: Dict[str, Any],
    fig_path: Path,
    *,
    include_target: bool,
    metric_kind: str,
) -> None:
    if metric_kind == "to_target":
        cols = ["dist_wt_to_target", "dist_drug1_to_target", "dist_drug2_to_target"]
        ylabel = "Euclidean distance to target centroid"
        title = "Distance to target across sequential drug steps"
        if include_target:
            cols.append("_target_zero")
    else:
        cols = ["dist_wt_from_wt", "dist_drug1_from_wt", "dist_drug2_from_wt"]
        ylabel = "Euclidean distance from WT centroid"
        title = "Cumulative distance from WT across sequential drug steps"
        if include_target:
            cols.append("dist_target_from_wt")

    labels = [short_state_label(metadata, "start"), short_state_label(metadata, "drug1"), short_state_label(metadata, "drug2")]
    if include_target:
        labels.append(short_state_label(metadata, "target"))

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    x = np.arange(len(cols))
    values = []
    for _, row in metrics.iterrows():
        y = []
        for col in cols:
            y.append(0.0 if col == "_target_zero" else float(row[col]))
        values.append(y)
        ax.plot(x, y, color="#777777", alpha=0.45, lw=1.3, marker="o", ms=4.5)

    arr = np.asarray(values, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    ax.errorbar(x, mean, yerr=std, color="#111111", lw=2.8, marker="o", ms=7, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    polish_axes(ax)
    savefig(fig_path)


def plot_stepwise_gain(
    metrics: Any,
    fig_path: Path,
    *,
    include_target: bool,
) -> None:
    columns = [
        ("drug1_gain_to_target", "Drug 1 gain", STATE_COLORS["drug1"]),
        ("drug2_gain_to_target", "Drug 2 gain", STATE_COLORS["drug2"]),
    ]
    if include_target:
        columns.append(("remaining_target_gap", "Remaining gap", STATE_COLORS["target"]))
    else:
        columns.append(("total_gain_to_target", "Total gain", "#7b3294"))

    fig, ax = plt.subplots(figsize=(5.8, 4.7))
    x = np.arange(len(columns))
    means = [float(metrics[col].mean()) for col, _, _ in columns]
    stds = [float(metrics[col].std(ddof=1)) if len(metrics) > 1 else 0.0 for col, _, _ in columns]
    colors = [color for _, _, color in columns]
    labels = [label for _, label, _ in columns]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.88, capsize=4, edgecolor="#333333", linewidth=0.7)

    rng = np.random.default_rng(17)
    for i, (col, _, _) in enumerate(columns):
        jitter = rng.normal(0, 0.035, size=len(metrics))
        ax.scatter(np.full(len(metrics), i) + jitter, metrics[col], color="#111111", s=22, alpha=0.65, zorder=4)

    ax.axhline(0, color="#222222", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Euclidean distance units")
    ax.set_title("Stepwise target-distance gain")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    polish_axes(ax)
    savefig(fig_path)


def pooled_visible_cells(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    include_target: bool,
    max_cells_per_group: int,
    seed: int,
) -> Tuple[Any, Any, List[str]]:
    xs = []
    labels = []
    label_names: List[str] = []
    for i, key in enumerate(visible_state_keys(include_target)):
        label = short_state_label(metadata, key)
        x = sample_rows(flatten_state(data[key]), max_cells_per_group, seed + i * 103)
        xs.append(x)
        labels.extend([i] * x.shape[0])
        label_names.append(label)
    return np.vstack(xs), np.asarray(labels, dtype=int), label_names


def reduce_for_umap(x: Any, seed: int) -> Any:
    n_components = min(50, int(x.shape[0] - 1), int(x.shape[1]))
    if n_components < 2 or x.shape[1] <= n_components:
        return x
    pca = PCA(n_components=n_components, random_state=seed)
    return pca.fit_transform(x)


def safe_silhouette(embedding: Any, labels: Any) -> float:
    try:
        if len(np.unique(labels)) <= 1:
            return math.nan
        return float(silhouette_score(embedding, labels))
    except Exception:
        return math.nan


def plot_umap_grid_search(
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    fig_path: Path,
    *,
    include_target: bool,
    max_cells_per_group: int,
    seed: int,
) -> Any:
    x, labels, label_names = pooled_visible_cells(
        data,
        metadata,
        include_target=include_target,
        max_cells_per_group=max_cells_per_group,
        seed=seed,
    )
    x_reduced = reduce_for_umap(x, seed)

    rows: List[Dict[str, Any]] = []
    best_embedding = None
    best_score = math.nan
    best_row: Dict[str, Any] = {}

    if UMAP is None:
        pca = PCA(n_components=2, random_state=seed)
        embedding = pca.fit_transform(x_reduced)
        score = safe_silhouette(embedding, labels)
        best_embedding = embedding
        best_score = score
        best_row = {
            "target_included": bool(include_target),
            "method": "pca_fallback",
            "n_neighbors": math.nan,
            "min_dist": math.nan,
            "metric": "not_applicable",
            "silhouette_score": score,
            "selected": True,
            "note": "umap-learn was not available; PCA fallback used",
        }
        rows.append(best_row)
    else:
        n_samples = int(x_reduced.shape[0])
        neighbor_grid = [n for n in [10, 30, 60] if n < n_samples]
        if not neighbor_grid:
            neighbor_grid = [max(2, n_samples - 1)]
        min_dist_grid = [0.0, 0.2]
        metric_grid = ["cosine", "euclidean"]

        for n_neighbors in neighbor_grid:
            for min_dist in min_dist_grid:
                for metric in metric_grid:
                    reducer = UMAP(
                        n_components=2,
                        n_neighbors=int(n_neighbors),
                        min_dist=float(min_dist),
                        metric=metric,
                        random_state=int(seed),
                    )
                    embedding = reducer.fit_transform(x_reduced)
                    score = safe_silhouette(embedding, labels)
                    selected = bool(
                        best_embedding is None
                        or (np.isfinite(score) and (not np.isfinite(best_score) or score > best_score))
                    )
                    row = {
                        "target_included": bool(include_target),
                        "method": "umap",
                        "n_neighbors": int(n_neighbors),
                        "min_dist": float(min_dist),
                        "metric": metric,
                        "silhouette_score": score,
                        "selected": False,
                        "note": "",
                    }
                    rows.append(row)
                    if selected:
                        best_score = score
                        best_embedding = embedding
                        best_row = row

        for row in rows:
            row["selected"] = (
                row["method"] == best_row.get("method")
                and row["n_neighbors"] == best_row.get("n_neighbors")
                and row["min_dist"] == best_row.get("min_dist")
                and row["metric"] == best_row.get("metric")
            )

    fig, ax = plt.subplots(figsize=(6.6, 5.1))
    for i, key in enumerate(visible_state_keys(include_target)):
        mask = labels == i
        ax.scatter(
            best_embedding[mask, 0],
            best_embedding[mask, 1],
            s=18,
            alpha=0.72,
            color=STATE_COLORS[key],
            label=label_names[i],
            linewidths=0,
        )

    method = str(best_row.get("method", "umap"))
    if method == "umap":
        subtitle = (
            f"best n_neighbors={best_row.get('n_neighbors')}, "
            f"min_dist={best_row.get('min_dist')}, metric={best_row.get('metric')}, "
            f"silhouette={best_score:.3f}"
        )
    else:
        subtitle = f"PCA fallback, silhouette={best_score:.3f}"
    ax.set_title(f"UMAP separation of trajectory states\n{subtitle}")
    ax.set_xlabel("UMAP 1" if method == "umap" else "PC 1")
    ax.set_ylabel("UMAP 2" if method == "umap" else "PC 2")
    polish_axes(ax)
    make_legend_opaque(ax, loc="best")
    savefig(fig_path)
    return pd.DataFrame(rows)


def plot_angle_to_target(
    metrics: Any,
    fig_path: Path,
    *,
    include_target: bool,
) -> None:
    columns = [
        ("step1_angle_to_target_deg", "Drug 1"),
        ("step2_angle_to_remaining_target_deg", "Drug 2"),
    ]
    if include_target:
        columns.append(("total_angle_to_target_deg", "Total path"))

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    raw_values = [metrics[col].astype(float).to_numpy() for col, _ in columns]
    values = [vals[np.isfinite(vals)] for vals in raw_values]
    labels = [label for _, label in columns]
    colors = [STATE_COLORS["drug1"], STATE_COLORS["drug2"], "#7b3294"][: len(columns)]
    valid_positions = [i + 1 for i, vals in enumerate(values) if len(vals)]
    valid_values = [vals for vals in values if len(vals)]
    valid_colors = [colors[i] for i, vals in enumerate(values) if len(vals)]
    if valid_values:
        parts = ax.violinplot(valid_values, positions=valid_positions, showmeans=True, showextrema=False)
        for body, color in zip(parts["bodies"], valid_colors):
            body.set_facecolor(color)
            body.set_alpha(0.35)
            body.set_edgecolor(color)
        parts["cmeans"].set_color("#111111")
        parts["cmeans"].set_linewidth(2.0)

    rng = np.random.default_rng(23)
    for i, vals in enumerate(values, start=1):
        if len(vals):
            jitter = rng.normal(0, 0.035, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                color=colors[i - 1],
                edgecolor="#333333",
                s=24,
                alpha=0.85,
            )
        else:
            ax.text(i, 8, "undefined", rotation=90, ha="center", va="bottom", color="#666666", fontsize=9)

    ax.axhline(90, color="#555555", lw=1.0, ls="--")
    ax.text(0.55, 91.5, "90 deg = sideways", fontsize=9, color="#555555")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Angle to target direction (degrees)")
    ax.set_title("Directional alignment of each drug step")
    ax.set_ylim(0, 180)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    polish_axes(ax)
    savefig(fig_path)


def write_summary(
    report_dir: Path,
    metadata: Dict[str, Any],
    metrics: Any,
    umap_grid: Any,
    *,
    max_cells_per_group: int,
    embedding_space: str,
    embedding_dim: int,
    projection_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    selected = umap_grid[umap_grid["selected"].astype(bool)].copy() if not umap_grid.empty else pd.DataFrame()
    lines = [
        "# Explicit 2-drug trajectory report",
        "",
        f"- Pair: `{metadata.get('pair_id', 'unknown')}`",
        f"- First perturbation: `{metadata.get('first_perturbation', 'unknown')}`",
        f"- Second perturbation: `{metadata.get('second_perturbation', 'unknown')}`",
        f"- Embedding space: `{embedding_space}`",
        f"- Embedding dimension used for metrics/figures: `{embedding_dim}`",
        f"- Batches: `{len(metrics)}`",
        f"- UMAP/grid max cells per group: `{max_cells_per_group}`",
        "",
        "## Mean trajectory metrics",
        "",
        f"- Drug 1 gain toward target: `{metrics['drug1_gain_to_target'].mean():.6g}`",
        f"- Drug 2 gain toward target: `{metrics['drug2_gain_to_target'].mean():.6g}`",
        f"- Total gain toward target: `{metrics['total_gain_to_target'].mean():.6g}`",
        f"- Remaining target gap after drug 2: `{metrics['remaining_target_gap'].mean():.6g}`",
        f"- Mean drug 1 angle to target: `{metrics['step1_angle_to_target_deg'].mean():.6g}` degrees",
        f"- Mean drug 2 angle to remaining target: `{metrics['step2_angle_to_remaining_target_deg'].mean():.6g}` degrees",
        "",
        "## UMAP selection",
        "",
        "The UMAP panels intentionally select parameters by silhouette score over the visible groups.",
        "Use them as visual separation views, not as an unbiased biological distance estimate.",
    ]

    if projection_metadata:
        lines.extend(
            [
                "",
                "## Projection",
                "",
                f"- Projection: `{projection_summary(projection_metadata)}`",
                f"- Input dimension: `{projection_metadata.get('input_dim', 'unknown')}`",
                f"- Whiten components: `{projection_metadata.get('whiten_components', 'unknown')}`",
            ]
        )

    if not selected.empty:
        for _, row in selected.iterrows():
            label = "with target" if bool(row["target_included"]) else "without target"
            lines.append(
                f"- `{label}`: method `{row['method']}`, n_neighbors `{row['n_neighbors']}`, "
                f"min_dist `{row['min_dist']}`, metric `{row['metric']}`, silhouette `{row['silhouette_score']:.6g}`"
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `figures/with_target/`: six trajectory views including target cells or target reference.",
            "- `figures/without_target/`: matching views with visible groups limited to WT, drug 1, and drug 1 plus drug 2.",
            "- `tables/trajectory_metrics_by_batch.tsv`: centroid distances, gains, path lengths, and angles by batch.",
            "- `tables/umap_grid_search_results.tsv`: UMAP parameters and silhouette scores.",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _make_single_trajectory_report(
    *,
    report_dir: Path,
    data: Dict[str, Any],
    metadata: Dict[str, Any],
    max_cells_per_group: int = 1200,
    seed: int = 42,
    embedding_space: str = "projection",
    projection_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, str], Any, Any]:
    table_dir = safe_mkdir(report_dir / "tables")
    fig_root = safe_mkdir(report_dir / "figures")
    fig_with = safe_mkdir(fig_root / "with_target")
    fig_without = safe_mkdir(fig_root / "without_target")

    metrics = compute_metrics(data)
    metrics_path = table_dir / "trajectory_metrics_by_batch.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)

    figure_paths: Dict[str, str] = {}
    umap_tables = []
    for include_target, fig_dir, prefix in [
        (True, fig_with, "with_target"),
        (False, fig_without, "without_target"),
    ]:
        figure_paths[f"{prefix}_target_aligned_trajectory"] = str(fig_dir / "01_target_aligned_trajectory.png")
        plot_target_aligned_trajectory(
            data,
            metadata,
            fig_dir / "01_target_aligned_trajectory.png",
            include_target=include_target,
            max_cells_per_group=max_cells_per_group,
            seed=seed,
        )

        figure_paths[f"{prefix}_distance_to_target"] = str(fig_dir / "02_distance_to_target.png")
        plot_line_metric(
            metrics,
            metadata,
            fig_dir / "02_distance_to_target.png",
            include_target=include_target,
            metric_kind="to_target",
        )

        figure_paths[f"{prefix}_distance_from_wt"] = str(fig_dir / "03_distance_from_wt.png")
        plot_line_metric(
            metrics,
            metadata,
            fig_dir / "03_distance_from_wt.png",
            include_target=include_target,
            metric_kind="from_wt",
        )

        figure_paths[f"{prefix}_stepwise_gain_waterfall"] = str(fig_dir / "04_stepwise_gain_waterfall.png")
        plot_stepwise_gain(metrics, fig_dir / "04_stepwise_gain_waterfall.png", include_target=include_target)

        figure_paths[f"{prefix}_umap_grid_search"] = str(fig_dir / "05_umap_grid_search.png")
        umap_tables.append(
            plot_umap_grid_search(
                data,
                metadata,
                fig_dir / "05_umap_grid_search.png",
                include_target=include_target,
                max_cells_per_group=max_cells_per_group,
                seed=seed,
            )
        )

        figure_paths[f"{prefix}_angle_to_target"] = str(fig_dir / "06_angle_to_target.png")
        plot_angle_to_target(metrics, fig_dir / "06_angle_to_target.png", include_target=include_target)

    umap_grid = pd.concat(umap_tables, ignore_index=True, sort=False) if umap_tables else pd.DataFrame()
    umap_grid_path = table_dir / "umap_grid_search_results.tsv"
    umap_grid.to_csv(umap_grid_path, sep="\t", index=False)

    write_summary(
        report_dir,
        metadata,
        metrics,
        umap_grid,
        max_cells_per_group=max_cells_per_group,
        embedding_space=embedding_space,
        embedding_dim=int(data["start"].shape[-1]),
        projection_metadata=projection_metadata,
    )

    out = {
        "report_dir": str(report_dir),
        "summary": str(report_dir / "summary.md"),
        "figures_dir": str(fig_root),
        "trajectory_metrics_by_batch": str(metrics_path),
        "umap_grid_search_results": str(umap_grid_path),
    }
    out.update(figure_paths)
    return out, metrics, umap_grid


def summarize_metrics(metrics: Any, *, space: str) -> Any:
    metric_cols = [
        "dist_wt_to_target",
        "dist_drug1_to_target",
        "dist_drug2_to_target",
        "step1_length",
        "step2_length",
        "path_length",
        "cumulative_displacement",
        "drug1_gain_to_target",
        "drug2_gain_to_target",
        "total_gain_to_target",
        "remaining_target_gap",
        "step1_angle_to_target_deg",
        "step2_angle_to_remaining_target_deg",
        "total_angle_to_target_deg",
    ]
    rows = []
    for col in metric_cols:
        vals = metrics[col].astype(float).to_numpy()
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "space": space,
                "metric": col,
                "mean": float(np.mean(vals)) if len(vals) else math.nan,
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def write_space_comparison_summary(
    *,
    report_dir: Path,
    metadata: Dict[str, Any],
    full_out: Dict[str, str],
    projection_out: Dict[str, str],
    comparison_path: Path,
    projection_metadata: Dict[str, Any],
) -> Path:
    lines = [
        "# Explicit 2-drug trajectory report comparison",
        "",
        f"- Pair: `{metadata.get('pair_id', 'unknown')}`",
        f"- First perturbation: `{metadata.get('first_perturbation', 'unknown')}`",
        f"- Second perturbation: `{metadata.get('second_perturbation', 'unknown')}`",
        f"- Projection: `{projection_summary(projection_metadata)}`",
        "",
        "## Reports",
        "",
        f"- Full-dimensional report: `{Path(full_out['summary']).relative_to(report_dir)}`",
        f"- Projected report: `{Path(projection_out['summary']).relative_to(report_dir)}`",
        f"- Metric comparison table: `{comparison_path.relative_to(report_dir)}`",
        "",
        "Distances and angles are computed in their own embedding spaces. Use the paired figures and rank/separation metrics to compare behavior; raw distance magnitudes are not directly comparable between 2058-D and projected PLS space.",
    ]
    summary_path = report_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def make_explicit_trajectory_report(
    *,
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    max_cells_per_group: int = 1200,
    seed: int = 42,
    embedding_space: str = "projection",
    projection_cache: Optional[str | Path] = None,
) -> Dict[str, str]:
    ensure_report_dependencies()

    embedding_space = str(embedding_space).lower()
    if embedding_space not in {"full", "projection", "both"}:
        raise ValueError("embedding_space must be one of: full, projection, both")

    run_dir = Path(run_dir)
    full_data, metadata = load_trajectory(run_dir)

    if embedding_space == "full":
        report_dir = Path(output_dir) if output_dir else run_dir / "trajectory_report"
        out, _metrics, _umap = _make_single_trajectory_report(
            report_dir=report_dir,
            data=full_data,
            metadata=metadata,
            max_cells_per_group=max_cells_per_group,
            seed=seed,
            embedding_space="full",
        )
        out["embedding_space"] = "full"
        return out

    projection_cache_path = resolve_projection_cache(run_dir, projection_cache)
    projection = load_projection(projection_cache_path)
    projected_data = transform_trajectory_data(full_data, projection)
    projection_meta = projection.metadata_dict()
    projection_meta["projection_cache"] = str(projection_cache_path)

    if embedding_space == "projection":
        report_dir = Path(output_dir) if output_dir else run_dir / "trajectory_report_projection"
        out, _metrics, _umap = _make_single_trajectory_report(
            report_dir=report_dir,
            data=projected_data,
            metadata=metadata,
            max_cells_per_group=max_cells_per_group,
            seed=seed,
            embedding_space="projection",
            projection_metadata=projection_meta,
        )
        out["embedding_space"] = "projection"
        out["projection_cache"] = str(projection_cache_path)
        return out

    report_dir = Path(output_dir) if output_dir else run_dir / "trajectory_report_compare"
    safe_mkdir(report_dir)
    full_out, full_metrics, _full_umap = _make_single_trajectory_report(
        report_dir=report_dir / "full",
        data=full_data,
        metadata=metadata,
        max_cells_per_group=max_cells_per_group,
        seed=seed,
        embedding_space="full",
    )
    projection_out, projection_metrics, _projection_umap = _make_single_trajectory_report(
        report_dir=report_dir / "projection",
        data=projected_data,
        metadata=metadata,
        max_cells_per_group=max_cells_per_group,
        seed=seed,
        embedding_space="projection",
        projection_metadata=projection_meta,
    )

    table_dir = safe_mkdir(report_dir / "tables")
    comparison = pd.concat(
        [
            summarize_metrics(full_metrics, space="full"),
            summarize_metrics(projection_metrics, space="projection"),
        ],
        ignore_index=True,
        sort=False,
    )
    comparison_path = table_dir / "trajectory_metric_summary_by_space.tsv"
    comparison.to_csv(comparison_path, sep="\t", index=False)
    summary_path = write_space_comparison_summary(
        report_dir=report_dir,
        metadata=metadata,
        full_out=full_out,
        projection_out=projection_out,
        comparison_path=comparison_path,
        projection_metadata=projection_meta,
    )

    return {
        "embedding_space": "both",
        "report_dir": str(report_dir),
        "summary": str(summary_path),
        "projection_cache": str(projection_cache_path),
        "full_report_dir": full_out["report_dir"],
        "full_summary": full_out["summary"],
        "projection_report_dir": projection_out["report_dir"],
        "projection_summary": projection_out["summary"],
        "trajectory_metric_summary_by_space": str(comparison_path),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate explicit --2drug-pair trajectory report and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory produced by positive_control_2drug_analysis.py.")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Report output directory. Defaults depend on --embedding-space: trajectory_report, trajectory_report_projection, or trajectory_report_compare.",
    )
    p.add_argument("--max-cells-per-group", type=int, default=1200, help="Maximum cells per visible group for scatter/UMAP.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for plotting subsampling and UMAP.")
    p.add_argument(
        "--embedding-space",
        choices=["full", "projection", "both"],
        default="projection",
        help="Generate figures/metrics in full 2058-D space, fitted projection space, or both.",
    )
    p.add_argument(
        "--projection-cache",
        default=None,
        help="Fitted projection .npz. If omitted, the report looks for <run-dir>/cache/projection.npz.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = make_explicit_trajectory_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        max_cells_per_group=args.max_cells_per_group,
        seed=args.seed,
        embedding_space=args.embedding_space,
        projection_cache=args.projection_cache,
    )
    print("\n=== Explicit trajectory report complete ===")
    print(f"summary: {out['summary']}")
    if out.get("embedding_space") == "both":
        print(f"full report:       {out['full_report_dir']}")
        print(f"projection report: {out['projection_report_dir']}")
        print(f"comparison table:  {out['trajectory_metric_summary_by_space']}")
    else:
        print(f"figures: {out['figures_dir']}")
        print(f"tables:  {Path(out['trajectory_metrics_by_batch']).parent}")


if __name__ == "__main__":
    main()
