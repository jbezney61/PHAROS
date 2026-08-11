#!/usr/bin/env python
"""
make_embedding_manifold_qc_report.py

Generate figures and a concise report for embedding_manifold_qc_analysis.py
score-query outputs.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


CALL_COLORS = {
    "core": "#2ca25f",
    "boundary": "#f28e2b",
    "OOD": "#d73027",
}

CALIBRATION_COLORS = {
    "seen_reference_state": "#6baed6",
    "heldout_reference_state": "#756bb1",
    "heldout_cell_name": "#252525",
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


def ensure_report_dependencies() -> None:
    global np, pd, plt
    if np is not None and pd is not None and plt is not None:
        return
    import numpy as _np
    import pandas as _pd
    import matplotlib.pyplot as _plt

    np = _np
    pd = _pd
    plt = _plt
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


def support_call_handles():
    return [
        plt.Line2D([0], [0], marker="s", color="w", label=call, markerfacecolor=color, markersize=9)
        for call, color in CALL_COLORS.items()
    ]


def make_legend_opaque(ax: Any, **kwargs: Any) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    if legend is None:
        return
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)


def read_table(path: str | Path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def to_dense_float32(x: Any):
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding batch, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def transform_embeddings(x: Any, metric: str):
    arr = to_dense_float32(x)
    if metric == "cosine":
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-12)
    return np.ascontiguousarray(arr, dtype=np.float32)


def read_obsm_rows(adata: Any, embed_key: str, indices: Sequence[int], *, metric: str):
    ids = np.asarray(indices, dtype=np.int64)
    if len(ids) == 0:
        return np.empty((0, int(adata.obsm[embed_key].shape[1])), dtype=np.float32)
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    x = adata.obsm[embed_key]
    try:
        batch = x[unique_ids]
    except Exception:
        pieces = [x[int(i) : int(i) + 1] for i in unique_ids]
        batch = np.vstack([to_dense_float32(p) for p in pieces])
    batch = transform_embeddings(batch, metric)
    return batch[inverse]


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "manifold_qc_config.used.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing query QC manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tables(run_dir: Path) -> Dict[str, Any]:
    manifest = load_manifest(run_dir)
    paths = manifest.get("paths", {})
    required = {
        "state_scores": paths.get("query_state_scores"),
        "cell_scores": paths.get("query_cell_scores"),
        "state_votes": paths.get("nearest_reference_state_votes"),
        "tissue_votes": paths.get("nearest_reference_tissue_votes"),
        "calibration_state_scores": paths.get("calibration_state_scores"),
    }
    missing = [name for name, path in required.items() if not path or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required manifold QC tables: {missing}")
    return {
        "manifest": manifest,
        "state_scores": read_table(required["state_scores"]),
        "cell_scores": read_table(required["cell_scores"]),
        "state_votes": read_table(required["state_votes"]),
        "tissue_votes": read_table(required["tissue_votes"]),
        "calibration_state_scores": read_table(required["calibration_state_scores"]),
    }


def finite_values(values: Sequence[float]):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def wrap_label(label: str, width: int = 28) -> str:
    label = str(label)
    wrapped = textwrap.wrap(label, width=width, break_long_words=False, break_on_hyphens=False)
    return "\n".join(wrapped or [label])


def markdown_table(df, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No rows._"
    d = df.head(max_rows).copy()

    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        s = str(x).replace("|", "\\|")
        return s[:137] + "..." if len(s) > 140 else s

    headers = [str(c).replace("|", "\\|") for c in d.columns]
    rows = [[fmt(v) for v in row] for row in d.itertuples(index=False, name=None)]
    widths = []
    for j, header in enumerate(headers):
        widths.append(max(len(header), max([len(row[j]) for row in rows], default=0)))
    header = "| " + " | ".join(h.ljust(widths[j]) for j, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def state_sort_column(state_scores) -> str:
    for col in [
        "ood_percentile_vs_heldout_cell_name",
        "ood_percentile_vs_heldout_reference_state",
        "ood_percentile_vs_seen_reference_state",
        "median_cell_ood_percentile_vs_seen",
        "median_local_density_ratio",
    ]:
        if col in state_scores.columns:
            return col
    raise ValueError("No suitable OOD score column found in query_state_scores.")


def top_states(state_scores, top_n_states: int):
    sort_col = state_sort_column(state_scores)
    d = state_scores.copy()
    d[sort_col] = pd.to_numeric(d[sort_col], errors="coerce")
    return d.sort_values([sort_col, "frac_cells_above_seen_99"], ascending=[False, False]).head(int(top_n_states))


def plot_state_percentiles(state_scores, fig_dir: Path, top_n_states: int) -> None:
    path = fig_dir / "01_query_state_ood_percentiles.png"
    d = top_states(state_scores, top_n_states).copy()
    sort_col = state_sort_column(d)
    d = d.sort_values(sort_col, ascending=True)
    plt.figure(figsize=(7.4, max(4.8, 0.38 * len(d) + 1.8)))
    ax = plt.gca()
    y = np.arange(len(d))
    colors = [CALL_COLORS.get(str(x), "#9aa0a6") for x in d["reference_support_call"]]
    ax.barh(y, d[sort_col].astype(float), color=colors, edgecolor="#333333", linewidth=0.6)
    ax.axvline(95, color="#666666", linestyle=":", linewidth=1.3)
    ax.axvline(99, color="#111111", linestyle=":", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, width=34) for x in d["query_cell_type"]], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("State-level OOD percentile versus reference calibration")
    ax.set_title("Which query states fall outside the reference manifold?")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.legend(handles=support_call_handles(), frameon=False, loc="lower right", title="Reference support")
    polish_axes(ax)
    savefig(path)


def plot_cell_percentile_boxplot(cell_scores, state_scores, fig_dir: Path, top_n_states: int) -> None:
    path = fig_dir / "02_cell_level_seen_percentiles_by_state.png"
    d_states = top_states(state_scores, top_n_states)
    order = d_states["query_cell_type"].astype(str).tolist()
    d = cell_scores[cell_scores["query_cell_type"].astype(str).isin(order)].copy()
    if d.empty:
        plt.figure(figsize=(7, 4))
        plt.gca().text(0.5, 0.5, "No cell-level scores available", ha="center", va="center")
        plt.gca().axis("off")
        savefig(path)
        return
    data = [finite_values(d.loc[d["query_cell_type"].astype(str) == state, "ood_percentile_vs_seen_cell"]) for state in order]
    plt.figure(figsize=(max(7.2, 0.42 * len(order) + 2.8), 5.0))
    ax = plt.gca()
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    call_map = dict(zip(d_states["query_cell_type"].astype(str), d_states["reference_support_call"].astype(str)))
    for patch, state in zip(bp["boxes"], order):
        patch.set_facecolor(CALL_COLORS.get(call_map.get(state, ""), "#9aa0a6"))
        patch.set_alpha(0.9)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#333333")
    ax.axhline(95, color="#666666", linestyle=":", linewidth=1.3)
    ax.axhline(99, color="#111111", linestyle=":", linewidth=1.5)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([wrap_label(x, width=18) for x in order], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cell-level OOD percentile\nversus seen reference cells")
    ax.set_title("How far do individual cells extend beyond seen references?")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.legend(handles=support_call_handles(), frameon=False, loc="lower right", title="Reference support")
    polish_axes(ax)
    savefig(path)


def ecdf(values):
    values = np.sort(finite_values(values))
    if len(values) == 0:
        return values, values
    y = np.arange(1, len(values) + 1) / float(len(values))
    return values, y


def plot_calibration_overlay(calibration_state_scores, state_scores, fig_dir: Path) -> None:
    path = fig_dir / "03_calibration_ecdf_with_query_states.png"
    score_col = "median_local_density_ratio"
    plt.figure(figsize=(6.8, 4.8))
    ax = plt.gca()
    for cal_name, sub in calibration_state_scores.groupby("calibration", sort=False):
        if score_col not in sub.columns:
            continue
        x, y = ecdf(sub[score_col])
        if len(x) == 0:
            continue
        ax.plot(
            x,
            y,
            linewidth=2.6,
            color=CALIBRATION_COLORS.get(str(cal_name), "#9aa0a6"),
            label=f"{cal_name} (n={len(x)})",
        )
    q = state_scores.copy()
    q[score_col] = pd.to_numeric(q[score_col], errors="coerce")
    for call, sub in q.groupby("reference_support_call", sort=False):
        vals = finite_values(sub[score_col])
        if len(vals):
            ax.scatter(
                vals,
                np.full(len(vals), 1.02),
                s=42,
                color=CALL_COLORS.get(str(call), "#9aa0a6"),
                label=f"query {call}",
                clip_on=False,
                alpha=0.9,
            )
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Median local density ratio")
    ax.set_ylabel("Calibration ECDF")
    ax.set_title("Query density scores relative to reference calibration")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    make_legend_opaque(ax, fontsize=9, loc="lower right")
    polish_axes(ax)
    savefig(path)


def plot_outlier_fraction(state_scores, fig_dir: Path, top_n_states: int) -> None:
    path = fig_dir / "04_query_state_cell_tail_fractions.png"
    d = top_states(state_scores, top_n_states).copy()
    d = d.sort_values("frac_cells_above_seen_99", ascending=True)
    plt.figure(figsize=(7.4, max(4.8, 0.38 * len(d) + 1.8)))
    ax = plt.gca()
    y = np.arange(len(d))
    ax.barh(
        y - 0.18,
        d["frac_cells_above_seen_95"].astype(float),
        height=0.34,
        color="#9ecae1",
        edgecolor="#333333",
        linewidth=0.5,
        label="> seen-cell p95",
    )
    ax.barh(
        y + 0.18,
        d["frac_cells_above_seen_99"].astype(float),
        height=0.34,
        color="#d73027",
        edgecolor="#333333",
        linewidth=0.5,
        label="> seen-cell p99",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, width=34) for x in d["query_cell_type"]], fontsize=9)
    ax.set_xlim(0, max(1.0, float(d[["frac_cells_above_seen_95", "frac_cells_above_seen_99"]].max().max()) * 1.08))
    ax.set_xlabel("Fraction of scored cells")
    ax.set_title("How much of each query state lies in the reference tail?")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    make_legend_opaque(ax, loc="lower right")
    polish_axes(ax)
    savefig(path)


def plot_tissue_heatmap(tissue_votes, state_scores, fig_dir: Path, top_n_states: int, max_tissues: int = 14) -> None:
    path = fig_dir / "05_nearest_reference_tissue_support_heatmap.png"
    states = top_states(state_scores, top_n_states)["query_cell_type"].astype(str).tolist()
    d = tissue_votes[tissue_votes["query_cell_type"].astype(str).isin(states)].copy()
    if d.empty:
        plt.figure(figsize=(7, 4))
        plt.gca().text(0.5, 0.5, "No tissue votes available", ha="center", va="center")
        plt.gca().axis("off")
        savefig(path)
        return
    tissue_order = (
        d.groupby("label", as_index=False)["weight_fraction"]
        .max()
        .sort_values("weight_fraction", ascending=False)
        .head(int(max_tissues))["label"]
        .astype(str)
        .tolist()
    )
    d = d[d["label"].astype(str).isin(tissue_order)]
    pivot = (
        d.pivot_table(index="query_cell_type", columns="label", values="weight_fraction", aggfunc="sum", fill_value=0.0)
        .reindex(index=states, columns=tissue_order, fill_value=0.0)
    )
    plt.figure(figsize=(max(7.0, 0.48 * len(tissue_order) + 2.8), max(4.8, 0.38 * len(states) + 1.8)))
    ax = plt.gca()
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.01, float(pivot.values.max())))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([wrap_label(x, width=34) for x in pivot.index], fontsize=9)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([wrap_label(x, width=14) for x in pivot.columns], rotation=45, ha="right", fontsize=9)
    ax.set_title("Which reference tissues support each query state?")
    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Neighbor vote fraction")
    cbar.ax.tick_params(labelsize=11)
    polish_axes(ax)
    savefig(path)


def select_local_reference_indices(
    neighbor_indices,
    *,
    neighbors_per_query: int = 50,
    max_reference_cells: int = 100_000,
):
    neigh = np.asarray(neighbor_indices, dtype=np.int64)
    if neigh.ndim != 2 or neigh.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(int(neighbors_per_query), neigh.shape[1])
    flat = neigh[:, :k].reshape(-1)
    flat = flat[flat >= 0]
    if len(flat) == 0:
        return np.empty(0, dtype=np.int64)
    values, counts = np.unique(flat, return_counts=True)
    order = np.lexsort((values, -counts))
    values = values[order]
    if len(values) > int(max_reference_cells):
        values = values[: int(max_reference_cells)]
    return np.sort(values.astype(np.int64, copy=False))


def reduce_local_umap(x, *, seed: int):
    try:
        from sklearn.decomposition import PCA

        n_pca = min(50, int(x.shape[0] - 1), int(x.shape[1]))
        x_reduced = PCA(n_components=n_pca, random_state=int(seed)).fit_transform(x) if n_pca >= 2 else x
    except Exception:
        x_reduced = x

    try:
        from umap import UMAP

        n_neighbors = min(30, max(2, int(x_reduced.shape[0] - 1)))
        reducer = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.15,
            metric="euclidean",
            random_state=int(seed),
        )
        return reducer.fit_transform(x_reduced), "UMAP"
    except Exception:
        try:
            from sklearn.decomposition import PCA

            return PCA(n_components=2, random_state=int(seed)).fit_transform(x_reduced), "PCA fallback"
        except Exception as exc:
            raise RuntimeError(f"Could not compute UMAP or PCA fallback: {exc}") from exc


def plot_local_reference_umap(
    *,
    manifest: Dict[str, Any],
    fig_dir: Path,
    neighbors_per_query: int = 50,
    max_reference_cells: int = 50_000,
    max_query_cells: int = 25_000,
    seed: int = 42,
) -> Optional[Path]:
    path = fig_dir / "06_local_reference_query_umap.png"
    paths = manifest.get("paths", {})
    neighbor_path = paths.get("query_reference_neighbors")
    if not neighbor_path or not Path(neighbor_path).exists():
        return None

    config = manifest.get("config", {})
    reference_manifest = manifest.get("reference_manifest", {})
    reference_paths = reference_manifest.get("paths", {})
    reference_config = reference_manifest.get("config", {})
    query_h5ad = config.get("query_h5ad")
    embed_key = config.get("embed_key", "X_state")
    metric = str(reference_config.get("metric", "l2"))
    reference_embeddings_path = reference_paths.get("reference_embeddings")
    if not query_h5ad or not Path(query_h5ad).exists() or not reference_embeddings_path:
        placeholder(path, "Local UMAP skipped because query h5ad or reference embeddings are unavailable.")
        return path

    z = np.load(neighbor_path)
    query_positions = np.asarray(z["query_position"], dtype=np.int64)
    neighbor_indices = np.asarray(z["neighbor_indices"], dtype=np.int64)
    if len(query_positions) == 0 or neighbor_indices.size == 0:
        placeholder(path, "Local UMAP skipped because the neighbor cache is empty.")
        return path

    rng = np.random.default_rng(int(seed))
    query_keep = np.arange(len(query_positions), dtype=np.int64)
    if len(query_keep) > int(max_query_cells):
        query_keep = np.sort(rng.choice(query_keep, size=int(max_query_cells), replace=False))
    query_positions_plot = query_positions[query_keep]
    neighbor_indices_plot = neighbor_indices[query_keep]
    reference_indices = select_local_reference_indices(
        neighbor_indices_plot,
        neighbors_per_query=int(neighbors_per_query),
        max_reference_cells=int(max_reference_cells),
    )
    if len(reference_indices) == 0:
        placeholder(path, "Local UMAP skipped because no valid reference neighbors were found.")
        return path

    reference_embeddings = np.load(reference_embeddings_path, mmap_mode="r")
    ref_x = np.asarray(reference_embeddings[reference_indices], dtype=np.float32)

    try:
        import anndata as ad
    except Exception:
        placeholder(path, "Local UMAP skipped because anndata is unavailable.")
        return path

    query_adata = ad.read_h5ad(query_h5ad, backed="r")
    try:
        query_x = read_obsm_rows(query_adata, embed_key, query_positions_plot, metric=metric)
    finally:
        if hasattr(query_adata, "file") and query_adata.file is not None:
            query_adata.file.close()

    x = np.vstack([ref_x, query_x]).astype(np.float32, copy=False)
    xy, method = reduce_local_umap(x, seed=int(seed))
    ref_xy = xy[: len(ref_x)]
    query_xy = xy[len(ref_x) :]

    plt.figure(figsize=(6.6, 5.2))
    ax = plt.gca()
    ax.scatter(
        ref_xy[:, 0],
        ref_xy[:, 1],
        s=7,
        color="#b8bcc4",
        alpha=0.36,
        linewidths=0,
        rasterized=True,
        label=f"Local Tahoe reference cells (n={len(ref_x):,})",
    )
    ax.scatter(
        query_xy[:, 0],
        query_xy[:, 1],
        s=10,
        color="dodgerblue",
        alpha=0.86,
        edgecolor="white",
        linewidth=0.15,
        rasterized=True,
        label=f"Query cells (n={len(query_x):,})",
    )
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title("Query cells over local Tahoe reference neighborhoods")
    '''
    ax.text(
        0.02,
        0.02,
        f"Reference cells are unique top-{int(neighbors_per_query)} KNN neighbors",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#444444",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
    )
    '''
    make_legend_opaque(ax, loc="best")
    polish_axes(ax)
    savefig(path)
    return path


def write_report_tables(report_dir: Path, state_scores, state_votes, tissue_votes, top_n_states: int) -> Dict[str, str]:
    table_dir = safe_mkdir(report_dir / "tables")
    sort_col = state_sort_column(state_scores)
    sorted_states = state_scores.sort_values([sort_col, "frac_cells_above_seen_99"], ascending=[False, False]).copy()
    top_outliers = sorted_states.head(int(top_n_states)).copy()
    top_state_votes = state_votes[state_votes["rank"].astype(int) <= 5].copy()
    top_tissue_votes = tissue_votes[tissue_votes["rank"].astype(int) <= 5].copy()
    paths = {
        "sorted_query_state_scores": table_dir / "sorted_query_state_scores.tsv",
        "top_query_state_scores": table_dir / "top_query_state_scores.tsv",
        "top_reference_state_votes": table_dir / "top_reference_state_votes.tsv",
        "top_reference_tissue_votes": table_dir / "top_reference_tissue_votes.tsv",
    }
    sorted_states.to_csv(paths["sorted_query_state_scores"], sep="\t", index=False)
    top_outliers.to_csv(paths["top_query_state_scores"], sep="\t", index=False)
    top_state_votes.to_csv(paths["top_reference_state_votes"], sep="\t", index=False)
    top_tissue_votes.to_csv(paths["top_reference_tissue_votes"], sep="\t", index=False)
    return {k: str(v) for k, v in paths.items()}


def write_summary(
    report_dir: Path,
    run_dir: Path,
    manifest: Dict[str, Any],
    state_scores,
    top_table,
    top_n_states: int,
    local_umap_path: Optional[Path] = None,
) -> None:
    metadata = manifest.get("metadata", {})
    cfg = manifest.get("config", {})
    ref_cfg = manifest.get("reference_manifest", {}).get("config", {})
    call_counts = state_scores["reference_support_call"].value_counts().to_dict()
    sort_col = state_sort_column(state_scores)
    show_cols = [
        "query_cell_type",
        "reference_support_call",
        sort_col,
        "median_local_density_ratio",
        "frac_cells_above_seen_99",
        "nearest_reference_state",
        "nearest_reference_tissue",
        "nearest_reference_state_weight_fraction",
    ]
    show_cols = [c for c in show_cols if c in top_table.columns]

    lines = [
        "# Embedding Manifold QC Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Report directory: `{report_dir}`",
        "",
        "## Executive summary",
        f"- Query h5ad: `{cfg.get('query_h5ad', 'unknown')}`",
        f"- Query state column: `{cfg.get('query_state_col', 'unknown')}`",
        f"- Reference h5ad: `{ref_cfg.get('reference_h5ad', 'unknown')}`",
        f"- Metric / k: `{metadata.get('reference_metric', 'unknown')}` / `{metadata.get('score_k', 'unknown')}`",
        f"- Query states scored: `{metadata.get('n_query_states', len(state_scores))}`",
        f"- Query cells scored: `{metadata.get('n_query_cells_scored', 'unknown')}`",
        f"- Support calls: `{call_counts}`",
        "",
        "## Highest OOD query states",
        markdown_table(top_table[show_cols] if show_cols else top_table, max_rows=int(top_n_states)),
        "",
        "## How to read the main table",
        "- `median_local_density_ratio` is the primary OOD score; larger means the query state is farther from local reference support.",
        "- `ood_percentile_vs_*` columns place each query state against reference calibration distributions.",
        "- `frac_cells_above_seen_99` is the fraction of scored query cells beyond the 99th percentile of seen reference cells.",
        "- `nearest_reference_tissue` is derived from the nearest reference state's `cell_name` joined to `cell_line_metadata.csv` `Organ`.",
        "",
        "## Figures",
        "- `figures/01_query_state_ood_percentiles.png`: calibrated state-level OOD percentiles.",
        "- `figures/02_cell_level_seen_percentiles_by_state.png`: cell-level percentile spread within each query state.",
        "- `figures/03_calibration_ecdf_with_query_states.png`: query scores over reference calibration ECDFs.",
        "- `figures/04_query_state_cell_tail_fractions.png`: fraction of cells above seen-reference p95/p99.",
        "- `figures/05_nearest_reference_tissue_support_heatmap.png`: distance-weighted reference tissue support.",
    ]
    if local_umap_path is not None:
        lines.append("- `figures/06_local_reference_query_umap.png`: query cells over their nearest Tahoe reference neighborhoods.")
    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_embedding_manifold_qc_report(
    *,
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    top_n_states: int = 40,
    local_umap_neighbors_per_query: int = 50,
    local_umap_max_reference_cells: int = 50_000,
    local_umap_max_query_cells: int = 25_000,
    local_umap_seed: int = 42,
) -> Dict[str, str]:
    ensure_report_dependencies()
    run_dir = Path(run_dir)
    report_dir = Path(output_dir) if output_dir else run_dir / "report"
    fig_dir = safe_mkdir(report_dir / "figures")
    safe_mkdir(report_dir / "tables")

    tables = load_tables(run_dir)
    manifest = tables["manifest"]
    state_scores = tables["state_scores"]
    cell_scores = tables["cell_scores"]
    state_votes = tables["state_votes"]
    tissue_votes = tables["tissue_votes"]
    calibration_state_scores = tables["calibration_state_scores"]

    if state_scores.empty:
        raise ValueError("query_state_scores is empty; no report can be generated.")

    plot_state_percentiles(state_scores, fig_dir, top_n_states=top_n_states)
    plot_cell_percentile_boxplot(cell_scores, state_scores, fig_dir, top_n_states=top_n_states)
    plot_calibration_overlay(calibration_state_scores, state_scores, fig_dir)
    plot_outlier_fraction(state_scores, fig_dir, top_n_states=top_n_states)
    plot_tissue_heatmap(tissue_votes, state_scores, fig_dir, top_n_states=top_n_states)
    local_umap_path = plot_local_reference_umap(
        manifest=manifest,
        fig_dir=fig_dir,
        neighbors_per_query=int(local_umap_neighbors_per_query),
        max_reference_cells=int(local_umap_max_reference_cells),
        max_query_cells=int(local_umap_max_query_cells),
        seed=int(local_umap_seed),
    )

    table_paths = write_report_tables(report_dir, state_scores, state_votes, tissue_votes, top_n_states)
    sorted_states = pd.read_csv(table_paths["sorted_query_state_scores"], sep="\t")
    top_table = sorted_states.head(int(top_n_states)).copy()
    write_summary(report_dir, run_dir, manifest, state_scores, top_table, top_n_states, local_umap_path=local_umap_path)

    return {
        "report_dir": str(report_dir),
        "summary": str(report_dir / "summary.md"),
        "sorted_query_state_scores": table_paths["sorted_query_state_scores"],
        "top_query_state_scores": table_paths["top_query_state_scores"],
        "top_reference_state_votes": table_paths["top_reference_state_votes"],
        "top_reference_tissue_votes": table_paths["top_reference_tissue_votes"],
        "state_percentiles": str(fig_dir / "01_query_state_ood_percentiles.png"),
        "cell_percentiles": str(fig_dir / "02_cell_level_seen_percentiles_by_state.png"),
        "calibration_ecdf": str(fig_dir / "03_calibration_ecdf_with_query_states.png"),
        "tail_fractions": str(fig_dir / "04_query_state_cell_tail_fractions.png"),
        "tissue_heatmap": str(fig_dir / "05_nearest_reference_tissue_support_heatmap.png"),
        "local_reference_query_umap": str(local_umap_path) if local_umap_path is not None else "",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate report figures for embedding manifold query QC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory produced by embedding_manifold_qc_analysis.py score-query.")
    p.add_argument("--output-dir", default=None, help="Report output directory. Default: <run-dir>/report.")
    p.add_argument("--top-n-states", type=int, default=40, help="Maximum query states shown in crowded figures.")
    p.add_argument(
        "--local-umap-neighbors-per-query",
        type=int,
        default=50,
        help="Number of nearest Tahoe neighbors per query cell used to build the local reference UMAP.",
    )
    p.add_argument(
        "--local-umap-max-reference-cells",
        type=int,
        default=50_000,
        help="Maximum unique Tahoe reference cells shown in the local neighborhood UMAP.",
    )
    p.add_argument(
        "--local-umap-max-query-cells",
        type=int,
        default=25_000,
        help="Maximum query cells shown in the local neighborhood UMAP.",
    )
    p.add_argument("--local-umap-seed", type=int, default=42, help="Random seed for local UMAP downsampling/layout.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = make_embedding_manifold_qc_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        top_n_states=args.top_n_states,
        local_umap_neighbors_per_query=args.local_umap_neighbors_per_query,
        local_umap_max_reference_cells=args.local_umap_max_reference_cells,
        local_umap_max_query_cells=args.local_umap_max_query_cells,
        local_umap_seed=args.local_umap_seed,
    )
    print("\n=== Embedding manifold QC report complete ===")
    print(f"summary: {out['summary']}")
    print(f"figures: {Path(out['state_percentiles']).parent}")
    print(f"tables:  {Path(out['sorted_query_state_scores']).parent}")


if __name__ == "__main__":
    main()
