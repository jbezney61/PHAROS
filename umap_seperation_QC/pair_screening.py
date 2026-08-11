#!/usr/bin/env python
"""
pair_screening.py

Pre-search screening of start/target cell-line pairs using SE embeddings.

Steps:
  1. Keep cell lines with at least cells_per_line cells and sample exactly that many.
  2. L2-normalize embeddings (same convention as scoring.py / search).
  3. UMAP visualization.
  4. Per-cell KNN purity QC (exclude low-purity lines).
  5. Energy distance for all directed pairs among QC-passing lines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import yaml
from sklearn.neighbors import NearestNeighbors

from data_loader import _sample_indices
from scoring import energy_distance, l2_normalize_embeddings


PUBLICATION_RC = {
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
    "figure.dpi": 120,
    "savefig.dpi": 320,
}


def _set_publication_style(plt: Any) -> None:
    plt.rcParams.update(PUBLICATION_RC)


def _polish_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.0, length=3.5)


def _opaque_legend(ax: Any, **kwargs: Any) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    if legend is None:
        return
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)


@dataclass
class SubsampledEmbeddings:
    """All subsampled cells stacked with line labels."""

    X: np.ndarray  # [n_cells, emb_dim] float32, not yet normalized
    labels: np.ndarray  # [n_cells] str
    obs_names: np.ndarray  # [n_cells] str
    line_to_row_indices: Dict[str, np.ndarray]
    cell_col: str
    embed_key: str
    h5ad_path: str
    max_cells_per_line: int
    seed: int
    skipped_insufficient_cells: Dict[str, int] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.X.shape[0])

    @property
    def emb_dim(self) -> int:
        return int(self.X.shape[1])

    def metadata(self) -> Dict[str, Any]:
        return {
            "h5ad_path": self.h5ad_path,
            "cell_col": self.cell_col,
            "embed_key": self.embed_key,
            "max_cells_per_line": self.max_cells_per_line,
            "cells_per_line": self.max_cells_per_line,
            "seed": self.seed,
            "n_cells": self.n_cells,
            "emb_dim": self.emb_dim,
            "n_cell_lines": len(self.line_to_row_indices),
            "cell_lines": sorted(self.line_to_row_indices.keys()),
            "n_skipped_insufficient_cells": len(self.skipped_insufficient_cells),
            "skipped_insufficient_cells": self.skipped_insufficient_cells,
        }


def load_subsampled_embeddings(
    h5ad_path: str | Path,
    cell_col: str = "cell_name",
    embed_key: str = "X_state",
    max_cells_per_line: int = 256,
    seed: int = 42,
    min_cells_per_line: int = 1,
    dtype: np.dtype = np.float32,
) -> SubsampledEmbeddings:
    """
    Load SE embeddings and sample exactly max_cells_per_line cells per label.

    Cell lines with fewer than max_cells_per_line cells are skipped. This keeps
    every retained cell-line distribution the same size for batched scoring.
    """
    h5ad_path = Path(h5ad_path)
    rng = np.random.default_rng(seed)

    ad = sc.read_h5ad(h5ad_path)
    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available: {list(ad.obs.columns)}")
    if embed_key not in ad.obsm:
        raise KeyError(f"embed_key={embed_key!r} not found in adata.obsm. Available: {list(ad.obsm.keys())}")

    labels_all = ad.obs[cell_col].astype(str).values
    X_all = np.asarray(ad.obsm[embed_key], dtype=dtype)
    obs_names_all = ad.obs_names.astype(str).values

    line_to_indices: Dict[str, np.ndarray] = {}
    chosen_rows: List[np.ndarray] = []
    chosen_labels: List[str] = []
    chosen_obs: List[str] = []
    skipped_insufficient_cells: Dict[str, int] = {}

    for line in sorted(set(labels_all)):
        avail = np.where(labels_all == line)[0]
        if len(avail) < max_cells_per_line:
            skipped_insufficient_cells[str(line)] = int(len(avail))
            continue
        if len(avail) < min_cells_per_line:
            continue

        idx, _ = _sample_indices(
            avail,
            sample=max_cells_per_line,
            rng=rng,
            replace_if_needed=False,
            label=f"cell_line={line}",
        )
        line_to_indices[line] = idx
        chosen_rows.append(idx)
        chosen_labels.extend([line] * len(idx))
        chosen_obs.extend(obs_names_all[idx].tolist())

    if not chosen_rows:
        raise ValueError(
            "No cell lines had enough cells for exact sampling. "
            f"Requested max_cells_per_line={max_cells_per_line}. "
            "Lower --cells-per-line/--max-cells-per-line or check cell counts."
        )

    all_idx = np.concatenate(chosen_rows)
    X = X_all[all_idx].astype(dtype, copy=True)
    labels = np.asarray(chosen_labels, dtype=str)
    obs_names = np.asarray(chosen_obs, dtype=str)

    # Map global row indices in stacked X for each line.
    line_to_row: Dict[str, np.ndarray] = {}
    offset = 0
    for line in sorted(line_to_indices.keys()):
        n = len(line_to_indices[line])
        line_to_row[line] = np.arange(offset, offset + n, dtype=np.int64)
        offset += n

    return SubsampledEmbeddings(
        X=X,
        labels=labels,
        obs_names=obs_names,
        line_to_row_indices=line_to_row,
        cell_col=cell_col,
        embed_key=embed_key,
        h5ad_path=str(h5ad_path.resolve()),
        max_cells_per_line=max_cells_per_line,
        seed=seed,
        skipped_insufficient_cells=skipped_insufficient_cells,
    )


def normalize_embeddings_np(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows; matches scoring.py convention."""
    t = torch.as_tensor(X, dtype=torch.float32)
    t = l2_normalize_embeddings(t)
    return t.cpu().numpy()


def compute_knn_purity(
    X_norm: np.ndarray,
    labels: np.ndarray,
    knn_k: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-cell and per-cell-line KNN purity.

    purity_i = (# neighbors with same label) / k_eff
    k_eff = min(knn_k, n_line - 1) evaluated per cell based on its line size.
    """
    n = X_norm.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 cells for KNN purity.")

    # Query enough neighbors so that after dropping self each cell still has knn_k neighbors when possible.
    n_neighbors_query = min(n, max(knn_k + 1, 2))

    nn = NearestNeighbors(n_neighbors=n_neighbors_query, metric="cosine", algorithm="brute")
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm, return_distance=True)

    purities: List[float] = []
    k_effs: List[int] = []
    cell_lines: List[str] = []

    for i in range(n):
        line = str(labels[i])
        line_mask = labels == line
        n_line = int(line_mask.sum())
        k_eff = min(knn_k, max(1, n_line - 1))

        neigh_idx = indices[i]
        # Exclude self.
        neigh_idx = neigh_idx[neigh_idx != i]
        if len(neigh_idx) > k_eff:
            neigh_idx = neigh_idx[:k_eff]

        same = int(np.sum(labels[neigh_idx] == line))
        k_used = len(neigh_idx)
        purity = float(same / k_used) if k_used > 0 else 0.0

        purities.append(purity)
        k_effs.append(k_eff)
        cell_lines.append(line)

    per_cell = pd.DataFrame(
        {
            "obs_name": np.arange(n),  # placeholder overwritten by caller if needed
            "cell_line": cell_lines,
            "knn_purity": purities,
            "k_eff": k_effs,
        }
    )

    per_line = (
        per_cell.groupby("cell_line", as_index=False)
        .agg(
            n_cells=("knn_purity", "size"),
            k_eff=("k_eff", "max"),
            mean_knn_purity=("knn_purity", "mean"),
            median_knn_purity=("knn_purity", "median"),
        )
        .sort_values("cell_line")
    )

    return per_cell, per_line


def apply_purity_qc(
    per_line: pd.DataFrame,
    min_mean_knn_purity: float = 0.8,
) -> pd.DataFrame:
    """Add pass_qc column."""
    out = per_line.copy()
    out["pass_qc"] = out["mean_knn_purity"] >= float(min_mean_knn_purity)
    return out


def compute_pairwise_energy_distances(
    data: SubsampledEmbeddings,
    X_norm: np.ndarray,
    valid_lines: Sequence[str],
    *,
    device: torch.device,
    energy_batch_size: int = 32,
) -> pd.DataFrame:
    """
    Full energy distance for all directed pairs (start, target), start != target.
    """
    lines = list(valid_lines)
    pairs: List[Tuple[str, str]] = []
    for start in lines:
        for target in lines:
            if start != target:
                pairs.append((start, target))

    if not pairs:
        return pd.DataFrame(
            columns=[
                "start_cell_line",
                "target_cell_line",
                "distance_energy",
            ]
        )

    rows: List[Dict[str, Any]] = []

    for batch_start in range(0, len(pairs), energy_batch_size):
        batch_pairs = pairs[batch_start : batch_start + energy_batch_size]
        start_tensors = []
        target_tensors = []
        for s, t in batch_pairs:
            s_idx = data.line_to_row_indices[s]
            t_idx = data.line_to_row_indices[t]
            start_tensors.append(torch.as_tensor(X_norm[s_idx], dtype=torch.float32))
            target_tensors.append(torch.as_tensor(X_norm[t_idx], dtype=torch.float32))

        X_batch = torch.stack(start_tensors, dim=0).to(device)
        Y_batch = torch.stack(target_tensors, dim=0).to(device)

        scores = energy_distance(
            predicted_states=X_batch,
            target_state=Y_batch,
            normalize=True,
            device=device,
        ).detach().cpu().numpy()

        for (s, t), sc_val in zip(batch_pairs, scores):
            rows.append(
                {
                    "start_cell_line": s,
                    "target_cell_line": t,
                    "distance_energy": float(sc_val),
                }
            )

    return pd.DataFrame(rows)


def run_umap(
    X_norm: np.ndarray,
    labels: np.ndarray,
    obs_names: np.ndarray,
    *,
    seed: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
) -> pd.DataFrame:
    """Compute UMAP coordinates with scanpy (cosine neighbors on normalized X)."""
    ad = sc.AnnData(X=X_norm.copy())
    ad.obs["cell_line"] = labels.astype(str)
    ad.obs_names = obs_names.astype(str)

    sc.pp.neighbors(ad, n_neighbors=min(n_neighbors, max(2, ad.n_obs - 1)), use_rep="X", metric="cosine")
    sc.tl.umap(ad, min_dist=min_dist, random_state=seed)

    return pd.DataFrame(
        {
            "obs_name": ad.obs_names.astype(str),
            "cell_line": ad.obs["cell_line"].astype(str).values,
            "UMAP1": ad.obsm["X_umap"][:, 0],
            "UMAP2": ad.obsm["X_umap"][:, 1],
        }
    )


def build_symmetric_energy_matrix(pair_df: pd.DataFrame, valid_lines: Sequence[str]) -> pd.DataFrame:
    """Mean of (A->B, B->A) for heatmap; fallback to single direction if only one exists."""
    lines = list(valid_lines)
    mat = pd.DataFrame(np.nan, index=lines, columns=lines, dtype=float)

    for s in lines:
        for t in lines:
            if s == t:
                mat.loc[s, t] = 0.0
                continue
            d_st = pair_df.loc[
                (pair_df["start_cell_line"] == s) & (pair_df["target_cell_line"] == t),
                "distance_energy",
            ]
            d_ts = pair_df.loc[
                (pair_df["start_cell_line"] == t) & (pair_df["target_cell_line"] == s),
                "distance_energy",
            ]
            vals = []
            if len(d_st):
                vals.append(float(d_st.iloc[0]))
            if len(d_ts):
                vals.append(float(d_ts.iloc[0]))
            if vals:
                mat.loc[s, t] = float(np.mean(vals))

    return mat


def rank_pairs(
    pair_df: pd.DataFrame,
    per_line: pd.DataFrame,
) -> pd.DataFrame:
    """Attach purity columns and sort by energy distance ascending."""
    if pair_df.empty:
        return pair_df.copy()

    purity_map = per_line.set_index("cell_line")["mean_knn_purity"].to_dict()
    out = pair_df.copy()
    out["start_mean_purity"] = out["start_cell_line"].map(purity_map)
    out["target_mean_purity"] = out["target_cell_line"].map(purity_map)
    out = out.sort_values("distance_energy", ascending=True).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def write_summary_md(
    path: Path,
    *,
    metadata: Dict[str, Any],
    per_line: pd.DataFrame,
    skipped_counts: pd.DataFrame,
    ranked: pd.DataFrame,
    min_mean_knn_purity: float,
    top_n: int = 20,
) -> None:
    n_total = len(per_line)
    n_pass = int(per_line["pass_qc"].sum())
    n_fail = n_total - n_pass
    n_skipped_count = int(len(skipped_counts))

    lines = [
        "# Cell-line pair screening summary",
        "",
        "## Run metadata",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        f"## QC (mean KNN purity >= {min_mean_knn_purity:.0%})",
        f"- Cell lines sampled for screening: **{n_total}**",
        f"- Cell lines skipped before screening because they had fewer than {metadata['cells_per_line']} cells: **{n_skipped_count}**",
        f"- Passed QC: **{n_pass}**",
        f"- Failed QC: **{n_fail}**",
        "",
    ]

    if n_skipped_count:
        lines.append("### Excluded cell lines (insufficient cell count)")
        for _, row in skipped_counts.head(50).iterrows():
            lines.append(f"- `{row['cell_line']}` ({int(row['n_available_cells'])} cells)")
        if n_skipped_count > 50:
            lines.append(f"- ... {n_skipped_count - 50} more in `tables/cell_line_count_filter.tsv`")
        lines.append("")

    if n_fail:
        failed = per_line.loc[~per_line["pass_qc"], "cell_line"].tolist()
        lines.append("### Excluded cell lines (low purity)")
        for x in failed:
            mp = per_line.loc[per_line["cell_line"] == x, "mean_knn_purity"].iloc[0]
            lines.append(f"- `{x}` (mean purity = {mp:.3f})")
        lines.append("")

    lines.append("## Top directed pairs (lowest energy distance)")
    lines.append("")
    if ranked.empty:
        lines.append("_No pairs available (need at least two QC-passing lines)._")
    else:
        show = ranked.head(top_n)
        lines.append("| rank | start | target | energy distance | start purity | target purity |")
        lines.append("|------|-------|--------|-----------------|--------------|---------------|")
        for _, row in show.iterrows():
            lines.append(
                f"| {int(row['rank'])} | `{row['start_cell_line']}` | `{row['target_cell_line']}` | "
                f"{row['distance_energy']:.6g} | {row['start_mean_purity']:.3f} | "
                f"{row['target_mean_purity']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "- Lower **energy distance** = closer cell-state distributions in SE space.",
            "- Use `tables/ranked_pairs.tsv` to choose `--start-cell` and `--target-cell` for `cell_converter.py`.",
            "- Pair screening samples exactly `cells_per_line` cells per retained cell line.",
            "- Lines below the purity threshold are mixed in embedding space and are excluded from pair ranking.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def save_screening_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def plot_umap(umap_df: pd.DataFrame, path: Path, max_legend: int = 24) -> None:
    import matplotlib.pyplot as plt

    _set_publication_style(plt)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))

    lines = sorted(umap_df["cell_line"].unique())
    cmap = plt.get_cmap("tab20", max(len(lines), 1))

    for i, line in enumerate(lines):
        sub = umap_df[umap_df["cell_line"] == line]
        ax.scatter(
            sub["UMAP1"],
            sub["UMAP2"],
            s=9,
            alpha=0.82,
            linewidths=0,
            label=line,
            color=cmap(i % 20),
            rasterized=True,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Embeddings by cell line")
    ax.set_aspect("equal", adjustable="datalim")

    if len(lines) <= max_legend:
        ncol = min(3, max(1, int(np.ceil(len(lines) / 8))))

        leg = ax.legend(
            loc="best",              
            ncol=ncol,
            fontsize=12,
            markerscale=2.0,
            frameon=True,
            fancybox=True,
            borderpad=0.4,
            labelspacing=0.35,
            handletextpad=0.4,
            columnspacing=0.8,
        )

        frame = leg.get_frame()
        frame.set_facecolor((1, 1, 1, 0.72))   
        frame.set_edgecolor((0, 0, 0, 0.35))   
        frame.set_linewidth(0.8)

    else:
        ax.text(
            0.02,
            0.98,
            f"{len(lines)} cell lines",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            bbox=dict(
                facecolor=(1, 1, 1, 0.72),
                edgecolor=(0, 0, 0, 0.35),
                linewidth=0.8,
                boxstyle="round,pad=0.3",
            ),
        )

    _polish_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_purity_bar(per_line: pd.DataFrame, path: Path, min_mean_knn_purity: float) -> None:
    import matplotlib.pyplot as plt

    _set_publication_style(plt)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = per_line.sort_values("mean_knn_purity", ascending=True)
    fig_h = min(2, max(4.8, 0.1 * len(d) + 1))
    fig, ax = plt.subplots(figsize=(5, fig_h))

    colors = ["#238b45" if p else "#cb181d" for p in d["pass_qc"]]
    ax.barh(d["cell_line"], d["mean_knn_purity"], color=colors, height=0.72)
    ax.axvline(
        min_mean_knn_purity,
        color="#222222",
        linestyle="--",
        linewidth=1.2,
        label=f"QC threshold ({min_mean_knn_purity:.0%})",
    )
    ax.set_xlabel("Mean KNN purity")
    ax.set_title("Cell-line cohesion in SE space")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7.5 if len(d) > 35 else 8.5)
    _opaque_legend(ax, loc="lower right")
    _polish_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_energy_heatmap(matrix: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    _set_publication_style(plt)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(matrix)
    size = min(8.2, max(5.4, 0.13 * n + 3.8))
    fig, ax = plt.subplots(figsize=(size, size))

    data = matrix.values.astype(float)
    im = ax.imshow(data, aspect="equal", cmap="viridis_r")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    tick_fontsize = 6.2 if n > 35 else 7.5
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=tick_fontsize)
    ax.set_yticklabels(matrix.index, fontsize=tick_fontsize)
    ax.set_title("Symmetrized energy distance")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Energy distance")
    cbar.ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def run_pair_screening(
    h5ad_path: str | Path,
    output_dir: str | Path,
    *,
    cell_col: str = "cell_name",
    embed_key: str = "X_state",
    max_cells_per_line: int = 256,
    seed: int = 42,
    knn_k: int = 30,
    min_mean_knn_purity: float = 0.8,
    energy_batch_size: int = 32,
    device: Optional[str] = None,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.3,
    min_cells_per_line: int = 2,
    top_n_summary: int = 20,
) -> Dict[str, Any]:
    """
    Full screening pipeline; writes tables, figures, and summary under output_dir.
    """
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    cache_dir = output_dir / "cache"
    for d in (tables_dir, figures_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    data = load_subsampled_embeddings(
        h5ad_path=h5ad_path,
        cell_col=cell_col,
        embed_key=embed_key,
        max_cells_per_line=max_cells_per_line,
        seed=seed,
        min_cells_per_line=min_cells_per_line,
    )

    X_norm = normalize_embeddings_np(data.X)

    per_cell, per_line = compute_knn_purity(X_norm, data.labels, knn_k=knn_k)
    per_cell["obs_name"] = data.obs_names
    per_line = apply_purity_qc(per_line, min_mean_knn_purity=min_mean_knn_purity)

    valid_lines = per_line.loc[per_line["pass_qc"], "cell_line"].tolist()

    pair_df = compute_pairwise_energy_distances(
        data,
        X_norm,
        valid_lines,
        device=device,
        energy_batch_size=energy_batch_size,
    )

    ranked = rank_pairs(pair_df, per_line)

    umap_df = run_umap(
        X_norm,
        data.labels,
        data.obs_names,
        seed=seed,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
    )

    # Attach purity to all pairs table.
    if not pair_df.empty:
        pair_all = rank_pairs(pair_df, per_line).drop(columns=["rank"], errors="ignore")
    else:
        pair_all = pair_df.copy()

    skipped_counts = pd.DataFrame(
        [
            {"cell_line": line, "n_available_cells": n, "required_cells_per_line": max_cells_per_line}
            for line, n in sorted(data.skipped_insufficient_cells.items())
        ]
    )

    per_cell.to_csv(tables_dir / "per_cell_knn_purity.tsv", sep="\t", index=False)
    per_line.to_csv(tables_dir / "cell_line_qc.tsv", sep="\t", index=False)
    skipped_counts.to_csv(tables_dir / "cell_line_count_filter.tsv", sep="\t", index=False)
    pair_all.to_csv(tables_dir / "pair_distances_all.tsv", sep="\t", index=False)
    ranked.to_csv(tables_dir / "ranked_pairs.tsv", sep="\t", index=False)
    umap_df.to_csv(cache_dir / "umap_coordinates.tsv", sep="\t", index=False)

    np.savez_compressed(
        cache_dir / "subsampled_embeddings.npz",
        X=data.X.astype(np.float32),
        X_norm=X_norm.astype(np.float32),
        labels=data.labels.astype(str),
        obs_names=data.obs_names.astype(str),
        metadata=json.dumps(data.metadata()),
    )

    if len(valid_lines) >= 2 and not pair_df.empty:
        sym = build_symmetric_energy_matrix(pair_df, valid_lines)
        sym.to_csv(tables_dir / "energy_distance_symmetric.tsv", sep="\t")
        plot_energy_heatmap(sym, figures_dir / "03_distance_heatmap.png")

    plot_umap(umap_df, figures_dir / "01_umap_by_cell_line.png")
    plot_purity_bar(per_line, figures_dir / "02_knn_purity_by_cell_line.png", min_mean_knn_purity)

    cfg = {
        "h5ad_path": str(Path(h5ad_path).resolve()),
        "cell_col": cell_col,
        "embed_key": embed_key,
        "max_cells_per_line": max_cells_per_line,
        "cells_per_line": max_cells_per_line,
        "seed": seed,
        "knn_k": knn_k,
        "min_mean_knn_purity": min_mean_knn_purity,
        "energy_batch_size": energy_batch_size,
        "device": str(device),
        "umap_n_neighbors": umap_n_neighbors,
        "umap_min_dist": umap_min_dist,
        "min_cells_per_line": min_cells_per_line,
        "n_cell_lines_subsampled": len(data.line_to_row_indices),
        "n_cell_lines_skipped_insufficient_cells": len(skipped_counts),
        "n_cell_lines_pass_qc": len(valid_lines),
        "n_directed_pairs": len(pair_df),
    }
    save_screening_config(cfg, output_dir / "screening_config.used.yaml")

    write_summary_md(
        output_dir / "summary.md",
        metadata=cfg,
        per_line=per_line,
        skipped_counts=skipped_counts,
        ranked=ranked,
        min_mean_knn_purity=min_mean_knn_purity,
        top_n=top_n_summary,
    )

    return {
        "output_dir": str(output_dir),
        "cell_line_qc": per_line,
        "cell_line_count_filter": skipped_counts,
        "ranked_pairs": ranked,
        "pair_distances_all": pair_all,
        "config": cfg,
    }
