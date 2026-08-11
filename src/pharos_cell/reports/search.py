#!/usr/bin/env python
"""
make_search_report.py

Generate a generic summary report for ST-SE sequential perturbation search outputs.

Input
-----
A search output directory produced by search.py containing:

    results.tsv
    checkpoint.pt
    search_config.used.yaml

Optional, but recommended for heterogeneity diagnostics:
    either:
        --target-npz J82_to_A172_states.npz
    or:
        --adata WT_256_per_cell_name.SE600M.h5ad --target-cell A-172 --cell-col cell_name --embed-key X_state

Outputs
-------
report/
    summary.md
    tables/
        top_paths.tsv
        best_by_depth.tsv
        drug_frequency.tsv
        drug_position_counts.tsv
        conversion_threshold_counts.tsv
        heterogeneity_diagnostics.tsv
    figures/
        01_best_score_by_depth.png
        02_top_path_trajectories.png
        03_delta_score_by_step.png
        04_drug_frequency_top_paths.png
        05_drug_position_heatmap.png
        06_path_similarity_heatmap.png
        07_variance_ratio_by_depth.png
        08_target_neighbor_coverage.png

Core plots
----------
1. Best Sinkhorn/energy score by depth
2. Top-path trajectories across sequential drug steps
3. Delta Sinkhorn improvement per step
4. Drug frequency across top paths
5. Drug × step-position heatmap
6. Path similarity heatmap
7. Variance ratio / population-collapse diagnostic
8. Target-neighbor coverage diagnostic

Notes
-----
Plots 7 and 8 require target embeddings. The script will skip those gracefully
if target embeddings are not provided.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib.pyplot as plt


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
plt.rcParams.update(PUBLICATION_RC)


# -----------------------------
# Parsing helpers
# -----------------------------


def parse_json_list(value) -> List[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    try:
        x = json.loads(str(value))
        if isinstance(x, list):
            return [str(v) for v in x]
    except Exception:
        pass
    if isinstance(value, str) and value.strip():
        return [x.strip() for x in value.split(" -> ") if x.strip()]
    return []


def safe_filename(text: str, max_len: int = 80) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in "-_.":
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    return out[:max_len].strip("_") or "item"


def load_yaml_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        x = yaml.safe_load(f)
    return x or {}


def load_target_embeddings(args) -> Optional[np.ndarray]:
    """
    Load target embeddings if supplied.

    Priority:
      1. --target-npz from data_loader.py save_npz()
      2. --adata + --target-cell using data_loader.py
    """
    if args.target_npz:
        z = np.load(args.target_npz, allow_pickle=True)
        if "target_embeddings" not in z:
            raise KeyError(f"{args.target_npz} does not contain 'target_embeddings'")
        return z["target_embeddings"].astype(np.float32, copy=False)

    if args.adata and args.target_cell:
        from ..data_loader import load_start_target_embeddings

        # start_cell is not actually needed for report diagnostics, but the loader
        # expects one. Use target_cell as both if no start_cell is given.
        start_cell = args.start_cell or args.target_cell
        pair = load_start_target_embeddings(
            h5ad_path=args.adata,
            start_cell=start_cell,
            target_cell=args.target_cell,
            cell_col=args.cell_col,
            embed_key=args.embed_key,
            start_sample=args.start_sample,
            target_sample=args.target_sample,
            seed=args.seed,
        )
        return pair.target_embeddings.astype(np.float32, copy=False)

    return None


# -----------------------------
# Tables
# -----------------------------


def add_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["path_list"] = df["path_json"].apply(parse_json_list) if "path_json" in df.columns else [[] for _ in range(len(df))]
    df["drug_name_list"] = (
        df["drug_names_json"].apply(parse_json_list) if "drug_names_json" in df.columns else [[] for _ in range(len(df))]
    )
    df["path_tuple"] = df["path_list"].apply(tuple)
    df["drug_name_tuple"] = df["drug_name_list"].apply(tuple)
    return df


def selection_sort_column(df: pd.DataFrame) -> str:
    """Use adjusted_score when available because it is the search selection score."""
    if "adjusted_score" in df.columns:
        vals = pd.to_numeric(df["adjusted_score"], errors="coerce")
        if vals.notna().any():
            return "adjusted_score"
    return "score_sinkhorn_ot"


def best_by_depth_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sort_col = selection_sort_column(df)
    for depth, g in df.groupby("depth"):
        g = g.sort_values(sort_col)
        best = g.iloc[0]
        rows.append(
            {
                "depth": int(depth),
                "n_paths_retained": int(len(g)),
                "best_score_sinkhorn_ot": float(best["score_sinkhorn_ot"]),
                "best_score_energy_distance": float(best["score_energy_distance"]),
                "best_adjusted_score": float(best.get("adjusted_score", best["score_sinkhorn_ot"])),
                "best_drug_name_string": best.get("drug_name_string", ""),
                "best_path_string": best.get("path_string", ""),
            }
        )
    return pd.DataFrame(rows).sort_values("depth")


def top_paths_table(df: pd.DataFrame, top_n: int = 25, final_depth_only: bool = False) -> pd.DataFrame:
    d = df.copy()
    if final_depth_only and len(d) > 0:
        d = d[d["depth"] == d["depth"].max()].copy()
    sort_col = selection_sort_column(d)
    cols = [
        "algorithm",
        "depth",
        "rank",
        "num_drugs",
        "drug_name_string",
        "path_string",
        "score_sinkhorn_ot",
        "score_energy_distance",
        "score_prefilter",
        "prefilter_metric",
        "adjusted_score",
        "delta_sinkhorn_from_parent",
        "robust_score_mean",
        "robust_score_std",
        "robust_score_worst",
        "robust_score_aggregate",
        "robust_n_samples",
        "start_cell",
        "target_cell",
    ]
    cols = [c for c in cols if c in d.columns]
    return d.sort_values(sort_col)[cols].head(top_n)


def drug_frequency_table(df: pd.DataFrame, top_n_paths: int = 50, final_depth_only: bool = False) -> pd.DataFrame:
    d = df.copy()
    if final_depth_only and len(d) > 0:
        d = d[d["depth"] == d["depth"].max()].copy()

    d = d.sort_values(selection_sort_column(d)).head(top_n_paths)

    counts: Dict[str, int] = {}
    path_counts: Dict[str, int] = {}
    for drugs in d["drug_name_list"]:
        for drug in drugs:
            counts[drug] = counts.get(drug, 0) + 1
        for drug in set(drugs):
            path_counts[drug] = path_counts.get(drug, 0) + 1

    rows = []
    for drug, n in counts.items():
        rows.append(
            {
                "drug_name": drug,
                "total_occurrences": n,
                "num_paths_with_drug": path_counts.get(drug, 0),
                "fraction_top_paths_with_drug": path_counts.get(drug, 0) / max(1, len(d)),
            }
        )

    return pd.DataFrame(rows).sort_values(["num_paths_with_drug", "total_occurrences"], ascending=False)


def drug_position_table(df: pd.DataFrame, top_n_paths: int = 50, final_depth_only: bool = False) -> pd.DataFrame:
    d = df.copy()
    if final_depth_only and len(d) > 0:
        d = d[d["depth"] == d["depth"].max()].copy()
    d = d.sort_values(selection_sort_column(d)).head(top_n_paths)

    rows = []
    for _, row in d.iterrows():
        for i, drug in enumerate(row["drug_name_list"], start=1):
            rows.append({"drug_name": drug, "step": i, "count": 1})

    if not rows:
        return pd.DataFrame(columns=["drug_name", "step", "count"])

    out = pd.DataFrame(rows)
    return out.groupby(["drug_name", "step"], as_index=False)["count"].sum()


def conversion_threshold_table(df: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    rows = []
    for depth, g in df.groupby("depth"):
        for th in thresholds:
            rows.append(
                {
                    "depth": int(depth),
                    "threshold": float(th),
                    "num_paths_at_or_below_threshold": int((g["score_sinkhorn_ot"] <= th).sum()),
                    "total_paths_retained": int(len(g)),
                    "fraction_paths_at_or_below_threshold": float((g["score_sinkhorn_ot"] <= th).mean()),
                }
            )
    return pd.DataFrame(rows).sort_values(["threshold", "depth"])


# -----------------------------
# Plot helpers
# -----------------------------


def savefig(path: Path):
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


def add_colorbar(im: Any, label: str) -> None:
    cbar = plt.colorbar(im)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=11)


def rescore_best_by_depth_full_dim(
    checkpoint_path: Path,
    target_embeddings: np.ndarray,
    *,
    device: Optional[str] = None,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
) -> pd.DataFrame:
    """
    Re-score checkpoint states in full embedding space (no projection).

    Diagnostic only: compares search-optimized projected scores with native 2058-dim Sinkhorn.
    """
    if not checkpoint_path.exists():
        return pd.DataFrame()

    from ..scoring import DistributionScorer

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    scorer = DistributionScorer(
        target_state=target_embeddings,
        device=device,
        normalize=True,
        sinkhorn_metric="cosine",
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        projection=None,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    depths = ckpt.get("depths", {})
    rows = []
    for depth_key, info in depths.items():
        depth = int(depth_key)
        if depth == 0:
            continue
        states = info.get("states", None)
        if states is None or not torch.is_tensor(states) or states.numel() == 0:
            continue
        states = states.to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            scores = scorer.sinkhorn(states).detach().cpu().numpy()
        rows.append({"depth": depth, "best_score_sinkhorn_full_dim": float(np.min(scores))})
    return pd.DataFrame(rows)


def plot_best_score_by_depth(best_df: pd.DataFrame, fig_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True)

    axes[0].plot(best_df["depth"], best_df["best_score_sinkhorn_ot"], marker="o", color="#2f6fed", lw=2.4, ms=6)
    axes[0].set_xlabel("Number of sequential drugs")
    axes[0].set_ylabel("Sinkhorn OT to target (lower is better)")
    axes[0].set_title("Best Sinkhorn OT by depth")

    if "best_score_energy_distance" in best_df.columns:
        axes[1].plot(best_df["depth"], best_df["best_score_energy_distance"], marker="o", color="#f28e2b", lw=2.4, ms=6)
        axes[1].set_ylabel("Energy distance to target")
    else:
        axes[1].text(0.5, 0.5, "No energy distance data", ha="center", va="center")
        axes[1].set_yticks([])
    axes[1].set_xlabel("Number of sequential drugs")
    axes[1].set_title("Best energy distance by depth")
    for ax in axes:
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        polish_axes(ax)

    fig.suptitle("Best distance to target by search depth")
    savefig(fig_dir / "01_best_score_by_depth.png")


def plot_top_path_trajectories(df: pd.DataFrame, fig_dir: Path, top_n: int = 10):
    """
    For top final-depth paths, plot prefix scores when the prefix is present in results.tsv.
    """
    final_depth = int(df["depth"].max())
    final = df[df["depth"] == final_depth].sort_values(selection_sort_column(df)).head(top_n)

    score_by_path = {tuple(row["path_list"]): float(row["score_sinkhorn_ot"]) for _, row in df.iterrows()}

    plt.figure(figsize=(6.8, 4.8))
    ax = plt.gca()
    for rank, (_, row) in enumerate(final.iterrows(), start=1):
        path = row["path_list"]
        xs = []
        ys = []
        for d in range(1, len(path) + 1):
            prefix = tuple(path[:d])
            xs.append(d)
            ys.append(score_by_path.get(prefix, np.nan))
        label = f"rank {rank}: " + " -> ".join(row["drug_name_list"][:3])
        if len(row["drug_name_list"]) > 3:
            label += " ..."
        ax.plot(xs, ys, marker="o", alpha=0.9, lw=2.0, ms=5.5, label=label)

    ax.set_xlabel("Drug step")
    ax.set_ylabel("Sinkhorn OT to target (lower is better)")
    ax.set_title(f"Top {len(final)} final-path trajectories")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    if len(final) <= 10:
        make_legend_opaque(ax, loc="best")
    polish_axes(ax)
    savefig(fig_dir / "02_top_path_trajectories.png")


def plot_delta_score_by_step(df: pd.DataFrame, fig_dir: Path):
    d = df.copy()
    d = d[np.isfinite(pd.to_numeric(d["delta_sinkhorn_from_parent"], errors="coerce"))]
    if d.empty:
        plt.figure(figsize=(6.2, 4.4))
        plt.text(0.5, 0.5, "No finite delta scores available", ha="center", va="center")
        plt.axis("off")
        savefig(fig_dir / "03_delta_score_by_step.png")
        return

    depths = sorted(d["depth"].unique())
    data = [d.loc[d["depth"] == depth, "delta_sinkhorn_from_parent"].astype(float).values for depth in depths]

    plt.figure(figsize=(6.2, 4.4))
    ax = plt.gca()
    ax.axhline(0, linestyle="--", linewidth=1.2, color="#444444")
    plt.boxplot(data, tick_labels=[str(x) for x in depths], showfliers=True)
    ax.set_xlabel("Drug step / depth")
    ax.set_ylabel("Delta Sinkhorn OT from parent")
    ax.set_title("Stepwise improvement distribution")
    ax.text(0.02, 0.96, "Negative values improve target fit", transform=ax.transAxes, ha="left", va="top", fontsize=11)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    polish_axes(ax)
    savefig(fig_dir / "03_delta_score_by_step.png")


def plot_drug_frequency(freq_df: pd.DataFrame, fig_dir: Path, top_n: int = 20):
    d = freq_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(6.6, max(4.0, 0.34 * len(d) + 1.3)))
    ax = plt.gca()
    if d.empty:
        plt.text(0.5, 0.5, "No drug frequency data", ha="center", va="center")
        plt.axis("off")
    else:
        ax.barh(d["drug_name"], d["num_paths_with_drug"], color="#2f6fed", edgecolor="#333333", linewidth=0.6)
        ax.set_xlabel("Number of top paths containing drug")
        ax.set_ylabel("Drug")
        ax.set_title("Most frequent drugs across top paths")
        ax.grid(axis="x", color="#e5e5e5", lw=0.8)
        polish_axes(ax)
    savefig(fig_dir / "04_drug_frequency_top_paths.png")


def plot_drug_position_heatmap(pos_df: pd.DataFrame, fig_dir: Path, top_n_drugs: int = 30):
    if pos_df.empty:
        plt.figure(figsize=(6.2, 4.4))
        plt.text(0.5, 0.5, "No drug position data", ha="center", va="center")
        plt.axis("off")
        savefig(fig_dir / "05_drug_position_heatmap.png")
        return

    top_drugs = (
        pos_df.groupby("drug_name")["count"].sum().sort_values(ascending=False).head(top_n_drugs).index.tolist()
    )
    mat = pos_df[pos_df["drug_name"].isin(top_drugs)].pivot_table(
        index="drug_name", columns="step", values="count", fill_value=0, aggfunc="sum"
    )
    mat = mat.loc[top_drugs]

    plt.figure(figsize=(6.4, max(4.0, 0.32 * len(mat) + 1.4)))
    ax = plt.gca()
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
    add_colorbar(im, "Count")
    plt.xticks(np.arange(mat.shape[1]), [str(c) for c in mat.columns])
    plt.yticks(np.arange(mat.shape[0]), mat.index)
    ax.set_xlabel("Step position")
    ax.set_ylabel("Drug")
    ax.set_title("Drug usage by position in top paths")
    polish_axes(ax)
    savefig(fig_dir / "05_drug_position_heatmap.png")


def path_jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    aa = set(a)
    bb = set(b)
    if not aa and not bb:
        return 1.0
    return len(aa & bb) / max(1, len(aa | bb))


def plot_path_similarity_heatmap(df: pd.DataFrame, fig_dir: Path, top_n: int = 30, final_depth_only: bool = True):
    d = df.copy()
    if final_depth_only and len(d) > 0:
        d = d[d["depth"] == d["depth"].max()].copy()
    d = d.sort_values(selection_sort_column(d)).head(top_n)

    paths = d["drug_name_list"].tolist()
    n = len(paths)

    if n == 0:
        plt.figure(figsize=(6.2, 4.4))
        plt.text(0.5, 0.5, "No paths available", ha="center", va="center")
        plt.axis("off")
        savefig(fig_dir / "06_path_similarity_heatmap.png")
        return

    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sim[i, j] = path_jaccard(paths[i], paths[j])

    labels = [f"rank {i+1}" for i in range(n)]

    plt.figure(figsize=(5.8, 5.2))
    ax = plt.gca()
    im = ax.imshow(sim, vmin=0, vmax=1, aspect="auto", cmap="magma")
    add_colorbar(im, "Jaccard similarity")
    plt.xticks(np.arange(n), labels, rotation=90, fontsize=7)
    plt.yticks(np.arange(n), labels, fontsize=7)
    ax.set_title("Similarity among top paths")
    polish_axes(ax)
    savefig(fig_dir / "06_path_similarity_heatmap.png")


# -----------------------------
# Heterogeneity diagnostics
# -----------------------------


def total_variance(x: torch.Tensor) -> torch.Tensor:
    """
    Total variance across embedding dimensions.
    x: [N, D]
    """
    return torch.var(x.float(), dim=0, unbiased=False).sum()


def nearest_target_coverage(
    state: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int = 1024,
) -> float:
    """
    Fraction of target cells that are nearest neighbors of at least one predicted cell.

    state:  [N, D]
    target: [M, D]

    Uses cosine distance on L2-normalized embeddings.
    """
    state = state.float()
    target = target.float()

    state = state / (torch.linalg.norm(state, dim=1, keepdim=True) + 1e-8)
    target = target / (torch.linalg.norm(target, dim=1, keepdim=True) + 1e-8)

    nearest = []
    for i in range(0, state.shape[0], chunk_size):
        x = state[i : i + chunk_size]
        sim = x @ target.T
        idx = torch.argmax(sim, dim=1)
        nearest.append(idx)

    nearest = torch.cat(nearest)
    unique = torch.unique(nearest).numel()
    return float(unique / target.shape[0])


def compute_heterogeneity_diagnostics(
    checkpoint_path: Path,
    target_embeddings: Optional[np.ndarray],
    max_nodes_per_depth: int = 50,
    device: Optional[str] = None,
) -> pd.DataFrame:
    if target_embeddings is None or not checkpoint_path.exists():
        return pd.DataFrame()

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    depths = ckpt.get("depths", {})

    target = torch.as_tensor(target_embeddings, dtype=torch.float32, device=device)
    target_var = float(total_variance(target).detach().cpu())

    rows = []
    for depth_key, info in depths.items():
        depth = int(depth_key)
        states = info.get("states", None)
        paths = info.get("paths", [])
        scores = info.get("scores_sinkhorn", [])

        if states is None or not torch.is_tensor(states) or states.numel() == 0:
            continue

        n_nodes = min(max_nodes_per_depth, states.shape[0])
        for i in range(n_nodes):
            state = states[i].to(device=device, dtype=torch.float32)
            var = float(total_variance(state).detach().cpu())
            coverage = nearest_target_coverage(state, target)

            rows.append(
                {
                    "depth": depth,
                    "rank": i + 1,
                    "score_sinkhorn_ot": float(scores[i]) if i < len(scores) else np.nan,
                    "variance_total": var,
                    "target_variance_total": target_var,
                    "variance_ratio_to_target": var / target_var if target_var > 0 else np.nan,
                    "target_neighbor_coverage": coverage,
                    "path_json": json.dumps(list(paths[i]), ensure_ascii=False) if i < len(paths) else "[]",
                }
            )

    return pd.DataFrame(rows)


def plot_variance_ratio(het_df: pd.DataFrame, fig_dir: Path):
    plt.figure(figsize=(6.2, 4.4))
    ax = plt.gca()
    if het_df.empty:
        plt.text(0.5, 0.5, "Target embeddings not provided; skipped", ha="center", va="center")
        plt.axis("off")
    else:
        depths = sorted(het_df["depth"].unique())
        data = [het_df.loc[het_df["depth"] == d, "variance_ratio_to_target"].values for d in depths]
        ax.axhline(1.0, linestyle="--", linewidth=1.2, color="#444444", label="target variance")
        ax.boxplot(data, tick_labels=[str(d) for d in depths], showfliers=True)
        ax.set_xlabel("Search depth")
        ax.set_ylabel("Predicted variance / target variance")
        ax.set_title("Population-collapse diagnostic")
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        make_legend_opaque(ax, loc="best")
        polish_axes(ax)
    savefig(fig_dir / "07_variance_ratio_by_depth.png")


def plot_target_neighbor_coverage(het_df: pd.DataFrame, fig_dir: Path):
    plt.figure(figsize=(6.2, 4.4))
    ax = plt.gca()
    if het_df.empty:
        plt.text(0.5, 0.5, "Target embeddings not provided; skipped", ha="center", va="center")
        plt.axis("off")
    else:
        depths = sorted(het_df["depth"].unique())
        data = [het_df.loc[het_df["depth"] == d, "target_neighbor_coverage"].values for d in depths]
        ax.boxplot(data, tick_labels=[str(d) for d in depths], showfliers=True)
        ax.set_xlabel("Search depth")
        ax.set_ylabel("Fraction of target cells covered")
        ax.set_title("Target-neighbor coverage")
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        polish_axes(ax)
    savefig(fig_dir / "08_target_neighbor_coverage.png")


# -----------------------------
# Summary markdown
# -----------------------------


def markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No rows._"
    d = df.head(max_rows).copy()
    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        s = str(x).replace("|", "\\|")
        if len(s) > 120:
            s = s[:117] + "..."
        return s

    headers = [str(c).replace("|", "\\|") for c in d.columns]
    rows = [[fmt(v) for v in row] for row in d.itertuples(index=False, name=None)]

    widths = []
    for j, h in enumerate(headers):
        max_cell = max([len(row[j]) for row in rows], default=0)
        widths.append(max(len(h), max_cell))

    header_line = "| " + " | ".join(h.ljust(widths[j]) for j, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    row_lines = [
        "| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + row_lines)


def write_summary_md(
    report_dir: Path,
    search_dir: Path,
    cfg: Dict[str, Any],
    df: pd.DataFrame,
    best_df: pd.DataFrame,
    top_df: pd.DataFrame,
    freq_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    het_df: pd.DataFrame,
    args,
):
    lines = []
    lines.append("# ST-SE Drug Path Search Report\n")
    lines.append(f"Search directory: `{search_dir}`\n")
    lines.append(f"Report directory: `{report_dir}`\n")

    algorithm = df["algorithm"].iloc[0] if "algorithm" in df.columns and len(df) else cfg.get("search", {}).get("algorithm", "unknown")
    max_depth_observed = int(df["depth"].max()) if len(df) else 0
    n_rows = len(df)
    sort_col = selection_sort_column(df)
    best_row = df.sort_values(sort_col).iloc[0] if len(df) else None

    lines.append("## Executive summary\n")
    lines.append(f"- Algorithm: `{algorithm}`")
    lines.append(f"- Observed maximum depth: `{max_depth_observed}`")
    lines.append(f"- Total retained path rows: `{n_rows}`")
    if best_row is not None:
        lines.append(f"- Best overall Sinkhorn OT: `{float(best_row['score_sinkhorn_ot']):.6g}`")
        if sort_col == "adjusted_score":
            lines.append(f"- Best overall selection/adjusted score: `{float(best_row['adjusted_score']):.6g}`")
        lines.append(f"- Best overall number of drugs: `{int(best_row['num_drugs'])}`")
        lines.append(f"- Best overall drug path: `{best_row.get('drug_name_string', '')}`")
    if args.conversion_threshold is not None:
        reached = int((df["score_sinkhorn_ot"] <= args.conversion_threshold).sum())
        lines.append(f"- Paths at or below conversion threshold `{args.conversion_threshold}`: `{reached}`")
    lines.append("")

    lines.append("## Best path per depth\n")
    lines.append(markdown_table(best_df, max_rows=20))
    lines.append("")

    lines.append("## Top paths\n")
    show_cols = [
        "depth",
        "rank",
        "num_drugs",
        "drug_name_string",
        "score_sinkhorn_ot",
        "score_energy_distance",
        "score_prefilter",
        "prefilter_metric",
        "delta_sinkhorn_from_parent",
    ]
    show_cols = [c for c in show_cols if c in top_df.columns]
    lines.append(markdown_table(top_df[show_cols], max_rows=15))
    lines.append("")

    lines.append("## Most frequent drugs across top paths\n")
    lines.append(markdown_table(freq_df.head(15), max_rows=15))
    lines.append("")

    if args.conversion_threshold is not None and not threshold_df.empty:
        lines.append("## Conversion threshold summary\n")
        lines.append(
            f"Conversion threshold was defined as Sinkhorn OT <= `{args.conversion_threshold}`. "
            "This is a user-defined analysis threshold; the current search algorithm does not automatically stop when it is reached unless that logic is added to search.py.\n"
        )
        lines.append(markdown_table(threshold_df, max_rows=30))
        lines.append("")

    lines.append("## Heterogeneity diagnostics\n")
    if het_df.empty:
        lines.append(
            "Target embeddings were not provided, so variance-ratio and target-neighbor coverage diagnostics were skipped. "
            "Rerun with `--target-npz` or `--adata --target-cell` to enable plots 7 and 8.\n"
        )
    else:
        summary = het_df.groupby("depth").agg(
            median_variance_ratio=("variance_ratio_to_target", "median"),
            median_target_neighbor_coverage=("target_neighbor_coverage", "median"),
        ).reset_index()
        lines.append(markdown_table(summary, max_rows=20))
        lines.append("")
        lines.append(
            "Interpretation: variance ratio near 1 suggests target-like population spread. "
            "Very low variance ratio may indicate population collapse. Higher target-neighbor coverage suggests the predicted cells cover more of the target distribution."
        )
        lines.append("")

    lines.append("## Figures\n")
    for i, name in enumerate(
        [
            "best score by depth",
            "top path trajectories",
            "delta score by step",
            "drug frequency",
            "drug position heatmap",
            "path similarity heatmap",
            "variance ratio by depth",
            "target-neighbor coverage",
        ],
        start=1,
    ):
        lines.append(f"- Figure {i}: {name}")

    (report_dir / "summary.md").write_text("\n".join(lines))


# -----------------------------
# Main
# -----------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser()

    p.add_argument("--search-dir", required=True, help="Directory containing results.tsv and checkpoint.pt")
    p.add_argument("--output-dir", default=None, help="Report output directory. Default: <search-dir>/report")

    # Optional target embeddings for heterogeneity diagnostics.
    p.add_argument("--target-npz", default=None, help="Optional .npz produced by data_loader.py save_npz()")
    p.add_argument("--adata", default=None, help="Optional h5ad with target SE embeddings")
    p.add_argument("--start-cell", default=None, help="Optional start cell; only needed if using --adata")
    p.add_argument("--target-cell", default=None, help="Optional target cell if using --adata")
    p.add_argument("--cell-col", default="cell_name")
    p.add_argument("--embed-key", default="X_state")
    p.add_argument("--start-sample", default="256")
    p.add_argument("--target-sample", default="all")
    p.add_argument("--seed", type=int, default=42)

    # Report controls.
    p.add_argument("--top-n-paths", type=int, default=25)
    p.add_argument("--top-n-final-trajectories", type=int, default=10)
    p.add_argument("--top-n-drugs", type=int, default=20)
    p.add_argument("--top-n-path-similarity", type=int, default=30)
    p.add_argument("--final-depth-only-frequency", action="store_true")
    p.add_argument("--conversion-threshold", type=float, default=None)
    p.add_argument("--extra-thresholds", default=None, help="Comma-separated extra Sinkhorn thresholds")

    # Heterogeneity controls.
    p.add_argument("--max-nodes-per-depth-diagnostics", type=int, default=50)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--projection-cache",
        default=None,
        help="If set (or if checkpoint has projection_metadata), add full-dim Sinkhorn diagnostic curve.",
    )

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    search_dir = Path(args.search_dir)
    report_dir = Path(args.output_dir) if args.output_dir else search_dir / "report"
    fig_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    results_path = search_dir / "results.tsv"
    config_path = search_dir / "search_config.used.yaml"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.tsv: {results_path}")

    df = pd.read_csv(results_path, sep="\t")
    if df.empty:
        raise ValueError(f"results.tsv is empty: {results_path}")

    df = add_parsed_columns(df)
    cfg = load_yaml_if_exists(config_path)
    checkpoint_path = search_dir / "checkpoint.pt"
    target_embeddings = load_target_embeddings(args)

    # Tables
    best_df = best_by_depth_table(df)
    use_full_dim_diag = False
    if args.projection_cache:
        use_full_dim_diag = True
    elif checkpoint_path.exists():
        ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        use_full_dim_diag = bool(ckpt_meta.get("projection_metadata"))

    if use_full_dim_diag and target_embeddings is not None:
        full_dim_df = rescore_best_by_depth_full_dim(
            checkpoint_path=checkpoint_path,
            target_embeddings=target_embeddings,
            device=args.device,
        )
        if not full_dim_df.empty:
            best_df = best_df.merge(full_dim_df, on="depth", how="left")

    top_df = top_paths_table(df, top_n=args.top_n_paths, final_depth_only=False)
    freq_df = drug_frequency_table(
        df,
        top_n_paths=args.top_n_paths,
        final_depth_only=args.final_depth_only_frequency,
    )
    pos_df = drug_position_table(
        df,
        top_n_paths=args.top_n_paths,
        final_depth_only=args.final_depth_only_frequency,
    )

    thresholds = []
    if args.conversion_threshold is not None:
        thresholds.append(args.conversion_threshold)
    if args.extra_thresholds:
        thresholds.extend([float(x.strip()) for x in args.extra_thresholds.split(",") if x.strip()])
    threshold_df = conversion_threshold_table(df, thresholds) if thresholds else pd.DataFrame()

    # Heterogeneity diagnostics
    het_df = compute_heterogeneity_diagnostics(
        checkpoint_path=checkpoint_path,
        target_embeddings=target_embeddings,
        max_nodes_per_depth=args.max_nodes_per_depth_diagnostics,
        device=args.device,
    )

    # Save tables
    best_df.to_csv(table_dir / "best_by_depth.tsv", sep="\t", index=False)
    top_df.to_csv(table_dir / "top_paths.tsv", sep="\t", index=False)
    freq_df.to_csv(table_dir / "drug_frequency.tsv", sep="\t", index=False)
    pos_df.to_csv(table_dir / "drug_position_counts.tsv", sep="\t", index=False)
    if not threshold_df.empty:
        threshold_df.to_csv(table_dir / "conversion_threshold_counts.tsv", sep="\t", index=False)
    if not het_df.empty:
        het_df.to_csv(table_dir / "heterogeneity_diagnostics.tsv", sep="\t", index=False)

    # Plots
    plot_best_score_by_depth(best_df, fig_dir)
    plot_top_path_trajectories(df, fig_dir, top_n=args.top_n_final_trajectories)
    plot_delta_score_by_step(df, fig_dir)
    plot_drug_frequency(freq_df, fig_dir, top_n=args.top_n_drugs)
    plot_drug_position_heatmap(pos_df, fig_dir, top_n_drugs=args.top_n_drugs)
    plot_path_similarity_heatmap(df, fig_dir, top_n=args.top_n_path_similarity, final_depth_only=True)
    plot_variance_ratio(het_df, fig_dir)
    plot_target_neighbor_coverage(het_df, fig_dir)

    # Summary
    write_summary_md(
        report_dir=report_dir,
        search_dir=search_dir,
        cfg=cfg,
        df=df,
        best_df=best_df,
        top_df=top_df,
        freq_df=freq_df,
        threshold_df=threshold_df,
        het_df=het_df,
        args=args,
    )

    print("\n=== Report complete ===")
    print(f"summary: {report_dir / 'summary.md'}")
    print(f"figures: {fig_dir}")
    print(f"tables:  {table_dir}")


if __name__ == "__main__":
    main()
