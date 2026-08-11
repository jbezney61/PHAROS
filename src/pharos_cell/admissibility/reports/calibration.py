#!/usr/bin/env python
"""
make_target_calibration_qc_report.py

Generate target calibration QC figures from target_calibration_qc_analysis.py
outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


np = None
pd = None
plt = None

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


def setup_logger(report_dir: Path) -> logging.Logger:
    report_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("target_calibration_qc_report")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(report_dir / "target_calibration_qc_report.log", mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path):
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def polish_axes(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", width=1.1, length=4)


def placeholder(path: Path, message: str, *, figsize: Tuple[float, float] = (7.0, 4.2)) -> None:
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    ax.axis("off")
    savefig(path)


def wrap_label(label: Any, width: int = 22) -> str:
    text = str(label)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False) or [text])


def finite_values(values: Sequence[float]):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def load_manifest(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "target_calibration_qc_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing target calibration QC manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_tables(run_dir: Path) -> Dict[str, Any]:
    manifest = load_manifest(run_dir)
    paths = manifest.get("paths", {})
    required = {
        "scores": paths.get("scores"),
        "cell_line_summary": paths.get("cell_line_summary"),
        "drug_summary": paths.get("drug_summary"),
        "matched_targets": paths.get("matched_targets"),
    }
    missing = [name for name, path in required.items() if not path or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing target calibration QC tables: {missing}")
    return {
        "manifest": manifest,
        "scores": read_table(required["scores"]),
        "cell_line_summary": read_table(required["cell_line_summary"]),
        "drug_summary": read_table(required["drug_summary"]),
        "matched_targets": read_table(required["matched_targets"]),
    }


def global_quantiles(scores) -> Tuple[float, float]:
    values = finite_values(scores["sinkhorn_ot"])
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))


def ensure_calibration_mode_column(df, default: str = "raw"):
    d = df.copy()
    if "calibration_mode" not in d.columns:
        d["calibration_mode"] = default
    d["calibration_mode"] = d["calibration_mode"].astype(str).str.strip().str.replace("-", "_", regex=False)
    d.loc[d["calibration_mode"].isin(["", "nan", "none", "null"]) | d["calibration_mode"].isna(), "calibration_mode"] = default
    return d


def calibration_modes_in_scores(scores) -> Sequence[str]:
    present = [str(x) for x in scores["calibration_mode"].dropna().unique().tolist()]
    preferred = ["raw", "dmso_start_only", "dmso_adapter"]
    ordered = [mode for mode in preferred if mode in present]
    ordered.extend(sorted(mode for mode in present if mode not in set(ordered)))
    return ordered


def calibration_mode_label(mode: str) -> str:
    labels = {
        "raw": "raw SE target space",
        "dmso_start_only": "DMSO-adapted start, raw target",
        "dmso_adapter": "DMSO-adapted start and target",
    }
    return labels.get(str(mode), str(mode).replace("_", " "))


def add_conversion_percent_columns(scores):
    d = ensure_calibration_mode_column(scores)
    d["sinkhorn_ot"] = pd.to_numeric(d["sinkhorn_ot"], errors="coerce")
    baseline_col = (
        "baseline_start_to_target_sinkhorn_ot"
        if "baseline_start_to_target_sinkhorn_ot" in d.columns
        else "baseline_wt_to_target_sinkhorn_ot"
    )
    d[baseline_col] = pd.to_numeric(
        d.get(baseline_col, np.nan),
        errors="coerce",
    )
    if "baseline_wt_to_target_sinkhorn_ot" not in d.columns:
        d["baseline_wt_to_target_sinkhorn_ot"] = d[baseline_col]
    if "baseline_start_to_target_sinkhorn_ot" not in d.columns:
        d["baseline_start_to_target_sinkhorn_ot"] = d[baseline_col]
    baseline = d[baseline_col].astype(float)
    predicted = d["sinkhorn_ot"].astype(float)
    valid = np.isfinite(baseline) & np.isfinite(predicted) & (baseline > 1e-12)
    remaining = np.full(len(d), np.nan, dtype=float)
    closed = np.full(len(d), np.nan, dtype=float)
    remaining[valid.to_numpy()] = 100.0 * predicted[valid].to_numpy() / baseline[valid].to_numpy()
    closed[valid.to_numpy()] = 100.0 * (baseline[valid].to_numpy() - predicted[valid].to_numpy()) / baseline[valid].to_numpy()
    d["remaining_ot_percent_of_baseline"] = remaining
    d["percent_total_ot_closed"] = closed
    d["percent_total_ot_closed_clipped_0_100"] = np.clip(closed, 0.0, 100.0)
    return d


def plot_kde_distribution(scores, fig_dir: Path, mode_label: str = "") -> Path:
    path = fig_dir / "01_sinkhorn_ot_distribution_kde.png"
    if scores.empty or "sinkhorn_ot" not in scores.columns:
        placeholder(path, "No Sinkhorn OT scores available.")
        return path
    values = finite_values(scores["sinkhorn_ot"])
    if len(values) == 0:
        placeholder(path, "No finite Sinkhorn OT scores available.")
        return path
    q05, q95 = float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))

    plt.figure(figsize=(6.2, 4.5))
    ax = plt.gca()
    x_min, x_max = float(values.min()), float(values.max())
    pad = max((x_max - x_min) * 0.08, 1e-4)
    xs = np.linspace(x_min - pad, x_max + pad, 512)
    plotted_kde = False
    if len(np.unique(values)) >= 3:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(values)
            ax.plot(xs, kde(xs), color="#1f77b4", linewidth=2.6)
            ax.fill_between(xs, kde(xs), color="#9ecae1", alpha=0.62)
            plotted_kde = True
        except Exception:
            plotted_kde = False
    if not plotted_kde:
        ax.hist(values, bins=35, density=True, color="#9ecae1", edgecolor="#1f77b4", alpha=0.82)

    for q, label, y_frac in [(q05, "5%", 0.86), (q95, "95%", 0.74)]:
        ax.axvline(q, color="#111111", linestyle=":", linewidth=1.7)
        ymax = ax.get_ylim()[1]
        ax.text(
            q,
            ymax * y_frac,
            f"{label} = {q:.4g}",
            rotation=90,
            ha="right" if label == "5%" else "left",
            va="center",
            fontsize=11,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
    ax.set_xlabel("Sinkhorn OT to observed target (lower is better)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted-to-actual 5uM Sinkhorn OT")
    ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.75)
    polish_axes(ax)
    savefig(path)
    return path


def plot_conversion_percent_kde(scores, fig_dir: Path, mode_label: str = "") -> Path:
    path = fig_dir / "02_percent_total_ot_closed_kde.png"
    if scores.empty or "percent_total_ot_closed" not in scores.columns:
        placeholder(path, "No percent-total-OT-closed values available.")
        return path
    values = finite_values(scores["percent_total_ot_closed"])
    if len(values) == 0:
        placeholder(path, "No finite percent-total-OT-closed values available.")
        return path
    q05, q50, q95 = float(np.quantile(values, 0.05)), float(np.quantile(values, 0.50)), float(np.quantile(values, 0.95))

    plt.figure(figsize=(6.2, 4.5))
    ax = plt.gca()
    x_min, x_max = float(values.min()), float(values.max())
    x_min = min(x_min, 0.0)
    x_max = max(x_max, 100.0)
    pad = max((x_max - x_min) * 0.08, 1.0)
    xs = np.linspace(x_min - pad, x_max + pad, 512)
    plotted_kde = False
    if len(np.unique(values)) >= 3:
        try:
            from scipy.stats import gaussian_kde

            kde = gaussian_kde(values)
            ys = kde(xs)
            ax.plot(xs, ys, color="#2ca25f", linewidth=2.6)
            ax.fill_between(xs, ys, color="#a1d99b", alpha=0.62)
            plotted_kde = True
        except Exception:
            plotted_kde = False
    if not plotted_kde:
        ax.hist(values, bins=35, density=True, color="#a1d99b", edgecolor="#2ca25f", alpha=0.82)

    ymax = ax.get_ylim()[1]
    for x, label in [(0.0, "0% no gain"), (100.0, "100% target reached")]:
        ax.axvline(x, color="#444444", linestyle=":", linewidth=1.4)
        ax.text(
            x,
            ymax * 0.86,
            label,
            rotation=90,
            ha="right" if x == 0.0 else "left",
            va="center",
            fontsize=10,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
        )
    for x, label, y_frac in [(q05, "5%", 0.70), (q50, "median", 0.58), (q95, "95%", 0.46)]:
        ax.axvline(x, color="#111111", linestyle="--", linewidth=1.2, alpha=0.78)
        ax.text(
            x,
            ymax * y_frac,
            f"{label} = {x:.3g}%",
            rotation=90,
            ha="left",
            va="center",
            fontsize=10,
            color="#111111",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.68, "pad": 1.0},
        )
    ax.set_xlabel("Percent of baseline start-to-target OT closed")
    ax.set_ylabel("Density")
    ax.set_title("Percent of total conversion distance closed")
    ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.75)
    polish_axes(ax)
    savefig(path)
    return path


def ranked_groups(scores, group_col: str, rank_stat: str, ascending: bool = True):
    d = scores.copy()
    d["sinkhorn_ot"] = pd.to_numeric(d["sinkhorn_ot"], errors="coerce")
    agg = "mean" if rank_stat == "mean" else "median"
    rank = (
        d.groupby(group_col, as_index=False)["sinkhorn_ot"]
        .agg(agg)
        .rename(columns={"sinkhorn_ot": "rank_value"})
        .sort_values(["rank_value", group_col], ascending=[ascending, True])
    )
    return rank[group_col].astype(str).tolist()


def plot_group_violin(
    scores,
    *,
    group_col: str,
    order: Sequence[str],
    output_path: Path,
    title: str,
    xlabel: str,
    q05: float,
    q95: float,
    width_per_group: float = 0.34,
    label_width: int = 18,
) -> Path:
    if scores.empty or not order:
        placeholder(output_path, "No scores available.")
        return output_path
    d = scores.copy()
    d["sinkhorn_ot"] = pd.to_numeric(d["sinkhorn_ot"], errors="coerce")
    d[group_col] = d[group_col].astype(str)
    data = [finite_values(d.loc[d[group_col] == group, "sinkhorn_ot"]) for group in order]
    keep = [(group, vals) for group, vals in zip(order, data) if len(vals)]
    if not keep:
        placeholder(output_path, "No finite grouped Sinkhorn OT scores available.")
        return output_path
    order = [x[0] for x in keep]
    data = [x[1] for x in keep]

    plt.figure(figsize=(max(7.0, width_per_group * len(order) + 2.2), 5.0))
    ax = plt.gca()
    parts = ax.violinplot(data, positions=np.arange(len(order)), showmedians=True, showextrema=False)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.18, 0.88, len(order)))
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#263238")
        body.set_linewidth(0.65)
        body.set_alpha(0.9)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#111111")
        parts["cmedians"].set_linewidth(1.3)
    for q, label in [(q05, "global 5%"), (q95, "global 95%")]:
        if np.isfinite(q):
            ax.axhline(q, color="#111111", linestyle=":", linewidth=1.3)
            ax.text(
                len(order) - 0.55,
                q,
                f" {label}: {q:.4g}",
                ha="left",
                va="center",
                fontsize=10,
                color="#111111",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
            )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([wrap_label(x, width=label_width) for x in order], rotation=90, ha="center", fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Sinkhorn OT to observed target (lower is better)")
    ax.set_title(title)
    ax.grid(axis="y", color="#d0d7de", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    polish_axes(ax)
    savefig(output_path)
    return output_path


def plot_cell_line_violin(scores, fig_dir: Path, rank_stat: str, q05: float, q95: float, mode_label: str = "") -> Path:
    order = ranked_groups(scores, "cell_type", rank_stat=rank_stat, ascending=True)
    title_prefix = f"Target calibration ({mode_label})" if mode_label else "Target calibration"
    return plot_group_violin(
        scores,
        group_col="cell_type",
        order=order,
        output_path=fig_dir / "03_cell_line_ranked_sinkhorn_ot_violin.png",
        title=f"{title_prefix} by cell line, ranked by {rank_stat} OT",
        xlabel="Cell line",
        q05=q05,
        q95=q95,
        width_per_group=0.32,
        label_width=14,
    )


def plot_top_bottom_drug_violins(
    scores,
    fig_dir: Path,
    rank_stat: str,
    q05: float,
    q95: float,
    top_n: int,
    mode_label: str = "",
) -> Tuple[Path, Path]:
    rank_order_high = ranked_groups(scores, "drug_name", rank_stat=rank_stat, ascending=False)[: int(top_n)]
    rank_order_low = ranked_groups(scores, "drug_name", rank_stat=rank_stat, ascending=True)[: int(top_n)]
    title_suffix = f" ({mode_label})" if mode_label else ""
    high_path = plot_group_violin(
        scores,
        group_col="drug_name",
        order=rank_order_high,
        output_path=fig_dir / "04_top_high_ot_drugs_sinkhorn_violin.png",
        title=f"Highest {top_n} drugs by {rank_stat} predicted-to-target OT{title_suffix}",
        xlabel="Drug",
        q05=q05,
        q95=q95,
        width_per_group=0.72,
        label_width=18,
    )
    low_path = plot_group_violin(
        scores,
        group_col="drug_name",
        order=rank_order_low,
        output_path=fig_dir / "05_bottom_low_ot_drugs_sinkhorn_violin.png",
        title=f"Lowest {top_n} drugs by {rank_stat} predicted-to-target OT{title_suffix}",
        xlabel="Drug",
        q05=q05,
        q95=q95,
        width_per_group=0.72,
        label_width=18,
    )
    return high_path, low_path


def markdown_table(df, max_rows: int = 16) -> str:
    if df is None or df.empty:
        return "_No rows._"
    d = df.head(int(max_rows)).copy()

    def fmt(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float):
            return f"{x:.6g}"
        s = str(x).replace("|", "\\|")
        return s[:137] + "..." if len(s) > 140 else s

    headers = [str(c).replace("|", "\\|") for c in d.columns]
    rows = [[fmt(v) for v in row] for row in d.itertuples(index=False, name=None)]
    widths = [max(len(h), max([len(row[j]) for row in rows], default=0)) for j, h in enumerate(headers)]
    header = "| " + " | ".join(h.ljust(widths[j]) for j, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def write_report_tables(report_dir: Path, scores, cell_summary, drug_summary, rank_stat: str, top_n_drugs: int) -> Dict[str, str]:
    table_dir = safe_mkdir(report_dir / "tables")
    score_col = f"{rank_stat}_sinkhorn_ot"
    scores = ensure_calibration_mode_column(scores)
    cell_summary = ensure_calibration_mode_column(cell_summary if cell_summary is not None else pd.DataFrame())
    drug_summary = ensure_calibration_mode_column(drug_summary if drug_summary is not None else pd.DataFrame())
    cell_sorted = cell_summary.sort_values(["calibration_mode", score_col], ascending=[True, True]).copy() if score_col in cell_summary.columns else cell_summary.copy()
    drug_high = drug_summary.sort_values(["calibration_mode", score_col], ascending=[True, False]).copy() if score_col in drug_summary.columns else drug_summary.copy()
    drug_low = drug_summary.sort_values(["calibration_mode", score_col], ascending=[True, True]).copy() if score_col in drug_summary.columns else drug_summary.copy()
    paths = {
        "scores_with_conversion_percent": table_dir / "target_calibration_scores_with_conversion_percent.tsv",
        "cell_line_ranked_summary": table_dir / "cell_line_ranked_summary.tsv",
        "top_high_ot_drugs": table_dir / "top_high_ot_drugs.tsv",
        "bottom_low_ot_drugs": table_dir / "bottom_low_ot_drugs.tsv",
    }
    scores.to_csv(paths["scores_with_conversion_percent"], sep="\t", index=False)
    cell_sorted.to_csv(paths["cell_line_ranked_summary"], sep="\t", index=False)
    drug_high.to_csv(paths["top_high_ot_drugs"], sep="\t", index=False)
    drug_low.to_csv(paths["bottom_low_ot_drugs"], sep="\t", index=False)

    for mode in calibration_modes_in_scores(scores):
        mode_table_dir = safe_mkdir(table_dir / mode)
        mode_cell = cell_summary[cell_summary["calibration_mode"].astype(str) == mode].copy()
        mode_drug = drug_summary[drug_summary["calibration_mode"].astype(str) == mode].copy()
        if score_col in mode_cell.columns:
            mode_cell = mode_cell.sort_values(score_col, ascending=True)
        if score_col in mode_drug.columns:
            mode_high = mode_drug.sort_values(score_col, ascending=False).head(int(top_n_drugs))
            mode_low = mode_drug.sort_values(score_col, ascending=True).head(int(top_n_drugs))
        else:
            mode_high = mode_drug.head(int(top_n_drugs))
            mode_low = mode_drug.head(int(top_n_drugs))
        mode_paths = {
            f"{mode}_cell_line_ranked_summary": mode_table_dir / "cell_line_ranked_summary.tsv",
            f"{mode}_top_high_ot_drugs": mode_table_dir / "top_high_ot_drugs.tsv",
            f"{mode}_bottom_low_ot_drugs": mode_table_dir / "bottom_low_ot_drugs.tsv",
        }
        mode_cell.to_csv(mode_paths[f"{mode}_cell_line_ranked_summary"], sep="\t", index=False)
        mode_high.to_csv(mode_paths[f"{mode}_top_high_ot_drugs"], sep="\t", index=False)
        mode_low.to_csv(mode_paths[f"{mode}_bottom_low_ot_drugs"], sep="\t", index=False)
        paths.update(mode_paths)
    return {k: str(v) for k, v in paths.items()}


def write_summary(
    report_dir: Path,
    run_dir: Path,
    manifest: Dict[str, Any],
    scores,
    cell_summary,
    drug_summary,
    figure_paths: Dict[str, str],
    table_paths: Dict[str, str],
) -> Path:
    metadata = manifest.get("metadata", {})
    global_summary = metadata.get("global_summary", {})
    scores = ensure_calibration_mode_column(scores)
    cell_summary = ensure_calibration_mode_column(cell_summary if cell_summary is not None else pd.DataFrame())
    drug_summary = ensure_calibration_mode_column(drug_summary if drug_summary is not None else pd.DataFrame())
    modes = calibration_modes_in_scores(scores)

    mode_lines = []
    for mode in modes:
        mode_scores = scores[scores["calibration_mode"].astype(str) == mode]
        ot_values = finite_values(mode_scores["sinkhorn_ot"])
        percent_values = finite_values(mode_scores["percent_total_ot_closed"]) if "percent_total_ot_closed" in mode_scores.columns else np.asarray([])
        mode_lines.append(
            "- "
            f"`{mode}` ({calibration_mode_label(mode)}): "
            f"n=`{len(mode_scores)}`, "
            f"median OT=`{float(np.median(ot_values)) if len(ot_values) else 'unknown'}`, "
            f"5%/95% OT=`{float(np.quantile(ot_values, 0.05)) if len(ot_values) else 'unknown'}`/"
            f"`{float(np.quantile(ot_values, 0.95)) if len(ot_values) else 'unknown'}`, "
            f"median % closed=`{float(np.median(percent_values)) if len(percent_values) else 'unknown'}`"
        )

    lines = [
        "# Target Calibration QC Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Report directory: `{report_dir}`",
        "",
        "## Summary",
        f"- Scores: `{metadata.get('n_scores', len(scores))}`",
        f"- Cell lines: `{metadata.get('n_cell_types', scores['cell_type'].nunique() if 'cell_type' in scores else 'unknown')}`",
        f"- Drugs: `{metadata.get('n_drugs', scores['drug_name'].nunique() if 'drug_name' in scores else 'unknown')}`",
        f"- Calibration modes: `{', '.join(modes)}`",
        f"- Scoring projection: `{metadata.get('projection_method', 'none')}`",
        f"- Projection components / split: `{metadata.get('projection_components', 'n/a')}` / `{metadata.get('projection_target_split', 'n/a')}`",
        f"- Resolved DMSO adapter label: `{metadata.get('resolved_dmso_adapter_label', 'not used')}`",
        f"- Mean / median OT: `{global_summary.get('mean_sinkhorn_ot', 'unknown')}` / `{global_summary.get('median_sinkhorn_ot', 'unknown')}`",
        f"- 5% / 95% OT: `{global_summary.get('p05_sinkhorn_ot', 'unknown')}` / `{global_summary.get('p95_sinkhorn_ot', 'unknown')}`",
        "",
        "## Mode Summary",
        *mode_lines,
        "",
        "## Figures",
    ]
    for key, path in figure_paths.items():
        lines.append(f"- `{key}`: `{path}`")
    high_drug_summary = (
        drug_summary.sort_values("mean_sinkhorn_ot", ascending=False)
        if "mean_sinkhorn_ot" in drug_summary.columns
        else drug_summary
    )
    low_drug_summary = (
        drug_summary.sort_values("mean_sinkhorn_ot", ascending=True)
        if "mean_sinkhorn_ot" in drug_summary.columns
        else drug_summary
    )
    lines.extend(
        [
            "",
            "## Ranked Cell Lines",
            markdown_table(cell_summary, max_rows=16),
            "",
            "## Highest OT Drugs",
            markdown_table(high_drug_summary, max_rows=10),
            "",
            "## Lowest OT Drugs",
            markdown_table(low_drug_summary, max_rows=10),
            "",
            "## Report Tables",
        ]
    )
    for key, path in table_paths.items():
        lines.append(f"- `{key}`: `{path}`")

    path = report_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_target_calibration_qc_report(
    *,
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    top_n_drugs: int = 10,
    rank_stat: str = "mean",
) -> Dict[str, str]:
    ensure_report_dependencies()
    run_dir = Path(run_dir)
    report_dir = Path(output_dir) if output_dir else run_dir / "report"
    fig_dir = safe_mkdir(report_dir / "figures")
    safe_mkdir(report_dir / "tables")
    logger = setup_logger(report_dir)
    logger.info("Generating target calibration QC report for %s", run_dir)

    tables = load_tables(run_dir)
    manifest = tables["manifest"]
    scores = tables["scores"]
    cell_summary = tables["cell_line_summary"]
    drug_summary = tables["drug_summary"]
    if scores.empty:
        raise ValueError("target_calibration_scores table is empty; no report can be generated.")
    scores = add_conversion_percent_columns(scores)
    cell_summary = ensure_calibration_mode_column(cell_summary)
    drug_summary = ensure_calibration_mode_column(drug_summary)

    figure_paths: Dict[str, str] = {}
    for mode in calibration_modes_in_scores(scores):
        mode_scores = scores[scores["calibration_mode"].astype(str) == mode].copy()
        mode_fig_dir = safe_mkdir(fig_dir / mode)
        mode_label = calibration_mode_label(mode)
        q05, q95 = global_quantiles(mode_scores)
        logger.info("Generating %s figures for %d scores.", mode, len(mode_scores))
        kde_path = plot_kde_distribution(mode_scores, mode_fig_dir, mode_label=mode_label)
        percent_kde_path = plot_conversion_percent_kde(mode_scores, mode_fig_dir, mode_label=mode_label)
        cell_path = plot_cell_line_violin(mode_scores, mode_fig_dir, rank_stat=rank_stat, q05=q05, q95=q95, mode_label=mode_label)
        high_path, low_path = plot_top_bottom_drug_violins(
            mode_scores,
            mode_fig_dir,
            rank_stat=rank_stat,
            q05=q05,
            q95=q95,
            top_n=top_n_drugs,
            mode_label=mode_label,
        )
        figure_paths.update(
            {
                f"{mode}_sinkhorn_ot_distribution_kde": str(kde_path),
                f"{mode}_percent_total_ot_closed_kde": str(percent_kde_path),
                f"{mode}_cell_line_ranked_violin": str(cell_path),
                f"{mode}_top_high_ot_drugs_violin": str(high_path),
                f"{mode}_bottom_low_ot_drugs_violin": str(low_path),
            }
        )
    table_paths = write_report_tables(report_dir, scores, cell_summary, drug_summary, rank_stat, top_n_drugs)
    summary = write_summary(report_dir, run_dir, manifest, scores, cell_summary, drug_summary, figure_paths, table_paths)
    logger.info("Report complete: %s", summary)

    out = {"report_dir": str(report_dir), "summary": str(summary)}
    out.update(figure_paths)
    out.update(table_paths)
    return out


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate report figures for target calibration QC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory produced by target_calibration_qc_analysis.py.")
    p.add_argument("--output-dir", default=None, help="Report output directory. Default: <run-dir>/report.")
    p.add_argument("--top-n-drugs", type=int, default=10, help="Number of high/low ranked drugs shown in report violins.")
    p.add_argument("--rank-stat", choices=["mean", "median"], default="mean", help="Statistic used to rank cell lines and drugs.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = make_target_calibration_qc_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        top_n_drugs=args.top_n_drugs,
        rank_stat=args.rank_stat,
    )
    print("\n=== Target calibration QC report complete ===")
    print(f"summary: {out['summary']}")
    print(f"figures: {Path(out['report_dir']) / 'figures'}")
    print(f"tables:  {Path(out['report_dir']) / 'tables'}")


if __name__ == "__main__":
    main()
