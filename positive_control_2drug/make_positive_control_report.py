#!/usr/bin/env python
"""
make_positive_control_report.py

Generate figures and a concise report for positive_control_2drug.py outputs.

Expected input directory:
    run_dir/
        positive_control_config.used.json
        tables/
            baseline_results.tsv
            selected_pairs.tsv
            evaluation_results.tsv
            explicit_pair_additive_results.tsv

Outputs:
    run_dir/report/  (or --output-dir)
        summary.md
        tables/
            plotted_bar_values.tsv
            top_moa_pairs.tsv
            explicit_pair_additive_summary.tsv
            explicit_pair_interaction_scores.tsv
            explicit_pair_interaction_scores_by_batch.tsv
        figures/
            01_positive_control_barplot.png
            02_positive_control_boxplot.png
            03_positive_control_kde.png
            04_explicit_pair_additive_boxplot.png
            05_explicit_pair_interaction_gains.png
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


GROUP_COLORS = {
    "random_pair": "#9aa0a6",
    "explicit_pair": "#2f6fed",
    "moa_pair": "#2ca25f",
    "baseline": "#222222",
    "baseline_stse_control_1pass": "#d95f0e",
    "baseline_stse_control_2pass": "#b30000",
}

CONTROL_BASELINE_GROUPS = (
    ("baseline_stse_control_1pass", "ST-SE control x1"),
    ("baseline_stse_control_2pass", "ST-SE control x2"),
)

ADDITIVE_MODE_COLORS = {
    "single_A": "#6baed6",
    "single_B": "#9ecae1",
    "additive_A_plus_B": "#31a354",
    "sequential_A_to_B": "#2f6fed",
    "sequential_B_to_A": "#756bb1",
}

INTERACTION_COLORS = {
    "additive_gain": "#31a354",
    "sequential_gain_A_to_B": "#2f6fed",
    "interaction_gain": "#f28e2b",
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
    """Import plotting/data dependencies lazily so --help works in a bare shell."""
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


def make_legend_opaque(ax: Any, **kwargs: Any) -> None:
    legend = ax.legend(frameon=False, **kwargs)
    if legend is None:
        return
    handles = getattr(legend, "legend_handles", None) or getattr(legend, "legendHandles", [])
    for handle in handles:
        if hasattr(handle, "set_alpha"):
            handle.set_alpha(1.0)


def load_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "positive_control_config.used.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_tables(run_dir: Path) -> Dict[str, pd.DataFrame]:
    tables_dir = run_dir / "tables"
    required = {
        "baseline": tables_dir / "baseline_results.tsv",
        "selected": tables_dir / "selected_pairs.tsv",
        "evaluation": tables_dir / "evaluation_results.tsv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required positive-control tables: {missing}")

    return {
        "baseline": pd.read_csv(required["baseline"], sep="\t"),
        "selected": pd.read_csv(required["selected"], sep="\t"),
        "evaluation": pd.read_csv(required["evaluation"], sep="\t"),
        "explicit_additive": pd.read_csv(tables_dir / "explicit_pair_additive_results.tsv", sep="\t")
        if (tables_dir / "explicit_pair_additive_results.tsv").exists()
        else pd.DataFrame(),
    }


def finite_values(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def error_value(values: np.ndarray, mode: str) -> float:
    values = finite_values(values)
    if len(values) <= 1:
        return 0.0
    if mode == "std":
        return float(np.std(values, ddof=1))
    if mode == "sem":
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))
    raise ValueError("errorbar must be 'std' or 'sem'")


def wrap_label(label: str, width: int = 22) -> str:
    label = str(label)
    label = label.replace(" + ", "\n+\n")
    parts = []
    for line in label.splitlines():
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        parts.extend(wrapped or [line])
    return "\n".join(parts)


def selected_pair_order(selected: pd.DataFrame, evaluation: pd.DataFrame) -> pd.DataFrame:
    selected = selected.copy()
    if "eval_mean_sinkhorn_ot" not in selected.columns or selected["eval_mean_sinkhorn_ot"].isna().all():
        stats = (
            evaluation[evaluation["group"].isin(["explicit_pair", "moa_pair"])]
            .groupby(["group", "pair_id"], as_index=False)["score_sinkhorn_ot"]
            .mean()
            .rename(columns={"score_sinkhorn_ot": "eval_mean_sinkhorn_ot"})
        )
        selected = selected.drop(columns=["eval_mean_sinkhorn_ot"], errors="ignore").merge(
            stats, on=["group", "pair_id"], how="left"
        )
    return selected


def build_plot_groups(
    evaluation: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    top_n_moa: int = 5,
    errorbar: str = "std",
) -> tuple[List[Dict[str, Any]], pd.DataFrame]:
    evaluation = evaluation.copy()
    selected = selected_pair_order(selected, evaluation)

    groups: List[Dict[str, Any]] = []

    random_values = finite_values(evaluation.loc[evaluation["group"] == "random_pair", "score_sinkhorn_ot"])
    if len(random_values):
        groups.append(
            {
                "plot_label": f"Random pairs\nn={len(random_values)}",
                "group": "random_pair",
                "pair_id": "random_pairs",
                "values": random_values,
                "mean": float(np.mean(random_values)),
                "error": error_value(random_values, errorbar),
                "n": int(len(random_values)),
                "color": GROUP_COLORS["random_pair"],
            }
        )

    explicit_rows = selected[selected["group"] == "explicit_pair"].copy()
    if not explicit_rows.empty:
        explicit_pair_id = str(explicit_rows.iloc[0]["pair_id"])
        explicit_values = finite_values(
            evaluation.loc[
                (evaluation["group"] == "explicit_pair") & (evaluation["pair_id"] == explicit_pair_id),
                "score_sinkhorn_ot",
            ]
        )
        groups.append(
            {
                "plot_label": wrap_label(explicit_pair_id),
                "group": "explicit_pair",
                "pair_id": explicit_pair_id,
                "values": explicit_values,
                "mean": float(np.mean(explicit_values)) if len(explicit_values) else np.nan,
                "error": error_value(explicit_values, errorbar),
                "n": int(len(explicit_values)),
                "color": GROUP_COLORS["explicit_pair"],
            }
        )

    moa_selected = selected[selected["group"] == "moa_pair"].copy()
    moa_selected = moa_selected.sort_values(["eval_mean_sinkhorn_ot", "pair_id"], ascending=[True, True])
    top_moa = moa_selected.head(int(top_n_moa)).copy()
    for _, row in top_moa.iterrows():
        pair_id = str(row["pair_id"])
        values = finite_values(
            evaluation.loc[
                (evaluation["group"] == "moa_pair") & (evaluation["pair_id"] == pair_id),
                "score_sinkhorn_ot",
            ]
        )
        groups.append(
            {
                "plot_label": wrap_label(pair_id),
                "group": "moa_pair",
                "pair_id": pair_id,
                "values": values,
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "error": error_value(values, errorbar),
                "n": int(len(values)),
                "color": GROUP_COLORS["moa_pair"],
            }
        )

    return groups, top_moa


def baseline_line_value(baseline: pd.DataFrame, evaluation: pd.DataFrame) -> float:
    if "score_sinkhorn_ot" in baseline.columns and len(baseline):
        values = finite_values(baseline["score_sinkhorn_ot"])
        if len(values):
            return float(np.mean(values))
    values = finite_values(evaluation.loc[evaluation["group"] == "baseline", "score_sinkhorn_ot"])
    return float(np.mean(values)) if len(values) else np.nan


def extra_baseline_lines(evaluation: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Reference lines for the ST-SE 'null perturbation' baselines (control x1 / x2),
    if present in the evaluation table. Returned as draw-ready dicts.
    """
    lines: List[Dict[str, Any]] = []
    if "group" not in evaluation.columns:
        return lines
    for group, label in CONTROL_BASELINE_GROUPS:
        values = finite_values(evaluation.loc[evaluation["group"] == group, "score_sinkhorn_ot"])
        if len(values):
            value = float(np.mean(values))
            lines.append(
                {
                    "group": group,
                    "label": f"{label} ({value:.4g})",
                    "value": value,
                    "color": GROUP_COLORS.get(group, "#d95f0e"),
                }
            )
    return lines


def set_zoomed_ylim(
    ax,
    groups: Sequence[Dict[str, Any]],
    baseline_value: float,
    extra_baselines: Sequence[Dict[str, Any]] = (),
) -> None:
    values = []
    if np.isfinite(baseline_value):
        values.append(float(baseline_value))
    for line in extra_baselines:
        if np.isfinite(line.get("value", np.nan)):
            values.append(float(line["value"]))
    for g in groups:
        if np.isfinite(g.get("mean", np.nan)):
            values.append(float(g["mean"]))
        arr = finite_values(g.get("values", []))
        if len(arr):
            values.extend([float(np.min(arr)), float(np.max(arr))])
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    pad = max((y_max - y_min) * 0.12, abs(y_max) * 0.02, 1e-4)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)


def _draw_extra_baselines(ax, extra_baselines: Sequence[Dict[str, Any]], *, vertical: bool = False) -> None:
    for line in extra_baselines:
        if not np.isfinite(line.get("value", np.nan)):
            continue
        drawer = ax.axvline if vertical else ax.axhline
        drawer(
            float(line["value"]),
            linestyle="--",
            color=line["color"],
            linewidth=1.4,
            label=line["label"],
        )


def plot_bar(
    groups: Sequence[Dict[str, Any]],
    baseline_value: float,
    fig_dir: Path,
    extra_baselines: Sequence[Dict[str, Any]] = (),
) -> None:
    path = fig_dir / "01_positive_control_barplot.png"
    plt.figure(figsize=(max(6.8, 0.95 * len(groups) + 1.6), 4.9))
    ax = plt.gca()

    x = np.arange(len(groups))
    means = [g["mean"] for g in groups]
    errors = [g["error"] for g in groups]
    colors = [g["color"] for g in groups]
    labels = [g["plot_label"] for g in groups]

    ax.bar(x, means, yerr=errors, capsize=4, color=colors, edgecolor="#333333", linewidth=0.8)
    if np.isfinite(baseline_value):
        ax.axhline(
            baseline_value,
            linestyle=":",
            color=GROUP_COLORS["baseline"],
            linewidth=1.7,
            label=f"Baseline start to target ({baseline_value:.4g})",
        )
    #_draw_extra_baselines(ax, extra_baselines)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Sinkhorn OT to target (lower is better)")
    ax.set_title("Positive-control 2-drug distance to target")
    make_legend_opaque(ax, loc="best")
    set_zoomed_ylim(ax, groups, baseline_value, extra_baselines)
    polish_axes(ax)
    savefig(path)


def plot_box(
    groups: Sequence[Dict[str, Any]],
    baseline_value: float,
    fig_dir: Path,
    extra_baselines: Sequence[Dict[str, Any]] = (),
) -> None:
    path = fig_dir / "02_positive_control_boxplot.png"
    plt.figure(figsize=(max(6.8, 0.95 * len(groups) + 1.6), 4.9))
    ax = plt.gca()

    data = [finite_values(g["values"]) for g in groups]
    labels = [g["plot_label"] for g in groups]
    colors = [g["color"] for g in groups]

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#333333")

    if np.isfinite(baseline_value):
        ax.axhline(
            baseline_value,
            linestyle=":",
            color=GROUP_COLORS["baseline"],
            linewidth=1.7,
            label=f"Baseline start to target ({baseline_value:.4g})",
        )
    #_draw_extra_baselines(ax, extra_baselines)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Sinkhorn OT to target (lower is better)")
    ax.set_title("Positive-control 2-drug distance distributions")
    make_legend_opaque(ax, loc="best")
    set_zoomed_ylim(ax, groups, baseline_value, extra_baselines)
    polish_axes(ax)
    savefig(path)


def kde_line(ax, values: Sequence[float], *, color: str, label: str) -> None:
    values = finite_values(values)
    if len(values) == 0:
        return
    if len(values) < 2 or float(np.std(values)) == 0.0:
        ax.axvline(float(np.mean(values)), color=color, linewidth=2, label=label)
        return
    try:
        from scipy.stats import gaussian_kde

        x_min = float(np.min(values))
        x_max = float(np.max(values))
        if x_min == x_max:
            ax.axvline(x_min, color=color, linewidth=2, label=label)
            return
        xs = np.linspace(x_min, x_max, 300)  # cut=0 behavior
        ys = gaussian_kde(values)(xs)
        ax.plot(xs, ys, color=color, linewidth=2.8, label=label)
    except Exception:
        hist, edges = np.histogram(values, bins=min(30, max(5, len(values) // 5)), density=True)
        xs = (edges[:-1] + edges[1:]) / 2
        ax.plot(xs, hist, color=color, linewidth=2.8, label=label)


def plot_kde(
    evaluation: pd.DataFrame,
    baseline_value: float,
    fig_dir: Path,
    extra_baselines: Sequence[Dict[str, Any]] = (),
) -> None:
    path = fig_dir / "03_positive_control_kde.png"
    plt.figure(figsize=(6.4, 4.7))
    ax = plt.gca()

    random_values = evaluation.loc[evaluation["group"] == "random_pair", "score_sinkhorn_ot"]
    explicit_values = evaluation.loc[evaluation["group"] == "explicit_pair", "score_sinkhorn_ot"]
    moa_values = evaluation.loc[evaluation["group"] == "moa_pair", "score_sinkhorn_ot"]

    kde_line(ax, random_values, color=GROUP_COLORS["random_pair"], label="Random pairs")
    kde_line(ax, explicit_values, color=GROUP_COLORS["explicit_pair"], label="2-drug pair")
    kde_line(ax, moa_values, color=GROUP_COLORS["moa_pair"], label="MOA pairs")

    if np.isfinite(baseline_value):
        ax.axvline(baseline_value, linestyle=":", color=GROUP_COLORS["baseline"], linewidth=1.7, label="Baseline")
    #_draw_extra_baselines(ax, extra_baselines, vertical=True)
    ax.set_xlabel("Sinkhorn OT to target (lower is better)")
    ax.set_ylabel("Density")
    ax.set_title("Distance distributions")
    make_legend_opaque(ax, loc="best")
    polish_axes(ax)
    savefig(path)


def additive_summary_table(additive_df: pd.DataFrame, errorbar: str) -> pd.DataFrame:
    if additive_df.empty:
        return pd.DataFrame()
    d = additive_df.copy()
    summary = (
        d.groupby(["pair_id", "source_pair_selection", "mode_order", "mode", "mode_label"], as_index=False)
        .agg(
            n_batches=("score_sinkhorn_ot", "size"),
            mean_sinkhorn_ot=("score_sinkhorn_ot", "mean"),
            std_sinkhorn_ot=("score_sinkhorn_ot", "std"),
            mean_gain_sinkhorn_vs_baseline=("gain_sinkhorn_vs_baseline", "mean"),
            std_gain_sinkhorn_vs_baseline=("gain_sinkhorn_vs_baseline", "std"),
            mean_energy_distance=("score_energy_distance", "mean"),
            std_energy_distance=("score_energy_distance", "std"),
        )
        .sort_values("mode_order")
    )
    summary["std_sinkhorn_ot"] = summary["std_sinkhorn_ot"].fillna(0.0)
    summary["std_gain_sinkhorn_vs_baseline"] = summary["std_gain_sinkhorn_vs_baseline"].fillna(0.0)
    summary["std_energy_distance"] = summary["std_energy_distance"].fillna(0.0)
    if errorbar == "sem":
        summary["sem_sinkhorn_ot"] = summary["std_sinkhorn_ot"] / np.sqrt(summary["n_batches"].clip(lower=1))
        summary["sem_gain_sinkhorn_vs_baseline"] = summary["std_gain_sinkhorn_vs_baseline"] / np.sqrt(
            summary["n_batches"].clip(lower=1)
        )
        summary["sem_energy_distance"] = summary["std_energy_distance"] / np.sqrt(summary["n_batches"].clip(lower=1))
    return summary


def interaction_scores_table(additive_df: pd.DataFrame, errorbar: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if additive_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = additive_df.pivot_table(
        index="batch_index",
        columns="mode",
        values="gain_sinkhorn_vs_baseline",
        aggfunc="first",
    )
    required = {"additive_A_plus_B", "sequential_A_to_B"}
    if not required.issubset(set(pivot.columns)):
        return pd.DataFrame(), pd.DataFrame()

    per_batch = pd.DataFrame(
        {
            "batch_index": pivot.index.astype(int),
            "additive_gain": pivot["additive_A_plus_B"].values,
            "sequential_gain_A_to_B": pivot["sequential_A_to_B"].values,
        }
    )
    per_batch["interaction_gain"] = per_batch["sequential_gain_A_to_B"] - per_batch["additive_gain"]

    labels = {
        "additive_gain": "Additive gain",
        "sequential_gain_A_to_B": "Sequential gain A->B",
        "interaction_gain": "Interaction gain",
    }
    order = {"additive_gain": 1, "sequential_gain_A_to_B": 2, "interaction_gain": 3}

    long_rows = []
    for _, row in per_batch.iterrows():
        for score_type in ["additive_gain", "sequential_gain_A_to_B", "interaction_gain"]:
            long_rows.append(
                {
                    "batch_index": int(row["batch_index"]),
                    "score_type": score_type,
                    "score_label": labels[score_type],
                    "score_order": order[score_type],
                    "gain_sinkhorn_ot": float(row[score_type]),
                }
            )
    long_df = pd.DataFrame(long_rows)
    summary = (
        long_df.groupby(["score_order", "score_type", "score_label"], as_index=False)
        .agg(
            n_batches=("gain_sinkhorn_ot", "size"),
            mean_gain_sinkhorn_ot=("gain_sinkhorn_ot", "mean"),
            std_gain_sinkhorn_ot=("gain_sinkhorn_ot", "std"),
        )
        .sort_values("score_order")
    )
    summary["std_gain_sinkhorn_ot"] = summary["std_gain_sinkhorn_ot"].fillna(0.0)
    if errorbar == "sem":
        summary["sem_gain_sinkhorn_ot"] = summary["std_gain_sinkhorn_ot"] / np.sqrt(summary["n_batches"].clip(lower=1))
    return long_df, summary


def plot_explicit_additive_boxplot(
    additive_df: pd.DataFrame,
    baseline_value: float,
    fig_dir: Path,
    *,
    errorbar: str,
) -> None:
    path = fig_dir / "04_explicit_pair_additive_boxplot.png"
    plt.figure(figsize=(7.2, 4.9))
    ax = plt.gca()

    if additive_df.empty:
        ax.text(0.5, 0.5, "No explicit-pair additive results available", ha="center", va="center")
        ax.axis("off")
        savefig(path)
        return

    d = additive_df.sort_values(["mode_order", "batch_index"]).copy()
    modes = d[["mode_order", "mode", "mode_label"]].drop_duplicates().sort_values("mode_order")
    data = [finite_values(d.loc[d["mode"] == mode, "score_sinkhorn_ot"]) for mode in modes["mode"]]
    labels = [wrap_label(str(x), width=18) for x in modes["mode_label"]]
    colors = [ADDITIVE_MODE_COLORS.get(str(mode), "#bdbdbd") for mode in modes["mode"]]

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#333333")

    means = [float(np.mean(vals)) if len(vals) else np.nan for vals in data]
    errors = [error_value(vals, errorbar) for vals in data]
    x = np.arange(1, len(data) + 1)
    ax.errorbar(x, means, yerr=errors, fmt="o", color="#111111", capsize=4, markersize=5, label=f"mean +/- {errorbar}")

    if np.isfinite(baseline_value):
        ax.axhline(
            baseline_value,
            linestyle=":",
            color=GROUP_COLORS["baseline"],
            linewidth=1.7,
            label=f"Baseline start to target ({baseline_value:.4g})",
        )

    groups = [
        {"mean": mean, "values": vals}
        for mean, vals in zip(means, data)
    ]
    ax.set_ylabel("Sinkhorn OT to target (lower is better)")
    ax.set_title("Selected 2-drug pair: additive delta versus sequential order")
    make_legend_opaque(ax, loc="best")
    set_zoomed_ylim(ax, groups, baseline_value)
    polish_axes(ax)
    savefig(path)


def plot_interaction_gains(
    interaction_summary: pd.DataFrame,
    fig_dir: Path,
    *,
    errorbar: str,
) -> None:
    path = fig_dir / "05_explicit_pair_interaction_gains.png"
    plt.figure(figsize=(5.2, 4.6))
    ax = plt.gca()

    if interaction_summary.empty:
        ax.text(0.5, 0.5, "No interaction scores available", ha="center", va="center")
        ax.axis("off")
        savefig(path)
        return

    d = interaction_summary.sort_values("score_order").copy()
    err_col = "sem_gain_sinkhorn_ot" if errorbar == "sem" and "sem_gain_sinkhorn_ot" in d.columns else "std_gain_sinkhorn_ot"
    colors = [INTERACTION_COLORS.get(str(x), "#bdbdbd") for x in d["score_type"]]
    x = np.arange(len(d))
    ax.bar(
        x,
        d["mean_gain_sinkhorn_ot"],
        yerr=d[err_col],
        capsize=4,
        color=colors,
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(x, width=16) for x in d["score_label"]], rotation=20, ha="right")
    ax.set_ylabel("Gain versus baseline Sinkhorn OT")
    ax.set_title("Additive and interaction gains")
    polish_axes(ax)
    savefig(path)


def plotted_values_table(groups: Sequence[Dict[str, Any]], baseline_value: float, errorbar: str) -> pd.DataFrame:
    rows = []
    for i, g in enumerate(groups, start=1):
        rows.append(
            {
                "plot_order": i,
                "group": g["group"],
                "pair_id": g["pair_id"],
                "n": g["n"],
                "mean_sinkhorn_ot": g["mean"],
                f"{errorbar}_sinkhorn_ot": g["error"],
                "baseline_mean_sinkhorn_ot": baseline_value,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
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


def write_summary(
    report_dir: Path,
    run_dir: Path,
    config_payload: Dict[str, Any],
    baseline: pd.DataFrame,
    selected: pd.DataFrame,
    evaluation: pd.DataFrame,
    plotted: pd.DataFrame,
    top_moa: pd.DataFrame,
    additive_summary: pd.DataFrame,
    interaction_summary: pd.DataFrame,
    errorbar: str,
) -> None:
    cfg = config_payload.get("config", {}) if config_payload else {}
    metadata = config_payload.get("metadata", {}) if config_payload else {}

    baseline_value = baseline_line_value(baseline, evaluation)
    explicit = selected[selected["group"] == "explicit_pair"].copy()
    explicit_pair = explicit.iloc[0]["pair_id"] if not explicit.empty else "not found"
    additive_pair = additive_summary.iloc[0]["pair_id"] if not additive_summary.empty and "pair_id" in additive_summary else "not found"
    additive_source = (
        additive_summary.iloc[0]["source_pair_selection"]
        if not additive_summary.empty and "source_pair_selection" in additive_summary
        else "unknown"
    )
    explicit_mean = (
        float(explicit.iloc[0]["eval_mean_sinkhorn_ot"])
        if not explicit.empty and "eval_mean_sinkhorn_ot" in explicit.columns
        else np.nan
    )

    lines = [
        "# Positive-Control 2-Drug Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Report directory: `{report_dir}`",
        "",
        "## Executive summary",
        f"- Start cell/state: `{cfg.get('start_cell', 'unknown')}`",
        f"- Target cell/state: `{cfg.get('target_cell', 'unknown')}`",
        f"- Explicit ordered pair: `{explicit_pair}`",
        f"- Additive/sequential selected pair: `{additive_pair}` (`{additive_source}`)",
        f"- Batches for explicit/MOA evaluation: `{cfg.get('n_batches', 'unknown')}`",
        f"- Random pairs scored once: `{int((evaluation['group'] == 'random_pair').sum())}`",
        f"- Baseline mean Sinkhorn OT: `{baseline_value:.6g}`",
    ]
    if np.isfinite(explicit_mean):
        lines.append(f"- Explicit pair mean Sinkhorn OT: `{explicit_mean:.6g}`")
        lines.append(f"- Explicit pair mean delta from baseline: `{explicit_mean - baseline_value:.6g}`")
    if metadata:
        lines.append(f"- MOA selected pairs evaluated: `{metadata.get('n_moa_selected_pairs', 'unknown')}`")
    lines.append("")

    lines.append("## Plotted bar values")
    lines.append(markdown_table(plotted, max_rows=20))
    lines.append("")

    lines.append("## Top MOA pairs")
    show_cols = [
        "pair_id",
        "first_moa_fine",
        "second_moa_fine",
        "eval_mean_sinkhorn_ot",
        "eval_std_sinkhorn_ot",
        "selection_score_sinkhorn_ot",
        "first_perturbation",
        "second_perturbation",
    ]
    show_cols = [c for c in show_cols if c in top_moa.columns]
    lines.append(markdown_table(top_moa[show_cols] if show_cols else top_moa, max_rows=10))
    lines.append("")

    lines.append("## Explicit Pair Additive Analysis")
    show_cols = [
        "pair_id",
        "source_pair_selection",
        "mode",
        "mean_sinkhorn_ot",
        "std_sinkhorn_ot",
        "mean_gain_sinkhorn_vs_baseline",
        "std_gain_sinkhorn_vs_baseline",
    ]
    if errorbar == "sem":
        show_cols.extend(["sem_sinkhorn_ot", "sem_gain_sinkhorn_vs_baseline"])
    show_cols = [c for c in show_cols if c in additive_summary.columns]
    lines.append(markdown_table(additive_summary[show_cols] if show_cols else additive_summary, max_rows=10))
    lines.append("")

    lines.append("## Additive and Interaction Gains")
    show_cols = ["score_label", "mean_gain_sinkhorn_ot", "std_gain_sinkhorn_ot"]
    if errorbar == "sem":
        show_cols.append("sem_gain_sinkhorn_ot")
    show_cols = [c for c in show_cols if c in interaction_summary.columns]
    lines.append(markdown_table(interaction_summary[show_cols] if show_cols else interaction_summary, max_rows=10))
    lines.append("")

    lines.append("## Figures")
    lines.append("- `figures/01_positive_control_barplot.png`: mean Sinkhorn OT with error bars.")
    lines.append("- `figures/02_positive_control_boxplot.png`: same group layout as the bar plot, without outlier fliers.")
    lines.append("- `figures/03_positive_control_kde.png`: unfilled KDE lines for random, explicit-pair, and MOA-pair distributions.")
    lines.append("- `figures/04_explicit_pair_additive_boxplot.png`: single A, single B, additive A+B, A->B, and B->A distances.")
    lines.append("- `figures/05_explicit_pair_interaction_gains.png`: additive gain, sequential A->B gain, and interaction gain.")
    lines.append("")
    lines.append(f"Error bars use `{errorbar}`.")

    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def make_positive_control_report(
    *,
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    top_n_moa: int = 5,
    errorbar: str = "std",
) -> Dict[str, str]:
    ensure_report_dependencies()

    run_dir = Path(run_dir)
    report_dir = Path(output_dir) if output_dir else run_dir / "report"
    table_dir = safe_mkdir(report_dir / "tables")
    fig_dir = safe_mkdir(report_dir / "figures")

    tables = load_tables(run_dir)
    config_payload = load_config(run_dir)
    baseline = tables["baseline"]
    selected = tables["selected"]
    evaluation = tables["evaluation"]
    explicit_additive = tables["explicit_additive"]

    groups, top_moa = build_plot_groups(evaluation, selected, top_n_moa=top_n_moa, errorbar=errorbar)
    baseline_value = baseline_line_value(baseline, evaluation)
    extra_baselines = extra_baseline_lines(evaluation)

    if not groups:
        raise ValueError("No groups available to plot. Check evaluation_results.tsv.")

    plot_bar(groups, baseline_value, fig_dir, extra_baselines)
    plot_box(groups, baseline_value, fig_dir, extra_baselines)
    plot_kde(evaluation, baseline_value, fig_dir, extra_baselines)
    plot_explicit_additive_boxplot(explicit_additive, baseline_value, fig_dir, errorbar=errorbar)

    plotted = plotted_values_table(groups, baseline_value, errorbar)
    additive_summary = additive_summary_table(explicit_additive, errorbar)
    interaction_per_batch, interaction_summary = interaction_scores_table(explicit_additive, errorbar)
    plot_interaction_gains(interaction_summary, fig_dir, errorbar=errorbar)

    plotted_path = table_dir / "plotted_bar_values.tsv"
    top_moa_path = table_dir / "top_moa_pairs.tsv"
    additive_summary_path = table_dir / "explicit_pair_additive_summary.tsv"
    interaction_per_batch_path = table_dir / "explicit_pair_interaction_scores_by_batch.tsv"
    interaction_summary_path = table_dir / "explicit_pair_interaction_scores.tsv"
    plotted.to_csv(plotted_path, sep="\t", index=False)
    top_moa.to_csv(top_moa_path, sep="\t", index=False)
    additive_summary.to_csv(additive_summary_path, sep="\t", index=False)
    interaction_per_batch.to_csv(interaction_per_batch_path, sep="\t", index=False)
    interaction_summary.to_csv(interaction_summary_path, sep="\t", index=False)

    write_summary(
        report_dir,
        run_dir,
        config_payload,
        baseline,
        selected,
        evaluation,
        plotted,
        top_moa,
        additive_summary,
        interaction_summary,
        errorbar,
    )

    return {
        "report_dir": str(report_dir),
        "summary": str(report_dir / "summary.md"),
        "plotted_bar_values": str(plotted_path),
        "top_moa_pairs": str(top_moa_path),
        "explicit_pair_additive_summary": str(additive_summary_path),
        "explicit_pair_interaction_scores_by_batch": str(interaction_per_batch_path),
        "explicit_pair_interaction_scores": str(interaction_summary_path),
        "barplot": str(fig_dir / "01_positive_control_barplot.png"),
        "boxplot": str(fig_dir / "02_positive_control_boxplot.png"),
        "kde": str(fig_dir / "03_positive_control_kde.png"),
        "explicit_additive_boxplot": str(fig_dir / "04_explicit_pair_additive_boxplot.png"),
        "interaction_gains": str(fig_dir / "05_explicit_pair_interaction_gains.png"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the positive-control 2-drug report and figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory produced by positive_control_2drug_analysis.py.")
    p.add_argument("--output-dir", default=None, help="Report output directory. Default: <run-dir>/report.")
    p.add_argument("--top-n-moa", type=int, default=5, help="Number of top MOA pairs shown as individual bars/boxes.")
    p.add_argument("--errorbar", choices=["std", "sem"], default="std", help="Error bar summary for bar plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = make_positive_control_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        top_n_moa=args.top_n_moa,
        errorbar=args.errorbar,
    )
    print("\n=== Positive-control report complete ===")
    print(f"summary: {out['summary']}")
    print(f"figures: {Path(out['barplot']).parent}")
    print(f"tables:  {Path(out['plotted_bar_values']).parent}")


if __name__ == "__main__":
    main()
