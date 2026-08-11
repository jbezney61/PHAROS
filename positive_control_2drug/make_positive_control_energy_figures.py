#!/usr/bin/env python
"""
make_positive_control_energy_figures.py

One-off report add-on for positive-control 2-drug runs.

This script reads an existing positive_control_2drug_analysis.py output
directory and writes energy-distance versions of the first three report
figures. It does not rerun model inference and does not change the canonical
Sinkhorn report figures.

Inputs
------
run_dir/
    tables/
        baseline_results.tsv
        selected_pairs.tsv
        evaluation_results.tsv

Outputs
-------
run_dir/report/
    figures/
        01_positive_control_barplot_energy.png
        02_positive_control_boxplot_energy.png
        03_positive_control_kde_energy.png
    tables/
        plotted_energy_bar_values.tsv

Example
-------
python make_positive_control_energy_figures.py \
  --run-dir runs/PC_CPA_pano_alve
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


GROUP_COLORS = {
    "random_pair": "#9aa0a6",
    "explicit_pair": "#2f6fed",
    "moa_pair": "#2ca25f",
    "baseline": "#222222",
}

np = None
pd = None
plt = None


def ensure_dependencies() -> None:
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


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def finite_values(values: Sequence[float]):
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def error_value(values, mode: str) -> float:
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


def load_tables(run_dir: Path) -> Dict[str, Any]:
    tables_dir = run_dir / "tables"
    required = {
        "baseline": tables_dir / "baseline_results.tsv",
        "selected": tables_dir / "selected_pairs.tsv",
        "evaluation": tables_dir / "evaluation_results.tsv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required positive-control tables: {missing}")

    baseline = pd.read_csv(required["baseline"], sep="\t")
    selected = pd.read_csv(required["selected"], sep="\t")
    evaluation = pd.read_csv(required["evaluation"], sep="\t")
    top_moa_path = run_dir / "report" / "tables" / "top_moa_pairs.tsv"
    top_moa_existing = pd.read_csv(top_moa_path, sep="\t") if top_moa_path.exists() else pd.DataFrame()

    for name, df in [("baseline_results.tsv", baseline), ("evaluation_results.tsv", evaluation)]:
        if "score_energy_distance" not in df.columns:
            raise KeyError(f"{name} is missing required column: score_energy_distance")

    return {
        "baseline": baseline,
        "selected": selected,
        "evaluation": evaluation,
        "top_moa_existing": top_moa_existing,
    }


def selected_pair_order(selected, evaluation):
    selected = selected.copy()
    if selected.empty:
        return selected

    if "eval_mean_sinkhorn_ot" not in selected.columns or selected["eval_mean_sinkhorn_ot"].isna().all():
        sinkhorn_stats = (
            evaluation[evaluation["group"].isin(["explicit_pair", "moa_pair"])]
            .groupby(["group", "pair_id"], as_index=False)["score_sinkhorn_ot"]
            .mean()
            .rename(columns={"score_sinkhorn_ot": "eval_mean_sinkhorn_ot"})
        )
        selected = selected.drop(columns=["eval_mean_sinkhorn_ot"], errors="ignore").merge(
            sinkhorn_stats,
            on=["group", "pair_id"],
            how="left",
        )

    if "eval_mean_energy_distance" not in selected.columns or selected["eval_mean_energy_distance"].isna().all():
        energy_stats = (
            evaluation[evaluation["group"].isin(["explicit_pair", "moa_pair"])]
            .groupby(["group", "pair_id"], as_index=False)["score_energy_distance"]
            .mean()
            .rename(columns={"score_energy_distance": "eval_mean_energy_distance"})
        )
        selected = selected.drop(columns=["eval_mean_energy_distance"], errors="ignore").merge(
            energy_stats,
            on=["group", "pair_id"],
            how="left",
        )
    return selected


def build_energy_plot_groups(
    evaluation,
    selected,
    *,
    top_moa_existing,
    top_n_moa: int,
    errorbar: str,
) -> tuple[List[Dict[str, Any]], Any]:
    selected = selected_pair_order(selected, evaluation)
    groups: List[Dict[str, Any]] = []

    random_values = finite_values(evaluation.loc[evaluation["group"] == "random_pair", "score_energy_distance"])
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
                "score_energy_distance",
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

    if top_moa_existing is not None and not top_moa_existing.empty and "pair_id" in top_moa_existing.columns:
        # Prefer the canonical report's Sinkhorn-selected top MOA pairs so the
        # energy figures isolate the metric change instead of changing labels.
        pair_order = top_moa_existing["pair_id"].astype(str).head(int(top_n_moa)).tolist()
        top_moa = selected[
            (selected["group"] == "moa_pair") & (selected["pair_id"].astype(str).isin(pair_order))
        ].copy()
        order_map = {pair_id: i for i, pair_id in enumerate(pair_order)}
        top_moa["_plot_order"] = top_moa["pair_id"].astype(str).map(order_map)
        top_moa = top_moa.sort_values("_plot_order").drop(columns=["_plot_order"], errors="ignore")
    else:
        moa_selected = selected[selected["group"] == "moa_pair"].copy()
        if not moa_selected.empty:
            moa_selected = moa_selected.sort_values(["eval_mean_sinkhorn_ot", "pair_id"], ascending=[True, True])
        top_moa = moa_selected.head(int(top_n_moa)).copy()

    for _, row in top_moa.iterrows():
        pair_id = str(row["pair_id"])
        values = finite_values(
            evaluation.loc[
                (evaluation["group"] == "moa_pair") & (evaluation["pair_id"] == pair_id),
                "score_energy_distance",
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


def baseline_energy_value(baseline, evaluation) -> float:
    if "score_energy_distance" in baseline.columns and len(baseline):
        values = finite_values(baseline["score_energy_distance"])
        if len(values):
            return float(np.mean(values))
    values = finite_values(evaluation.loc[evaluation["group"] == "baseline", "score_energy_distance"])
    return float(np.mean(values)) if len(values) else np.nan


def set_zoomed_ylim(ax, groups: Sequence[Dict[str, Any]], baseline_value: float) -> None:
    values = []
    if np.isfinite(baseline_value):
        values.append(float(baseline_value))
    for group in groups:
        if np.isfinite(group.get("mean", np.nan)):
            values.append(float(group["mean"]))
        arr = finite_values(group.get("values", []))
        if len(arr):
            values.extend([float(np.min(arr)), float(np.max(arr))])
    if not values:
        return
    y_min = min(values)
    y_max = max(values)
    pad = max((y_max - y_min) * 0.12, abs(y_max) * 0.02, 1e-4)
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)


def plot_bar_energy(groups: Sequence[Dict[str, Any]], baseline_value: float, fig_dir: Path) -> Path:
    path = fig_dir / "01_positive_control_barplot_energy.png"
    plt.figure(figsize=(max(8, 1.25 * len(groups) + 2), 5.8))
    ax = plt.gca()

    x = np.arange(len(groups))
    means = [group["mean"] for group in groups]
    errors = [group["error"] for group in groups]
    colors = [group["color"] for group in groups]
    labels = [group["plot_label"] for group in groups]

    ax.bar(x, means, yerr=errors, capsize=4, color=colors, edgecolor="#333333", linewidth=0.6)
    if np.isfinite(baseline_value):
        ax.axhline(
            baseline_value,
            linestyle=":",
            color=GROUP_COLORS["baseline"],
            linewidth=1.4,
            label=f"Baseline start to target ({baseline_value:.4g})",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Energy distance to target")
    ax.set_title("Positive-control 2-drug energy distance to target")
    ax.legend(frameon=False, loc="best")
    set_zoomed_ylim(ax, groups, baseline_value)
    savefig(path)
    return path


def plot_box_energy(groups: Sequence[Dict[str, Any]], baseline_value: float, fig_dir: Path) -> Path:
    path = fig_dir / "02_positive_control_boxplot_energy.png"
    plt.figure(figsize=(max(8, 1.25 * len(groups) + 2), 5.8))
    ax = plt.gca()

    data = [finite_values(group["values"]) for group in groups]
    labels = [group["plot_label"] for group in groups]
    colors = [group["color"] for group in groups]

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, tick_labels=labels)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#333333")

    if np.isfinite(baseline_value):
        ax.axhline(
            baseline_value,
            linestyle=":",
            color=GROUP_COLORS["baseline"],
            linewidth=1.4,
            label=f"Baseline start to target ({baseline_value:.4g})",
        )
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Energy distance to target")
    ax.set_title("Positive-control 2-drug energy distance distributions")
    ax.legend(frameon=False, loc="best")
    set_zoomed_ylim(ax, groups, baseline_value)
    savefig(path)
    return path


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
        ax.plot(xs, ys, color=color, linewidth=2, label=label)
    except Exception:
        hist, edges = np.histogram(values, bins=min(30, max(5, len(values) // 5)), density=True)
        xs = (edges[:-1] + edges[1:]) / 2
        ax.plot(xs, hist, color=color, linewidth=2, label=label)


def plot_kde_energy(evaluation, baseline_value: float, fig_dir: Path) -> Path:
    path = fig_dir / "03_positive_control_kde_energy.png"
    plt.figure(figsize=(8, 5.2))
    ax = plt.gca()

    random_values = evaluation.loc[evaluation["group"] == "random_pair", "score_energy_distance"]
    explicit_values = evaluation.loc[evaluation["group"] == "explicit_pair", "score_energy_distance"]
    moa_values = evaluation.loc[evaluation["group"] == "moa_pair", "score_energy_distance"]

    kde_line(ax, random_values, color=GROUP_COLORS["random_pair"], label="Random pairs")
    kde_line(ax, explicit_values, color=GROUP_COLORS["explicit_pair"], label="2-drug pair")
    kde_line(ax, moa_values, color=GROUP_COLORS["moa_pair"], label="MOA pairs")

    if np.isfinite(baseline_value):
        ax.axvline(baseline_value, linestyle=":", color=GROUP_COLORS["baseline"], linewidth=1.4, label="Baseline")
    ax.set_xlabel("Energy distance to target")
    ax.set_ylabel("Density")
    ax.set_title("Energy-distance distributions")
    ax.legend(frameon=False)
    savefig(path)
    return path


def plotted_energy_values_table(groups: Sequence[Dict[str, Any]], baseline_value: float, errorbar: str):
    rows = []
    for i, group in enumerate(groups, start=1):
        rows.append(
            {
                "plot_order": i,
                "group": group["group"],
                "pair_id": group["pair_id"],
                "n": group["n"],
                "mean_energy_distance": group["mean"],
                f"{errorbar}_energy_distance": group["error"],
                "baseline_mean_energy_distance": baseline_value,
            }
        )
    return pd.DataFrame(rows)


def make_positive_control_energy_figures(
    *,
    run_dir: str | Path,
    output_dir: Optional[str | Path] = None,
    top_n_moa: int = 5,
    errorbar: str = "std",
) -> Dict[str, str]:
    ensure_dependencies()

    run_dir = Path(run_dir)
    report_dir = Path(output_dir) if output_dir else run_dir / "report"
    fig_dir = safe_mkdir(report_dir / "figures")
    table_dir = safe_mkdir(report_dir / "tables")

    tables = load_tables(run_dir)
    baseline = tables["baseline"]
    selected = tables["selected"]
    evaluation = tables["evaluation"]
    top_moa_existing = tables["top_moa_existing"]

    groups, _ = build_energy_plot_groups(
        evaluation,
        selected,
        top_moa_existing=top_moa_existing,
        top_n_moa=top_n_moa,
        errorbar=errorbar,
    )
    if not groups:
        raise ValueError("No groups available to plot. Check evaluation_results.tsv.")

    baseline_value = baseline_energy_value(baseline, evaluation)
    barplot = plot_bar_energy(groups, baseline_value, fig_dir)
    boxplot = plot_box_energy(groups, baseline_value, fig_dir)
    kde = plot_kde_energy(evaluation, baseline_value, fig_dir)

    plotted = plotted_energy_values_table(groups, baseline_value, errorbar)
    plotted_path = table_dir / "plotted_energy_bar_values.tsv"
    plotted.to_csv(plotted_path, sep="\t", index=False)

    return {
        "report_dir": str(report_dir),
        "plotted_energy_bar_values": str(plotted_path),
        "barplot_energy": str(barplot),
        "boxplot_energy": str(boxplot),
        "kde_energy": str(kde),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add energy-distance versions of positive-control report figures 01-03.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory produced by positive_control_2drug_analysis.py.")
    p.add_argument("--output-dir", default=None, help="Report output directory. Default: <run-dir>/report.")
    p.add_argument("--top-n-moa", type=int, default=5, help="Number of top MOA pairs shown as individual bars/boxes.")
    p.add_argument("--errorbar", choices=["std", "sem"], default="std", help="Error bar summary for bar plot.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = make_positive_control_energy_figures(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        top_n_moa=args.top_n_moa,
        errorbar=args.errorbar,
    )
    print("\n=== Positive-control energy figures complete ===")
    print(f"report:  {out['report_dir']}")
    print(f"barplot: {out['barplot_energy']}")
    print(f"boxplot: {out['boxplot_energy']}")
    print(f"kde:     {out['kde_energy']}")
    print(f"table:   {out['plotted_energy_bar_values']}")


if __name__ == "__main__":
    main()
