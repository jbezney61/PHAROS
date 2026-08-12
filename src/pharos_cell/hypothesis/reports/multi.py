#!/usr/bin/env python
"""
make_positive_control_multi_report.py

Collate several positive_control_2drug_analysis.py output directories into
publication-quality side-by-side comparison plots.

Expected input run directories:
    run_dir/
        positive_control_config.used.json
        tables/
            baseline_results.tsv
            selected_pairs.tsv
            evaluation_results.tsv
            explicit_pair_additive_results.tsv

Outputs:
    output_dir/
        summary.md
        tables/
            conversion_summary.tsv
            positive_control_distribution_tests.tsv
            positive_control_plot_values.tsv
            additive_plot_values.tsv
        figures/
            01_multi_positive_control_boxplot.png
            02_multi_positive_control_violinplot.png
            03_multi_explicit_pair_additive_boxplot.png

Example:
    python positive_control_2drug/make_positive_control_multi_report.py \\
      --run-dirs runs/PC_CPA_pano_alve runs/PC_CPA_trametinib \\
      --labels "Pano + Alves" "Trametinib + ..." \\
      --output-dir runs/positive_control_2drug_multi_report
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .pair import (
    ADDITIVE_MODE_COLORS,
    GROUP_COLORS,
    PUBLICATION_RC,
    baseline_line_value,
    ensure_report_dependencies as ensure_single_report_dependencies,
    finite_values,
    load_config,
    load_tables,
    polish_axes,
    selected_pair_order,
)


np = None
pd = None
plt = None


GROUP_ORDER = ("random_pair", "moa_pair", "explicit_pair")
GROUP_LABELS = {
    "random_pair": "Random pairs",
    "moa_pair": "MOA-shared pairs",
    "explicit_pair": "2-drug pair",
}
GROUP_PLOT_COLORS = {
    "random_pair": GROUP_COLORS["random_pair"],
    "moa_pair": GROUP_COLORS["moa_pair"],
    "explicit_pair": GROUP_COLORS["explicit_pair"],
}

ADDITIVE_MODE_ORDER = (
    "single_A",
    "single_B",
    "additive_A_plus_B",
    "sequential_A_to_B",
    "sequential_B_to_A",
)
ADDITIVE_SHORT_LABELS = {
    "single_A": "A",
    "single_B": "B",
    "additive_A_plus_B": "A+B",
    "sequential_A_to_B": "A->B",
    "sequential_B_to_A": "B->A",
}
ADDITIVE_LEGEND_LABELS = {
    "single_A": "Single A",
    "single_B": "Single B",
    "additive_A_plus_B": "Additive A+B",
    "sequential_A_to_B": "Sequential A then B",
    "sequential_B_to_A": "Sequential B then A",
}


def ensure_dependencies() -> None:
    """Import plotting/data dependencies lazily so --help works in a bare shell."""
    global np, pd, plt
    if np is not None and pd is not None and plt is not None:
        return
    ensure_single_report_dependencies()
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


def wrap_label(label: str, width: int = 18) -> str:
    label = str(label).replace(" + ", "\n+\n")
    parts: List[str] = []
    for line in label.splitlines():
        wrapped = textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False)
        parts.extend(wrapped or [line])
    return "\n".join(parts)


def short_label(label: str, max_len: int = 42) -> str:
    label = str(label)
    if len(label) <= max_len:
        return label
    return label[: max_len - 1].rstrip() + "..."


def savefig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=320)
    plt.close(fig)


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    pvals = np.asarray(pvals, dtype=float)
    qvals = np.full(len(pvals), np.nan, dtype=float)
    finite_mask = np.isfinite(pvals)
    finite = pvals[finite_mask]
    if len(finite) == 0:
        return qvals.tolist()
    order = np.argsort(finite)
    ranked = finite[order]
    ranked_q = np.empty(len(finite), dtype=float)
    prev = 1.0
    for i in range(len(finite) - 1, -1, -1):
        prev = min(prev, ranked[i] * len(finite) / (i + 1))
        ranked_q[order[i]] = prev
    qvals[finite_mask] = np.minimum(ranked_q, 1.0)
    return qvals.tolist()


def significance_label(q_value: float) -> str:
    if not np.isfinite(q_value):
        return "n/a"
    if q_value <= 1e-4:
        return "****"
    if q_value <= 1e-3:
        return "***"
    if q_value <= 1e-2:
        return "**"
    if q_value <= 5e-2:
        return "*"
    return "ns"


def mann_whitney_p_value(control: Sequence[float], treatment: Sequence[float]) -> float:
    """
    One-sided test for lower treatment OT than random control OT.

    Returns P(random values tend to be greater than treatment values).
    """
    control_arr = finite_values(control)
    treatment_arr = finite_values(treatment)
    if len(control_arr) == 0 or len(treatment_arr) == 0:
        return math.nan
    try:
        from scipy.stats import mannwhitneyu

        _, p_value = mannwhitneyu(control_arr, treatment_arr, alternative="greater")
        return float(p_value)
    except Exception:
        return permutation_mean_p_value(control_arr, treatment_arr)


def permutation_mean_p_value(control: Sequence[float], treatment: Sequence[float], n_iter: int = 10000) -> float:
    """Deterministic fallback: one-sided permutation test on mean OT difference."""
    control_arr = finite_values(control)
    treatment_arr = finite_values(treatment)
    if len(control_arr) == 0 or len(treatment_arr) == 0:
        return math.nan
    observed = float(np.mean(control_arr) - np.mean(treatment_arr))
    pooled = np.concatenate([control_arr, treatment_arr])
    n_control = len(control_arr)
    rng = np.random.default_rng(0)
    more_extreme = 1
    for _ in range(int(n_iter)):
        perm = rng.permutation(pooled)
        diff = float(np.mean(perm[:n_control]) - np.mean(perm[n_control:]))
        if diff >= observed:
            more_extreme += 1
    return float(more_extreme / (int(n_iter) + 1))


def percentile_lower_than_random(random_values: Sequence[float], explicit_values: Sequence[float]) -> float:
    random_arr = finite_values(random_values)
    explicit_arr = finite_values(explicit_values)
    if len(random_arr) == 0 or len(explicit_arr) == 0:
        return math.nan
    explicit_mean = float(np.mean(explicit_arr))
    return float(100.0 * np.mean(random_arr >= explicit_mean))


@dataclass
class RunData:
    run_dir: Path
    label: str
    conversion_label: str
    pair_id: str
    baseline_value: float
    baseline_df: Any
    selected_df: Any
    evaluation_df: Any
    additive_df: Any
    config_payload: Dict[str, Any]
    values: Dict[str, Any]
    percentile_vs_random: float


def derive_conversion_label(run_dir: Path, config_payload: Dict[str, Any], fallback: str) -> str:
    cfg = config_payload.get("config", {}) if config_payload else {}
    target = str(cfg.get("target_cell", "") or "").strip()
    start = str(cfg.get("start_cell", "") or "").strip()
    if target and start:
        return f"{start} -> {target}"
    if target:
        return target
    return fallback


def selected_pair_id(selected: Any, evaluation: Any) -> str:
    selected = selected_pair_order(selected, evaluation)
    explicit = selected[selected["group"] == "explicit_pair"].copy()
    if not explicit.empty:
        if "eval_mean_sinkhorn_ot" in explicit.columns:
            explicit = explicit.sort_values(["eval_mean_sinkhorn_ot", "pair_id"], ascending=[True, True])
        return str(explicit.iloc[0]["pair_id"])
    moa = selected[selected["group"] == "moa_pair"].copy()
    if not moa.empty:
        sort_col = "eval_mean_sinkhorn_ot" if "eval_mean_sinkhorn_ot" in moa.columns else "selection_score_sinkhorn_ot"
        return str(moa.sort_values([sort_col, "pair_id"], ascending=[True, True]).iloc[0]["pair_id"])
    return "selected pair unavailable"


def load_run(run_dir: Path, label: Optional[str]) -> RunData:
    tables = load_tables(run_dir)
    config_payload = load_config(run_dir)
    baseline = tables["baseline"]
    selected = tables["selected"]
    evaluation = tables["evaluation"]
    additive = tables["explicit_additive"]

    required_eval_cols = {"group", "pair_id", "score_sinkhorn_ot"}
    missing_eval = required_eval_cols - set(evaluation.columns)
    if missing_eval:
        raise ValueError(f"{run_dir}/tables/evaluation_results.tsv missing columns: {sorted(missing_eval)}")
    if "score_sinkhorn_ot" not in baseline.columns:
        raise ValueError(f"{run_dir}/tables/baseline_results.tsv missing column: score_sinkhorn_ot")

    values = {
        group: finite_values(evaluation.loc[evaluation["group"] == group, "score_sinkhorn_ot"])
        for group in GROUP_ORDER
    }
    pair_id = selected_pair_id(selected, evaluation)
    baseline_value = baseline_line_value(baseline, evaluation)
    fallback = run_dir.name
    conversion_label = derive_conversion_label(run_dir, config_payload, fallback=fallback)
    display_label = label if label is not None else fallback
    percentile = percentile_lower_than_random(values["random_pair"], values["explicit_pair"])

    return RunData(
        run_dir=run_dir,
        label=display_label,
        conversion_label=conversion_label,
        pair_id=pair_id,
        baseline_value=baseline_value,
        baseline_df=baseline,
        selected_df=selected,
        evaluation_df=evaluation,
        additive_df=additive,
        config_payload=config_payload,
        values=values,
        percentile_vs_random=percentile,
    )


def y_limits_for_values(value_groups: Sequence[Sequence[float]], baseline_value: float, extra_high: float = 0.22) -> Tuple[float, float, float]:
    all_values: List[float] = []
    if np.isfinite(baseline_value):
        all_values.append(float(baseline_value))
    for values in value_groups:
        arr = finite_values(values)
        if len(arr):
            all_values.extend([float(np.min(arr)), float(np.max(arr))])
    if not all_values:
        return 0.0, 1.0, 1.0
    y_min = min(all_values)
    y_max = max(all_values)
    span = max(y_max - y_min, abs(y_max) * 0.08, 1e-4)
    lower = max(0.0, y_min - 0.10 * span)
    upper = y_max + extra_high * span
    return lower, upper, span


def draw_significance_bracket(ax: Any, x1: float, x2: float, y: float, height: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#222222", linewidth=1.1, clip_on=False)
    ax.text((x1 + x2) / 2, y + height, label, ha="center", va="bottom", fontsize=10, color="#222222")


def draw_boxplot_at_positions(
    ax: Any,
    data: Sequence[Sequence[float]],
    positions: Sequence[float],
    colors: Sequence[str],
    *,
    width: float,
) -> None:
    bp = ax.boxplot(
        [finite_values(x) for x in data],
        positions=list(positions),
        widths=width,
        patch_artist=True,
        showfliers=False,
        manage_ticks=False,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.88)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(0.9)
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#333333")
            artist.set_linewidth(0.9)


def draw_violin_at_positions(
    ax: Any,
    data: Sequence[Sequence[float]],
    positions: Sequence[float],
    colors: Sequence[str],
    *,
    width: float,
) -> None:
    clean_data = [finite_values(x) for x in data]
    nonempty = [(i, arr) for i, arr in enumerate(clean_data) if len(arr)]
    if not nonempty:
        return
    use_positions = [positions[i] for i, _ in nonempty]
    use_data = [arr for _, arr in nonempty]
    parts = ax.violinplot(use_data, positions=use_positions, widths=width, showmeans=False, showmedians=True, showextrema=True)
    for body, (i, _) in zip(parts["bodies"], nonempty):
        body.set_facecolor(colors[i])
        body.set_edgecolor("#333333")
        body.set_alpha(0.82)
        body.set_linewidth(0.8)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        artist = parts.get(key)
        if artist is not None:
            artist.set_color("#333333")
            artist.set_linewidth(0.9)


def add_positive_control_legend(legend_ax: Any, *, show_baseline_line: bool = True) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=GROUP_PLOT_COLORS["random_pair"], edgecolor="#333333", label=GROUP_LABELS["random_pair"]),
        Patch(facecolor=GROUP_PLOT_COLORS["moa_pair"], edgecolor="#333333", label=GROUP_LABELS["moa_pair"]),
        Patch(facecolor=GROUP_PLOT_COLORS["explicit_pair"], edgecolor="#333333", label=GROUP_LABELS["explicit_pair"]),
    ]
    if show_baseline_line:
        handles.append(
            Line2D([0], [0], color=GROUP_COLORS["baseline"], linestyle=":", linewidth=1.8, label="Baseline start to target")
        )
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center left", frameon=False, borderaxespad=0.0)


def add_additive_legend(legend_ax: Any, *, show_baseline_line: bool = True) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=ADDITIVE_MODE_COLORS.get(mode, "#bdbdbd"), edgecolor="#333333", label=ADDITIVE_LEGEND_LABELS[mode])
        for mode in ADDITIVE_MODE_ORDER
    ]
    if show_baseline_line:
        handles.append(
            Line2D([0], [0], color=GROUP_COLORS["baseline"], linestyle=":", linewidth=1.8, label="Baseline start to target")
        )
    legend_ax.axis("off")
    legend_ax.legend(handles=handles, loc="center left", frameon=False, borderaxespad=0.0)


def make_axes_grid(n_runs: int, *, subplot_width: float, legend_width: float, height: float, title: str) -> Tuple[Any, List[Any], Any]:
    fig = plt.figure(figsize=(subplot_width * n_runs + legend_width, height))
    gs = fig.add_gridspec(
        1,
        n_runs + 1,
        width_ratios=[1.0] * n_runs + [legend_width / subplot_width],
        wspace=0.58,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_runs)]
    legend_ax = fig.add_subplot(gs[0, n_runs])
    if title:
        fig.suptitle(title, y=0.995, fontsize=16)
    return fig, axes, legend_ax


def plot_positive_control_distributions(
    runs: Sequence[RunData],
    tests: Any,
    fig_dir: Path,
    *,
    kind: str,
    subplot_width: float,
    legend_width: float,
    height: float,
    title: str,
    show_baseline_line: bool = True,
) -> Path:
    path = fig_dir / f"0{'1' if kind == 'box' else '2'}_multi_positive_control_{kind}plot.png"
    fig, axes, legend_ax = make_axes_grid(
        len(runs),
        subplot_width=subplot_width,
        legend_width=legend_width,
        height=height,
        title=title,
    )
    positions = [0.78, 1.0, 1.22]
    colors = [GROUP_PLOT_COLORS[group] for group in GROUP_ORDER]
    by_run_test = tests.set_index(["run_index", "comparison"]) if not tests.empty else None

    for run_index, (ax, run) in enumerate(zip(axes, runs), start=1):
        data = [run.values[group] for group in GROUP_ORDER]
        if kind == "box":
            draw_boxplot_at_positions(ax, data, positions, colors, width=0.16)
        elif kind == "violin":
            draw_violin_at_positions(ax, data, positions, colors, width=0.18)
        else:
            raise ValueError("kind must be 'box' or 'violin'")

        baseline_for_plot = run.baseline_value if show_baseline_line else math.nan
        if show_baseline_line and np.isfinite(run.baseline_value):
            ax.axhline(run.baseline_value, linestyle=":", color=GROUP_COLORS["baseline"], linewidth=1.7)

        lower, upper, span = y_limits_for_values(data, baseline_for_plot, extra_high=0.38)
        bracket_h = 0.035 * span
        y1 = upper - 0.22 * span
        y2 = upper - 0.11 * span
        if by_run_test is not None:
            for comparison, x2, y in [
                ("random_pair_vs_moa_pair", positions[1], y1),
                ("random_pair_vs_explicit_pair", positions[2], y2),
            ]:
                try:
                    q_value = float(by_run_test.loc[(run_index, comparison), "q_value_bh"])
                except Exception:
                    q_value = math.nan
                draw_significance_bracket(ax, positions[0], x2, y, bracket_h, significance_label(q_value))

        if np.isfinite(run.percentile_vs_random):
            ax.text(
                positions[2] + 0.12,
                np.mean(run.values["explicit_pair"]),
                f"{run.percentile_vs_random:.0f}%",
                ha="left",
                va="center",
                fontsize=9,
                color=GROUP_PLOT_COLORS["explicit_pair"],
            )

        ax.set_xlim(0.54, 1.58)
        ax.set_ylim(lower, upper)
        ax.set_xticks([1.0])
        ax.set_xticklabels([wrap_label(run.pair_id, width=14)], rotation=0, ha="center")
        ax.set_title(wrap_label(short_label(run.label, 34), width=16), fontsize=12, pad=8)
        if run_index == 1:
            ax.set_ylabel("Sinkhorn OT to target\n(lower is better)")
        else:
            ax.set_ylabel("")
        polish_axes(ax)

    add_positive_control_legend(legend_ax, show_baseline_line=show_baseline_line)
    fig.subplots_adjust(bottom=0.24)
    savefig(fig, path)
    return path


def plot_additive_distributions(
    runs: Sequence[RunData],
    fig_dir: Path,
    *,
    subplot_width: float,
    legend_width: float,
    height: float,
    title: str,
    show_baseline_line: bool = True,
) -> Path:
    path = fig_dir / "03_multi_explicit_pair_additive_boxplot.png"
    fig, axes, legend_ax = make_axes_grid(
        len(runs),
        subplot_width=subplot_width,
        legend_width=legend_width,
        height=height,
        title=title,
    )
    positions = np.arange(1, len(ADDITIVE_MODE_ORDER) + 1, dtype=float)
    colors = [ADDITIVE_MODE_COLORS.get(mode, "#bdbdbd") for mode in ADDITIVE_MODE_ORDER]

    for run_index, (ax, run) in enumerate(zip(axes, runs), start=1):
        additive = run.additive_df.copy()
        if additive.empty:
            ax.text(0.5, 0.5, "No additive\nresults", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if "mode" not in additive.columns or "score_sinkhorn_ot" not in additive.columns:
            ax.text(0.5, 0.5, "Invalid additive\ntable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        data = [
            finite_values(additive.loc[additive["mode"] == mode, "score_sinkhorn_ot"])
            for mode in ADDITIVE_MODE_ORDER
        ]
        draw_boxplot_at_positions(ax, data, positions, colors, width=0.52)
        baseline_for_plot = run.baseline_value if show_baseline_line else math.nan
        if show_baseline_line and np.isfinite(run.baseline_value):
            ax.axhline(run.baseline_value, linestyle=":", color=GROUP_COLORS["baseline"], linewidth=1.7)
        lower, upper, _ = y_limits_for_values(data, baseline_for_plot, extra_high=0.14)
        ax.set_ylim(lower, upper)
        ax.set_xlim(0.4, len(ADDITIVE_MODE_ORDER) + 0.6)
        ax.set_xticks([float(np.mean(positions))])
        ax.set_xticklabels(
            [wrap_label(run.pair_id, width=14)],
            rotation=0,
            ha="center",
        )
        ax.set_title(wrap_label(short_label(run.label, 34), width=16), fontsize=12, pad=8)
        if run_index == 1:
            ax.set_ylabel("Sinkhorn OT to target\n(lower is better)")
        else:
            ax.set_ylabel("")
        polish_axes(ax)

    add_additive_legend(legend_ax, show_baseline_line=show_baseline_line)
    fig.subplots_adjust(bottom=0.24)
    savefig(fig, path)
    return path


def distribution_tests_table(runs: Sequence[RunData]) -> Any:
    rows: List[Dict[str, Any]] = []
    for run_index, run in enumerate(runs, start=1):
        random_values = run.values["random_pair"]
        for group in ("moa_pair", "explicit_pair"):
            treatment_values = run.values[group]
            p_value = mann_whitney_p_value(random_values, treatment_values)
            rows.append(
                {
                    "run_index": run_index,
                    "run_label": run.label,
                    "run_dir": str(run.run_dir),
                    "pair_id": run.pair_id,
                    "comparison": f"random_pair_vs_{group}",
                    "treatment_group": group,
                    "test": "mann_whitney_u_one_sided_random_greater",
                    "p_value": p_value,
                    "random_n": int(len(random_values)),
                    "treatment_n": int(len(treatment_values)),
                    "random_mean_sinkhorn_ot": float(np.mean(random_values)) if len(random_values) else np.nan,
                    "treatment_mean_sinkhorn_ot": float(np.mean(treatment_values)) if len(treatment_values) else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].values)
    out["significance"] = [significance_label(q) for q in out["q_value_bh"]]
    return out


def positive_control_plot_values_table(runs: Sequence[RunData]) -> Any:
    rows: List[Dict[str, Any]] = []
    for run_index, run in enumerate(runs, start=1):
        for group in GROUP_ORDER:
            values = finite_values(run.values[group])
            rows.append(
                {
                    "run_index": run_index,
                    "run_label": run.label,
                    "run_dir": str(run.run_dir),
                    "conversion_label": run.conversion_label,
                    "pair_id": run.pair_id,
                    "group": group,
                    "n": int(len(values)),
                    "mean_sinkhorn_ot": float(np.mean(values)) if len(values) else np.nan,
                    "median_sinkhorn_ot": float(np.median(values)) if len(values) else np.nan,
                    "std_sinkhorn_ot": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "baseline_mean_sinkhorn_ot": run.baseline_value,
                    "explicit_percentile_lower_than_random": run.percentile_vs_random if group == "explicit_pair" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def additive_plot_values_table(runs: Sequence[RunData]) -> Any:
    rows: List[Dict[str, Any]] = []
    for run_index, run in enumerate(runs, start=1):
        additive = run.additive_df.copy()
        for mode_order, mode in enumerate(ADDITIVE_MODE_ORDER, start=1):
            if additive.empty or "mode" not in additive.columns or "score_sinkhorn_ot" not in additive.columns:
                values = np.asarray([], dtype=float)
            else:
                values = finite_values(additive.loc[additive["mode"] == mode, "score_sinkhorn_ot"])
            rows.append(
                {
                    "run_index": run_index,
                    "run_label": run.label,
                    "run_dir": str(run.run_dir),
                    "pair_id": run.pair_id,
                    "mode_order": mode_order,
                    "mode": mode,
                    "mode_label": ADDITIVE_LEGEND_LABELS[mode],
                    "n": int(len(values)),
                    "mean_sinkhorn_ot": float(np.mean(values)) if len(values) else np.nan,
                    "median_sinkhorn_ot": float(np.median(values)) if len(values) else np.nan,
                    "std_sinkhorn_ot": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "baseline_mean_sinkhorn_ot": run.baseline_value,
                }
            )
    return pd.DataFrame(rows)


def conversion_summary_table(runs: Sequence[RunData]) -> Any:
    rows: List[Dict[str, Any]] = []
    for run_index, run in enumerate(runs, start=1):
        cfg = run.config_payload.get("config", {}) if run.config_payload else {}
        rows.append(
            {
                "run_index": run_index,
                "run_label": run.label,
                "run_dir": str(run.run_dir),
                "conversion_label": run.conversion_label,
                "start_cell": cfg.get("start_cell", ""),
                "target_cell": cfg.get("target_cell", ""),
                "pair_id": run.pair_id,
                "baseline_mean_sinkhorn_ot": run.baseline_value,
                "random_n": int(len(run.values["random_pair"])),
                "moa_n": int(len(run.values["moa_pair"])),
                "explicit_n": int(len(run.values["explicit_pair"])),
                "explicit_percentile_lower_than_random": run.percentile_vs_random,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: Any, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    d = df.head(max_rows).copy()

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6g}"
        text = str(value).replace("|", "\\|")
        return text[:137] + "..." if len(text) > 140 else text

    headers = [str(c).replace("|", "\\|") for c in d.columns]
    rows = [[fmt(v) for v in row] for row in d.itertuples(index=False, name=None)]
    widths = [max(len(header), max([len(row[i]) for row in rows], default=0)) for i, header in enumerate(headers)]
    header = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def write_summary(
    output_dir: Path,
    runs: Sequence[RunData],
    conversion_summary: Any,
    tests: Any,
    figure_paths: Dict[str, Path],
    *,
    show_baseline_line: bool = True,
) -> Path:
    lines = [
        "# Positive-Control 2-Drug Multi-Run Report",
        "",
        f"Output directory: `{output_dir}`",
        f"Input runs: `{len(runs)}`",
        f"Baseline start-to-target reference line: `{'shown' if show_baseline_line else 'hidden'}`",
        "",
        "## Figures",
        f"- `{figure_paths['boxplot'].relative_to(output_dir)}`: side-by-side condensed positive-control boxplots.",
        f"- `{figure_paths['violinplot'].relative_to(output_dir)}`: side-by-side condensed positive-control violin plots.",
        f"- `{figure_paths['additive_boxplot'].relative_to(output_dir)}`: side-by-side additive/sequential boxplots.",
        "",
        "## Conversion Summary",
        markdown_table(conversion_summary, max_rows=50),
        "",
        "## Distribution Tests",
        "One-sided Mann-Whitney U tests compare each conversion's random-pair OT distribution against MOA-shared pairs and the explicit 2-drug pair. Q-values are Benjamini-Hochberg corrected across all comparisons in this report.",
        "",
        markdown_table(tests, max_rows=50),
        "",
    ]
    path = output_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_positive_control_multi_report(
    *,
    run_dirs: Sequence[str | Path],
    output_dir: str | Path,
    labels: Optional[Sequence[str]] = None,
    title_prefix: str = "Positive-control 2-drug comparison",
    subplot_width: float = 2.05,
    legend_width: float = 1.95,
    distribution_height: float = 5.5,
    additive_height: float = 5.5,
    show_baseline_line: bool = True,
) -> Dict[str, str]:
    ensure_dependencies()

    if len(run_dirs) == 0:
        raise ValueError("At least one --run-dirs value is required.")
    if labels is not None and len(labels) != len(run_dirs):
        raise ValueError(f"--labels must contain {len(run_dirs)} values; got {len(labels)}.")

    output_dir = Path(output_dir)
    table_dir = safe_mkdir(output_dir / "tables")
    fig_dir = safe_mkdir(output_dir / "figures")

    runs = [
        load_run(Path(run_dir), labels[i] if labels is not None else None)
        for i, run_dir in enumerate(run_dirs)
    ]

    tests = distribution_tests_table(runs)
    conversion_summary = conversion_summary_table(runs)
    positive_values = positive_control_plot_values_table(runs)
    additive_values = additive_plot_values_table(runs)

    conversion_summary_path = table_dir / "conversion_summary.tsv"
    tests_path = table_dir / "positive_control_distribution_tests.tsv"
    positive_values_path = table_dir / "positive_control_plot_values.tsv"
    additive_values_path = table_dir / "additive_plot_values.tsv"
    conversion_summary.to_csv(conversion_summary_path, sep="\t", index=False)
    tests.to_csv(tests_path, sep="\t", index=False)
    positive_values.to_csv(positive_values_path, sep="\t", index=False)
    additive_values.to_csv(additive_values_path, sep="\t", index=False)

    boxplot = plot_positive_control_distributions(
        runs,
        tests,
        fig_dir,
        kind="box",
        subplot_width=subplot_width,
        legend_width=legend_width,
        height=distribution_height,
        title=f"{title_prefix}: distribution summary",
        show_baseline_line=show_baseline_line,
    )
    violinplot = plot_positive_control_distributions(
        runs,
        tests,
        fig_dir,
        kind="violin",
        subplot_width=subplot_width,
        legend_width=legend_width,
        height=distribution_height,
        title=f"{title_prefix}: violin summary",
        show_baseline_line=show_baseline_line,
    )
    additive_boxplot = plot_additive_distributions(
        runs,
        fig_dir,
        subplot_width=subplot_width,
        legend_width=legend_width,
        height=additive_height,
        title=f"{title_prefix}: additive and sequential modes",
        show_baseline_line=show_baseline_line,
    )

    summary = write_summary(
        output_dir,
        runs,
        conversion_summary,
        tests,
        {
            "boxplot": boxplot,
            "violinplot": violinplot,
            "additive_boxplot": additive_boxplot,
        },
        show_baseline_line=show_baseline_line,
    )

    return {
        "report_dir": str(output_dir),
        "summary": str(summary),
        "conversion_summary": str(conversion_summary_path),
        "distribution_tests": str(tests_path),
        "positive_control_plot_values": str(positive_values_path),
        "additive_plot_values": str(additive_values_path),
        "boxplot": str(boxplot),
        "violinplot": str(violinplot),
        "additive_boxplot": str(additive_boxplot),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pharos hypothesis-driven summarize",
        description="Create side-by-side multi-run positive-control 2-drug comparison plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        help="Output directories produced by positive_control_2drug_analysis.py.",
    )
    p.add_argument("--output-dir", required=True, help="New report directory where figures/tables will be written.")
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional subplot labels, one per run directory. Defaults to each run directory name.",
    )
    p.add_argument("--title-prefix", default="Positive-control 2-drug comparison", help="Figure title prefix.")
    p.add_argument("--subplot-width", type=float, default=2.05, help="Width in inches for each conversion subplot.")
    p.add_argument("--legend-width", type=float, default=1.95, help="Dedicated legend column width in inches.")
    p.add_argument("--distribution-height", type=float, default=5.5, help="Height in inches for box/violin figures.")
    p.add_argument("--additive-height", type=float, default=5.5, help="Height in inches for additive figure.")
    p.add_argument(
        "--baseline-line",
        dest="show_baseline_line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Draw the dotted baseline start-to-target reference line and include it in plot y-limits. "
            "Use --no-baseline-line to hide it and autoscale to the plotted distributions."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    out = make_positive_control_multi_report(
        run_dirs=args.run_dirs,
        output_dir=args.output_dir,
        labels=args.labels,
        title_prefix=args.title_prefix,
        subplot_width=args.subplot_width,
        legend_width=args.legend_width,
        distribution_height=args.distribution_height,
        additive_height=args.additive_height,
        show_baseline_line=args.show_baseline_line,
    )
    print("\n=== Positive-control multi-run report complete ===")
    print(f"summary: {out['summary']}")
    print(f"figures: {Path(out['boxplot']).parent}")
    print(f"tables:  {Path(out['conversion_summary']).parent}")


if __name__ == "__main__":
    main()
