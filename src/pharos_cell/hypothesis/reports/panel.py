#!/usr/bin/env python
"""
Report for explicit 2-drug panel positive-control runs.

This report expects a run created by positive_control_2drug_panel_analysis.py:
multiple selected rows with group == "explicit_pair" and one shared random-pair
control distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd


EXPLICIT_COLOR = "#2f6fed"
RANDOM_COLOR = "#9aa0a6"
GAIN_COLOR = "#4c78a8"
PERCENT_COLOR = "#72b7b2"
FDA_COLOR = "#2f6fed"
FAILED_COLOR = "#e15759"
PAIR_GROUP_FALLBACK_COLORS = (
    "#59a14f",
    "#f28e2b",
    "#b07aa1",
    "#edc948",
    "#76b7b2",
    "#ff9da7",
)


def _load_matplotlib_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def finite_values(values: Sequence[Any]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def wrap_label(value: Any, width: int = 24) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def load_run_tables(run_dir: str | Path) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    config_path = run_dir / "positive_control_config.used.json"
    required = {
        "selected": tables_dir / "selected_pairs.tsv",
        "evaluation": tables_dir / "evaluation_results.tsv",
        "baseline": tables_dir / "baseline_results.tsv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required positive-control panel tables: {missing}")

    config_payload = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "run_dir": run_dir,
        "config": config_payload,
        "selected": pd.read_csv(required["selected"], sep="\t"),
        "evaluation": pd.read_csv(required["evaluation"], sep="\t"),
        "baseline": pd.read_csv(required["baseline"], sep="\t"),
    }


def summarize_panel(evaluation: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    explicit_eval = evaluation[evaluation["group"] == "explicit_pair"].copy()
    if explicit_eval.empty:
        raise ValueError("No group == 'explicit_pair' rows found in evaluation_results.tsv")

    random_values = finite_values(evaluation.loc[evaluation["group"] == "random_pair", "score_sinkhorn_ot"])
    summary = (
        explicit_eval.groupby(["group", "pair_id"], as_index=False)
        .agg(
            n_batches=("score_sinkhorn_ot", "size"),
            mean_sinkhorn_ot=("score_sinkhorn_ot", "mean"),
            median_sinkhorn_ot=("score_sinkhorn_ot", "median"),
            std_sinkhorn_ot=("score_sinkhorn_ot", "std"),
            min_sinkhorn_ot=("score_sinkhorn_ot", "min"),
            max_sinkhorn_ot=("score_sinkhorn_ot", "max"),
            mean_gain_sinkhorn_vs_baseline=("gain_sinkhorn_vs_baseline", "mean")
            if "gain_sinkhorn_vs_baseline" in explicit_eval.columns
            else ("delta_sinkhorn_from_baseline", lambda x: -float(pd.to_numeric(x, errors="coerce").mean())),
        )
        .copy()
    )
    summary["std_sinkhorn_ot"] = summary["std_sinkhorn_ot"].fillna(0.0)
    summary["sem_sinkhorn_ot"] = summary["std_sinkhorn_ot"] / np.sqrt(summary["n_batches"].clip(lower=1))
    if len(random_values):
        summary["percent_random_worse_or_equal"] = summary["mean_sinkhorn_ot"].map(
            lambda value: 100.0 * float(np.mean(random_values >= float(value)))
            if np.isfinite(float(value))
            else math.nan
        )
    else:
        summary["percent_random_worse_or_equal"] = math.nan

    selected_cols = [
        c
        for c in [
            "pair_id",
            "first_drug",
            "second_drug",
            "ordered_pair_id",
            "explicit_panel_index",
            "pair_group",
            "explicit_order",
            "first_dose",
            "first_dose_unit",
            "second_dose",
            "second_dose_unit",
            "selection_score_sinkhorn_ot",
            "selection_score_energy_distance",
            "n_concentration_combinations_scored",
            "n_ordered_drug_orders_scored",
        ]
        if c in selected.columns
    ]
    selected_explicit = selected[selected["group"] == "explicit_pair"].copy()
    selected_explicit["_selected_row_order"] = np.arange(len(selected_explicit))
    if "explicit_panel_index" in selected_explicit.columns:
        selected_explicit["_panel_table_order"] = pd.to_numeric(
            selected_explicit["explicit_panel_index"], errors="coerce"
        )
    else:
        selected_explicit["_panel_table_order"] = np.nan
    selected_explicit["_panel_table_order"] = selected_explicit["_panel_table_order"].fillna(
        selected_explicit["_selected_row_order"]
    )
    selected_info = selected_explicit[
        selected_cols + ["_panel_table_order", "_selected_row_order"]
    ].drop_duplicates("pair_id")
    summary = summary.merge(selected_info, on="pair_id", how="left")
    summary["_panel_table_order"] = summary["_panel_table_order"].fillna(len(selected_explicit))
    summary["_selected_row_order"] = summary["_selected_row_order"].fillna(len(selected_explicit))
    summary = summary.sort_values(
        ["_panel_table_order", "_selected_row_order", "pair_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return summary.drop(columns=["_panel_table_order", "_selected_row_order"])


def save_summary_table(summary: pd.DataFrame, output_dir: Path) -> Path:
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "explicit_pair_panel_summary.tsv"
    summary.to_csv(path, sep="\t", index=False)
    return path


def _padded_ylim(values: np.ndarray, pad_fraction: float = 0.08) -> tuple[float, float] | None:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if not np.isfinite(y_min) or not np.isfinite(y_max):
        return None
    if y_min == y_max:
        pad = max(abs(y_max) * pad_fraction, 1e-9)
    else:
        pad = max((y_max - y_min) * pad_fraction, 1e-9)
    lower = max(0.0, y_min - pad) if y_min >= 0.0 else y_min - pad
    return float(lower), float(y_max + pad)


def _full_data_ylim(
    groups: Sequence[np.ndarray],
    extra_values: Sequence[float] | np.ndarray = (),
    pad_fraction: float = 0.08,
) -> tuple[float, float] | None:
    values = np.concatenate([g for g in groups if len(g)]) if any(len(g) for g in groups) else np.asarray([])
    extra = np.asarray(extra_values, dtype=float)
    if len(extra):
        values = np.concatenate([values, extra])
    return _padded_ylim(values, pad_fraction=pad_fraction)


def _artist_y_values(artist: Any) -> np.ndarray:
    if hasattr(artist, "get_ydata"):
        return finite_values(artist.get_ydata())
    if hasattr(artist, "get_path"):
        vertices = artist.get_path().vertices
        if vertices is not None and len(vertices):
            return finite_values(vertices[:, 1])
    return np.asarray([])


def _boxplot_visible_ylim(
    boxplot_artists: Mapping[str, Any],
    extra_values: Sequence[float] | np.ndarray = (),
    pad_fraction: float = 0.08,
) -> tuple[float, float] | None:
    values = []
    for key in ["boxes", "whiskers", "caps", "medians", "means"]:
        for artist in boxplot_artists.get(key, []):
            artist_values = _artist_y_values(artist)
            if len(artist_values):
                values.append(artist_values)
    extra = np.asarray(extra_values, dtype=float)
    if len(extra):
        values.append(extra[np.isfinite(extra)])
    if not values:
        return None
    return _padded_ylim(np.concatenate(values), pad_fraction=pad_fraction)


def _boxplot_group_visible_values(boxplot_artists: Mapping[str, Any], group_index: int) -> np.ndarray:
    values = []
    if group_index < len(boxplot_artists.get("boxes", [])):
        values.append(_artist_y_values(boxplot_artists["boxes"][group_index]))
    if group_index < len(boxplot_artists.get("medians", [])):
        values.append(_artist_y_values(boxplot_artists["medians"][group_index]))
    for key in ["whiskers", "caps"]:
        artists = boxplot_artists.get(key, [])
        for artist in artists[2 * group_index : 2 * group_index + 2]:
            values.append(_artist_y_values(artist))
    values = [v for v in values if len(v)]
    return np.concatenate(values) if values else np.asarray([])


def _rounded_percent_label(value: Any) -> str | None:
    try:
        percent = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(percent):
        return None
    percent = min(max(percent, 0.0), 100.0)
    return f"{int(math.floor(percent + 0.5))}%"


def _best_explicit_pair_annotation(summary: pd.DataFrame) -> tuple[int, int, str] | None:
    if summary.empty or "mean_sinkhorn_ot" not in summary.columns or "percent_random_worse_or_equal" not in summary.columns:
        return None
    mean_scores = pd.to_numeric(summary["mean_sinkhorn_ot"], errors="coerce").to_numpy(dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(mean_scores))
    if len(finite_positions) == 0:
        return None
    summary_position = int(finite_positions[np.argmin(mean_scores[finite_positions])])
    label = _rounded_percent_label(summary.iloc[summary_position]["percent_random_worse_or_equal"])
    if label is None:
        return None
    group_index = summary_position + 1
    plot_position = summary_position + 2
    return group_index, plot_position, label


def _add_best_pair_percentile_label(
    ax: Any,
    summary: pd.DataFrame,
    groups: Sequence[np.ndarray],
    group_visible_values: Sequence[np.ndarray] | None = None,
) -> None:
    annotation = _best_explicit_pair_annotation(summary)
    if annotation is None:
        return
    group_index, plot_position, label = annotation
    if group_index >= len(groups):
        return

    if group_visible_values is not None and group_index < len(group_visible_values):
        group_values = finite_values(group_visible_values[group_index])
    else:
        group_values = finite_values(groups[group_index])
    if len(group_values) == 0:
        return

    y_low, y_high = ax.get_ylim()
    span = y_high - y_low
    if not np.isfinite(span) or span <= 0:
        return
    y = float(np.min(group_values)) - 0.035 * span
    y = max(y_low + 0.035 * span, y)
    ax.text(
        plot_position,
        y,
        label,
        color=EXPLICIT_COLOR,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        clip_on=False,
    )


def _add_random_vs_best_pair_significance(
    ax: Any,
    summary: pd.DataFrame,
    groups: Sequence[np.ndarray],
    comparison: Mapping[str, Any] | None,
) -> None:
    annotation = _best_explicit_pair_annotation(summary)
    if annotation is None or len(groups) == 0:
        return
    group_index, plot_position, _ = annotation
    if group_index >= len(groups):
        return

    random_values = finite_values(groups[0])
    best_values = finite_values(groups[group_index])
    if len(random_values) == 0 or len(best_values) == 0:
        return

    # The raw test is calculated before plotting, together with the other report
    # comparisons, so this annotation always uses the BH-adjusted result.
    label = str(comparison.get("significance", "n/a")) if comparison else "n/a"

    y_low, y_high = ax.get_ylim()
    span = y_high - y_low
    if not np.isfinite(span) or span <= 0:
        return
    bracket_base = max(float(np.max(random_values)), float(np.max(best_values))) + 0.035 * span
    bracket_height = 0.025 * span
    new_top = max(y_high, bracket_base + bracket_height + 0.05 * span)
    if new_top > y_high:
        ax.set_ylim(y_low, new_top)
    draw_significance_bracket(ax, 1, plot_position, bracket_base, bracket_height, label)


def _normalize_group_value(value: Any) -> str:
    return str(value).strip().casefold()


def _display_group_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _ordered_pair_group_values(summary: pd.DataFrame, pair_group_col: str) -> list[str]:
    if pair_group_col not in summary.columns:
        return []
    values: list[str] = []
    for value in summary[pair_group_col].tolist():
        display = _display_group_value(value)
        if display:
            values.append(display)
    return values


def _pair_group_color_map(
    group_values: Sequence[str],
    *,
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
) -> Dict[str, str]:
    ordered_groups = list(dict.fromkeys(_display_group_value(value) for value in group_values))
    ordered_groups = [value for value in ordered_groups if value]
    colors: Dict[str, str] = {}
    fallback_i = 0
    fda_norm = _normalize_group_value(fda_group_value)
    failed_norm = _normalize_group_value(failed_group_value)
    for group_value in ordered_groups:
        group_norm = _normalize_group_value(group_value)
        if group_norm == fda_norm:
            colors[group_value] = FDA_COLOR
        elif group_norm == failed_norm:
            colors[group_value] = FAILED_COLOR
        else:
            colors[group_value] = PAIR_GROUP_FALLBACK_COLORS[
                fallback_i % len(PAIR_GROUP_FALLBACK_COLORS)
            ]
            fallback_i += 1
    return colors


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value <= 1e-4:
        return "****"
    if p_value <= 1e-3:
        return "***"
    if p_value <= 1e-2:
        return "**"
    if p_value <= 5e-2:
        return "*"
    return "ns"


def permutation_mean_p_value(control: Sequence[float], treatment: Sequence[float], n_iter: int = 10000) -> float:
    """One-sided fallback for mean(control) > mean(treatment)."""
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
        shuffled = rng.permutation(pooled)
        diff = float(np.mean(shuffled[:n_control]) - np.mean(shuffled[n_control:]))
        if diff >= observed:
            more_extreme += 1
    return float(more_extreme / (int(n_iter) + 1))


def mann_whitney_p_value(control: Sequence[float], treatment: Sequence[float]) -> float:
    """One-sided test for lower treatment OT than control OT."""
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


def benjamini_hochberg_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini--Hochberg FDR-adjusted p-values in input order.

    Non-finite p-values are preserved as ``nan`` and are excluded from the
    number of hypotheses in the correction.
    """
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, math.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return adjusted

    valid_values = np.clip(values[valid], 0.0, 1.0)
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    n_tests = len(ranked)
    adjusted_ranked = ranked * n_tests / np.arange(1, n_tests + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_valid = np.empty(n_tests, dtype=float)
    adjusted_valid[order] = np.minimum(adjusted_ranked, 1.0)
    adjusted[valid] = adjusted_valid
    return adjusted


def apply_bh_correction(comparisons: Sequence[Dict[str, Any] | None]) -> None:
    """Apply BH correction across all valid Mann--Whitney tests in one report.

    The mapping is updated in place. ``p_value`` is the corrected value for
    backwards-compatible report consumers; the raw and corrected values are
    also retained explicitly.
    """
    valid_comparisons = [comparison for comparison in comparisons if comparison is not None]
    raw_p_values = [float(comparison["p_value_raw"]) for comparison in valid_comparisons]
    corrected_p_values = benjamini_hochberg_adjust(raw_p_values)
    n_tests = int(np.isfinite(raw_p_values).sum())
    for comparison, corrected_p_value in zip(valid_comparisons, corrected_p_values):
        comparison["p_value_bh"] = float(corrected_p_value)
        comparison["p_value"] = float(corrected_p_value)
        comparison["multiple_testing_correction"] = "Benjamini-Hochberg"
        comparison["n_tests_bh"] = n_tests
        comparison["significance"] = significance_label(corrected_p_value)


def _pair_group_mask(df: pd.DataFrame, pair_group_col: str, group_value: str) -> pd.Series:
    if pair_group_col not in df.columns:
        return pd.Series(False, index=df.index)
    target = _normalize_group_value(group_value)
    return df[pair_group_col].map(_normalize_group_value) == target


def compute_random_vs_best_pair_comparison(
    evaluation: pd.DataFrame,
    summary: pd.DataFrame,
) -> Dict[str, Any] | None:
    """Compute the raw one-sided random-control versus best-pair comparison."""
    annotation = _best_explicit_pair_annotation(summary)
    if annotation is None:
        return None
    best_pair_id = str(summary.iloc[annotation[0] - 1]["pair_id"])
    random_values = finite_values(evaluation.loc[evaluation["group"] == "random_pair", "score_sinkhorn_ot"])
    best_values = finite_values(
        evaluation.loc[
            (evaluation["group"] == "explicit_pair") & (evaluation["pair_id"].astype(str) == best_pair_id),
            "score_sinkhorn_ot",
        ]
    )
    if len(random_values) == 0 or len(best_values) == 0:
        return None
    p_value_raw = mann_whitney_p_value(random_values, best_values)
    return {
        "comparison": f"{best_pair_id} vs random pairs",
        "test": "mann_whitney_u_one_sided_random_greater_than_best_pair",
        "unit": "batch_level_sinkhorn_ot",
        "alternative": "random-pair OT greater than best explicit-pair OT",
        "n_random_rows": int(len(random_values)),
        "n_best_pair_rows": int(len(best_values)),
        "p_value_raw": float(p_value_raw),
    }


def compute_pair_group_comparison(
    summary: pd.DataFrame,
    *,
    pair_group_col: str = "pair_group",
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
) -> Dict[str, Any] | None:
    if summary.empty or pair_group_col not in summary.columns or "mean_sinkhorn_ot" not in summary.columns:
        return None
    failed_values = finite_values(summary.loc[_pair_group_mask(summary, pair_group_col, failed_group_value), "mean_sinkhorn_ot"])
    fda_values = finite_values(summary.loc[_pair_group_mask(summary, pair_group_col, fda_group_value), "mean_sinkhorn_ot"])
    if len(failed_values) == 0 or len(fda_values) == 0:
        return None
    p_value_raw = mann_whitney_p_value(failed_values, fda_values)
    return {
        "comparison": f"{fda_group_value} vs {failed_group_value}",
        "test": "mann_whitney_u_one_sided_failed_greater_than_fda",
        "unit": "pair_level_mean_sinkhorn_ot",
        "alternative": "failed_trial mean OT greater than FDA-approved mean OT",
        "n_failed_pairs": int(len(failed_values)),
        "n_fda_pairs": int(len(fda_values)),
        "failed_mean_of_pair_means": float(np.mean(failed_values)),
        "fda_mean_of_pair_means": float(np.mean(fda_values)),
        "delta_failed_minus_fda": float(np.mean(failed_values) - np.mean(fda_values)),
        "p_value_raw": float(p_value_raw),
    }


def save_pair_group_comparison_table(
    comparison: Mapping[str, Any] | None,
    output_dir: Path,
) -> Path | None:
    if not comparison:
        return None
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / "pair_group_comparison.tsv"
    pd.DataFrame([comparison]).to_csv(path, sep="\t", index=False)
    return path


def draw_significance_bracket(ax: Any, x1: float, x2: float, y: float, height: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], color="#222222", linewidth=1.1, clip_on=False)
    ax.text((x1 + x2) / 2, y + height, label, ha="center", va="bottom", fontsize=11, color="#222222")


def _panel_score_groups(
    evaluation: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    pair_group_col: str = "pair_group",
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
) -> tuple[list[str], list[np.ndarray], list[str], list[str | None], Dict[str, str]]:
    random_values = finite_values(evaluation.loc[evaluation["group"] == "random_pair", "score_sinkhorn_ot"])
    labels = ["Random\npairs"]
    groups = [random_values]
    colors = [RANDOM_COLOR]
    group_labels: list[str | None] = [None]
    group_color_map = _pair_group_color_map(
        _ordered_pair_group_values(summary, pair_group_col),
        fda_group_value=fda_group_value,
        failed_group_value=failed_group_value,
    )
    explicit_eval = evaluation[evaluation["group"] == "explicit_pair"].copy()
    explicit_eval["pair_id"] = explicit_eval["pair_id"].astype(str)
    for _, row in summary.iterrows():
        pair_id = str(row["pair_id"])
        pair_group = _display_group_value(row[pair_group_col]) if pair_group_col in summary.columns else ""
        labels.append(wrap_label(pair_id, width=18))
        groups.append(
            finite_values(
                explicit_eval.loc[explicit_eval["pair_id"] == pair_id, "score_sinkhorn_ot"]
            )
        )
        colors.append(group_color_map.get(pair_group, EXPLICIT_COLOR))
        group_labels.append(pair_group or None)
    return labels, groups, colors, group_labels, group_color_map


def _color_pair_group_tick_labels(ax: Any, colors: Sequence[str], group_labels: Sequence[str | None]) -> None:
    for tick, color, group_label in zip(ax.get_xticklabels(), colors, group_labels):
        if group_label:
            tick.set_color(color)
            tick.set_fontweight("bold")


def _add_panel_group_legend(
    ax: Any,
    *,
    group_color_map: Mapping[str, str],
    baseline_handle: Any = None,
    baseline_label: str = "Baseline mean",
) -> None:
    handles = []
    labels = []
    if group_color_map:
        from matplotlib.patches import Patch

        for group_label, color in group_color_map.items():
            handles.append(Patch(facecolor=color, edgecolor="#333333", alpha=0.72))
            labels.append(group_label)
    if baseline_handle is not None:
        handles.append(baseline_handle)
        labels.append(baseline_label)
    if handles:
        ax.legend(handles, labels, frameon=False, loc="best")


def plot_panel_boxplot(
    evaluation: pd.DataFrame,
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    fig_dir: Path,
    *,
    pair_group_col: str = "pair_group",
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
) -> Path:
    plt = _load_matplotlib_pyplot()
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "01_explicit_pair_panel_boxplot.png"

    labels, groups, colors, group_labels, group_color_map = _panel_score_groups(
        evaluation,
        summary,
        pair_group_col=pair_group_col,
        fda_group_value=fda_group_value,
        failed_group_value=failed_group_value,
    )

    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(labels)), 5.2), constrained_layout=True)
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor("#333333")
    for element in ["whiskers", "caps", "medians"]:
        for artist in bp[element]:
            artist.set_color("#333333")

    baseline_values = finite_values(baseline["score_sinkhorn_ot"]) if "score_sinkhorn_ot" in baseline.columns else np.asarray([])
    baseline_handle = None
    if len(baseline_values):
        baseline_handle = ax.axhline(
            float(np.mean(baseline_values)),
            color="#d62728",
            linestyle="--",
            linewidth=1.2,
        )

    ylim = _boxplot_visible_ylim(bp, extra_values=[float(np.mean(baseline_values))] if len(baseline_values) else [])
    if ylim is not None:
        ax.set_ylim(*ylim)
    group_visible_values = [_boxplot_group_visible_values(bp, i) for i in range(len(groups))]
    _add_best_pair_percentile_label(ax, summary, groups, group_visible_values=group_visible_values)
    ax.set_ylabel("Sinkhorn OT to target (lower is better)")
    ax.set_title("FDA-approved pair panel versus random 2-drug controls")
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    _color_pair_group_tick_labels(ax, colors, group_labels)
    _add_panel_group_legend(ax, group_color_map=group_color_map, baseline_handle=baseline_handle)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_panel_violin(
    evaluation: pd.DataFrame,
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    fig_dir: Path,
    *,
    pair_group_col: str = "pair_group",
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
    random_vs_best_comparison: Mapping[str, Any] | None = None,
) -> Path:
    plt = _load_matplotlib_pyplot()
    path = fig_dir / "04_explicit_pair_panel_violin.png"

    labels, groups, colors, group_labels, group_color_map = _panel_score_groups(
        evaluation,
        summary,
        pair_group_col=pair_group_col,
        fda_group_value=fda_group_value,
        failed_group_value=failed_group_value,
    )
    positions = np.arange(1, len(labels) + 1)
    nonempty = [(pos, group, color) for pos, group, color in zip(positions, groups, colors) if len(group)]

    fig_width = max(4.8, min(7.2, 2.0 + 0.42 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 3.35))
    if nonempty:
        violin = ax.violinplot(
            [group for _, group, _ in nonempty],
            positions=[pos for pos, _, _ in nonempty],
            showmeans=False,
            showmedians=True,
            showextrema=True,
            widths=0.55,
        )
        for body, (_, _, color) in zip(violin["bodies"], nonempty):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_linewidth(0.8)
            body.set_alpha(0.62)
        for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
            if key in violin:
                violin[key].set_color("#333333")
                violin[key].set_linewidth(0.85)

    baseline_values = finite_values(baseline["score_sinkhorn_ot"]) if "score_sinkhorn_ot" in baseline.columns else np.asarray([])
    baseline_handle = None
    if len(baseline_values):
        baseline_mean = float(np.mean(baseline_values))
        baseline_handle = ax.axhline(
            baseline_mean,
            color="#d62728",
            linestyle="--",
            linewidth=0.9,
        )

    ylim = _full_data_ylim(groups, extra_values=[float(np.mean(baseline_values))] if len(baseline_values) else [])
    if ylim is not None:
        y_low, y_high = ylim
        span = max(y_high - y_low, abs(y_high) * 0.08, 1e-9)
        ax.set_ylim(y_low - 0.10 * span, y_high)
    _add_random_vs_best_pair_significance(ax, summary, groups, random_vs_best_comparison)
    _add_best_pair_percentile_label(ax, summary, groups)
    ax.set_ylabel("Sinkhorn OT\n(lower is better)", fontsize=8)
    ax.set_title("Pair panel vs random controls", fontsize=9, pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    _color_pair_group_tick_labels(ax, colors, group_labels)
    ax.tick_params(axis="y", labelsize=8, length=3)
    ax.tick_params(axis="x", length=3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _add_panel_group_legend(ax, group_color_map=group_color_map, baseline_handle=baseline_handle)
    bottom_margin = 0.34 if len(labels) <= 9 else 0.40
    fig.subplots_adjust(left=0.15, right=0.98, bottom=bottom_margin, top=0.86)
    panel_dpi = 300
    fig.savefig(path, dpi=panel_dpi)
    plt.close(fig)
    return path


def plot_pair_group_comparison_violin(
    evaluation: pd.DataFrame,
    summary: pd.DataFrame,
    fig_dir: Path,
    *,
    pair_group_col: str = "pair_group",
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
    comparison: Mapping[str, Any] | None = None,
) -> Path | None:
    if pair_group_col not in evaluation.columns or pair_group_col not in summary.columns:
        return None

    plt = _load_matplotlib_pyplot()
    path = fig_dir / "05_pair_group_comparison_violin.png"
    random_values = finite_values(evaluation.loc[evaluation["group"] == "random_pair", "score_sinkhorn_ot"])
    explicit_eval = evaluation[evaluation["group"] == "explicit_pair"].copy()
    failed_values = finite_values(
        explicit_eval.loc[_pair_group_mask(explicit_eval, pair_group_col, failed_group_value), "score_sinkhorn_ot"]
    )
    fda_values = finite_values(
        explicit_eval.loc[_pair_group_mask(explicit_eval, pair_group_col, fda_group_value), "score_sinkhorn_ot"]
    )
    if len(failed_values) == 0 or len(fda_values) == 0:
        return None

    labels = ["Random\ncontrols", "Failed clinical\ntrials", "FDA\napproved"]
    groups = [random_values, failed_values, fda_values]
    colors = [RANDOM_COLOR, FAILED_COLOR, FDA_COLOR]
    positions = np.arange(1, len(labels) + 1)
    nonempty = [(pos, group, color) for pos, group, color in zip(positions, groups, colors) if len(group)]

    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    violin = ax.violinplot(
        [group for _, group, _ in nonempty],
        positions=[pos for pos, _, _ in nonempty],
        showmeans=False,
        showmedians=True,
        showextrema=True,
    )
    for body, (_, _, color) in zip(violin["bodies"], nonempty):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.58)
    for key in ["cbars", "cmins", "cmaxes", "cmedians"]:
        if key in violin:
            violin[key].set_color("#333333")
            violin[key].set_linewidth(1.0)

    ylim = _full_data_ylim(groups)
    if ylim is not None:
        y_low, y_high = ylim
        span = max(y_high - y_low, abs(y_high) * 0.08, 1e-9)
        bracket_y = y_high + 0.05 * span
        bracket_h = 0.04 * span
        ax.set_ylim(y_low, y_high + 0.18 * span)
        label = str(comparison.get("significance", "n/a")) if comparison else "n/a"
        draw_significance_bracket(ax, 2, 3, bracket_y, bracket_h, label)

    ax.set_ylabel("Sinkhorn OT to target (lower is better)")
    ax.set_title("FDA-approved versus failed-trial pair cohorts")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_percentile_bar(summary: pd.DataFrame, fig_dir: Path) -> Path:
    plt = _load_matplotlib_pyplot()
    path = fig_dir / "02_explicit_pair_percentile_vs_random.png"
    d = summary.sort_values("percent_random_worse_or_equal", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.42 * len(d))), constrained_layout=True)
    y = np.arange(len(d))
    ax.barh(y, d["percent_random_worse_or_equal"], color=PERCENT_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, width=28) for x in d["pair_id"]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Random pairs worse than or equal to explicit pair (%)")
    ax.set_title("Explicit pair rank versus random controls")
    for yi, value in zip(y, d["percent_random_worse_or_equal"]):
        if np.isfinite(value):
            ax.text(min(float(value) + 1.0, 99.0), yi, f"{float(value):.0f}%", va="center", fontsize=8)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_mean_gain(summary: pd.DataFrame, fig_dir: Path, errorbar: str = "std") -> Path:
    plt = _load_matplotlib_pyplot()
    path = fig_dir / "03_explicit_pair_mean_gain.png"
    d = summary.sort_values("mean_gain_sinkhorn_vs_baseline", ascending=False).copy()
    err_col = "sem_sinkhorn_ot" if errorbar == "sem" else "std_sinkhorn_ot"
    fig, ax = plt.subplots(figsize=(max(8.0, 0.65 * len(d)), 5.0), constrained_layout=True)
    x = np.arange(len(d))
    ax.bar(x, d["mean_gain_sinkhorn_vs_baseline"], color=GAIN_COLOR, alpha=0.88)
    if err_col in d.columns:
        ax.errorbar(x, d["mean_gain_sinkhorn_vs_baseline"], yerr=d[err_col], fmt="none", color="#333333", capsize=3)
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(x, width=18) for x in d["pair_id"]], rotation=45, ha="right")
    ax.set_ylabel("Gain versus baseline Sinkhorn OT")
    ax.set_title("Mean explicit-pair gain over baseline")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def write_summary_md(
    *,
    output_dir: Path,
    run_dir: Path,
    config_payload: Mapping[str, Any],
    summary: pd.DataFrame,
    table_path: Path,
    comparison_table_path: Path | None = None,
    comparison: Mapping[str, Any] | None = None,
    figure_paths: Mapping[str, Path],
) -> Path:
    def markdown_table(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No rows._"
        headers = list(df.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
        return "\n".join(lines)

    path = output_dir / "summary.md"
    metadata = config_payload.get("metadata", {}) if config_payload else {}
    config = config_payload.get("config", {}) if config_payload else {}
    show_cols = [
        "pair_id",
        "pair_group",
        "first_drug",
        "second_drug",
        "mean_sinkhorn_ot",
        "std_sinkhorn_ot",
        "mean_gain_sinkhorn_vs_baseline",
        "percent_random_worse_or_equal",
    ]
    show = summary[[c for c in show_cols if c in summary.columns]].copy()
    for col in show.columns:
        if pd.api.types.is_float_dtype(show[col]):
            show[col] = show[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
        else:
            show[col] = show[col].fillna("").astype(str)

    lines = [
        "# Explicit 2-Drug Pair Panel Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Explicit panel pairs selected: `{metadata.get('n_explicit_selected_pairs', len(summary))}`",
        f"- Random pairs scored: `{metadata.get('n_random_pairs_scored', 'unknown')}`",
        f"- Random pair evaluation rows: `{metadata.get('n_random_pair_evaluation_rows', 'unknown')}`",
        f"- Batches: `{metadata.get('n_batches', config.get('n_batches', 'unknown'))}`",
        "",
        "## Explicit Pair Summary",
        "",
        markdown_table(show),
        "",
        "## Outputs",
        "",
        f"- `tables/explicit_pair_panel_summary.tsv`: `{table_path}`",
    ]
    if comparison_table_path is not None:
        lines.append(f"- `tables/pair_group_comparison.tsv`: `{comparison_table_path}`")
    for label, fig_path in figure_paths.items():
        lines.append(f"- `{label}`: `{fig_path}`")
    if comparison:
        lines.extend(
            [
                "",
                "## Pair Group Comparison",
                "",
                (
                    f"- FDA-approved mean of pair means: "
                    f"`{float(comparison['fda_mean_of_pair_means']):.6g}`"
                ),
                (
                    f"- Failed-trial mean of pair means: "
                    f"`{float(comparison['failed_mean_of_pair_means']):.6g}`"
                ),
                f"- Delta failed minus FDA: `{float(comparison['delta_failed_minus_fda']):.6g}`",
                f"- Raw P-value: `{float(comparison['p_value_raw']):.6g}`",
                (
                    f"- BH-adjusted P-value: `{float(comparison['p_value_bh']):.6g}` "
                    f"({comparison['significance']}; {int(comparison['n_tests_bh'])} tests)"
                ),
                f"- Test unit: `{comparison['unit']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n")
    return path


def make_positive_control_panel_report(
    *,
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    errorbar: str = "std",
    pair_group_col: str = "pair_group",
    fda_group_value: str = "FDA_approved",
    failed_group_value: str = "failed_trial",
) -> Dict[str, Any]:
    data = load_run_tables(run_dir)
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir else run_dir / "panel_report"
    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize_panel(data["evaluation"], data["selected"])
    table_path = save_summary_table(summary, output_dir)
    random_vs_best_comparison = compute_random_vs_best_pair_comparison(data["evaluation"], summary)
    comparison = compute_pair_group_comparison(
        summary,
        pair_group_col=pair_group_col,
        fda_group_value=fda_group_value,
        failed_group_value=failed_group_value,
    )
    # Correct the complete family of Mann--Whitney tests shown in this report.
    # Plot asterisks and reported significance labels therefore reflect FDR-
    # adjusted, rather than raw, p-values.
    apply_bh_correction([random_vs_best_comparison, comparison])
    comparison_table_path = save_pair_group_comparison_table(comparison, output_dir)
    figure_paths = {
        "boxplot": plot_panel_boxplot(
            data["evaluation"],
            summary,
            data["baseline"],
            fig_dir,
            pair_group_col=pair_group_col,
            fda_group_value=fda_group_value,
            failed_group_value=failed_group_value,
        ),
        "violin": plot_panel_violin(
            data["evaluation"],
            summary,
            data["baseline"],
            fig_dir,
            pair_group_col=pair_group_col,
            fda_group_value=fda_group_value,
            failed_group_value=failed_group_value,
            random_vs_best_comparison=random_vs_best_comparison,
        ),
        "percentile_vs_random": plot_percentile_bar(summary, fig_dir),
        "mean_gain": plot_mean_gain(summary, fig_dir, errorbar=errorbar),
    }
    pair_group_violin = plot_pair_group_comparison_violin(
        data["evaluation"],
        summary,
        fig_dir,
        pair_group_col=pair_group_col,
        fda_group_value=fda_group_value,
        failed_group_value=failed_group_value,
        comparison=comparison,
    )
    if pair_group_violin is not None:
        figure_paths["pair_group_comparison_violin"] = pair_group_violin
    summary_md = write_summary_md(
        output_dir=output_dir,
        run_dir=run_dir,
        config_payload=data["config"],
        summary=summary,
        table_path=table_path,
        comparison_table_path=comparison_table_path,
        comparison=comparison,
        figure_paths=figure_paths,
    )
    tables = {"explicit_pair_panel_summary": str(table_path)}
    if comparison_table_path is not None:
        tables["pair_group_comparison"] = str(comparison_table_path)
    return {
        "output_dir": str(output_dir),
        "summary": str(summary_md),
        "tables": tables,
        "figures": {k: str(v) for k, v in figure_paths.items()},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate report for an explicit 2-drug pair panel positive-control run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-dir", required=True, help="Directory produced by positive_control_2drug_panel_analysis.py.")
    p.add_argument("--output-dir", default=None, help="Report directory. Default: <run-dir>/panel_report.")
    p.add_argument("--errorbar", choices=["std", "sem"], default="std", help="Error bars for mean-gain plot.")
    p.add_argument("--pair-group-col", default="pair_group", help="Column containing explicit pair cohort labels.")
    p.add_argument("--fda-group-value", default="FDA_approved", help="Value in --pair-group-col for the FDA-approved cohort.")
    p.add_argument("--failed-group-value", default="failed_trial", help="Value in --pair-group-col for the failed-trial cohort.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out = make_positive_control_panel_report(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        errorbar=args.errorbar,
        pair_group_col=args.pair_group_col,
        fda_group_value=args.fda_group_value,
        failed_group_value=args.failed_group_value,
    )
    print(f"panel report summary: {out['summary']}")


if __name__ == "__main__":
    main()
