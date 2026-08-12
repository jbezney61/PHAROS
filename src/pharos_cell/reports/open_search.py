#!/usr/bin/env python
"""
make_positive_control_search_report.py

Positive-control audit for PHAROS open-search output directories.

The report asks where a known two-drug positive-control pair appears among the
retained depth-2 search paths. It can use checkpoint.pt directly, or results.tsv
when that is preferable. Ranks and percentiles are always ranks among retained
search outputs, not all candidates ever expanded during prefiltering.

Examples
--------
pharos report open-search \
  --run-dir runs/full_2058/search \
  --drug-a Trametinib \
  --drug-b Palbociclib

Compare two runs:

pharos report open-search \
  --run-dir runs/full_2058/search \
  --run-dir runs/pls/search \
  --run-label full_2058 \
  --run-label pls \
  --drug-a Trametinib \
  --drug-b Palbociclib \
  --output-dir runs/positive_control_full_vs_pls
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


METRIC_SPECS = [
    ("score_sinkhorn_ot", "Sinkhorn OT", "rank_by_sinkhorn_ot"),
    ("score_energy_distance", "Energy distance", "rank_by_energy_distance"),
    ("adjusted_score", "Adjusted/search score", "rank_by_adjusted_score"),
]

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
    try:
        import numpy as _np
        import pandas as _pd
        import matplotlib.pyplot as _plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing report dependency. This CLI requires numpy, pandas, and matplotlib "
            "in the Python environment used to generate plots."
        ) from exc

    np = _np
    pd = _pd
    plt = _plt
    plt.rcParams.update(PUBLICATION_RC)


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(text: str, max_len: int = 80) -> str:
    out = []
    for ch in str(text):
        out.append(ch if ch.isalnum() or ch in "-_." else "_")
    return ("".join(out).strip("_") or "item")[:max_len]


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


def metric_axis_label(label: str) -> str:
    return f"{label} (lower is better)" if "Sinkhorn" in label or "Energy" in label else label


def parse_json_list(value) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if value is None or pd.isna(value):
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    if isinstance(value, str) and value.strip():
        return [x.strip() for x in value.split(" -> ") if x.strip()]
    return []


def perturbation_to_drug_name(perturbation_label: str) -> str:
    """Extract a base drug name from a Tahoe-style perturbation label."""
    s = str(perturbation_label)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, (tuple, list)) and first:
                return str(first[0])
        if isinstance(parsed, tuple) and parsed:
            return str(parsed[0])
    except Exception:
        pass
    m = re.search(r"['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)
    return s


def normalize_drug_name(value: str) -> str:
    return re.sub(r"\s+", " ", perturbation_to_drug_name(str(value)).strip()).casefold()


def compact_drug_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_drug_name(value))


def drug_name_matches(query: str, candidate: str) -> bool:
    q = compact_drug_name(query)
    c = compact_drug_name(candidate)
    return bool(q and c and (q == c or q in c or c in q))


def drug_list_contains(drug_norm_list: Sequence[str], query: str) -> bool:
    return any(drug_name_matches(query, candidate) for candidate in drug_norm_list)


def drug_order_matches(drug_norm_list: Sequence[str], expected: Sequence[str]) -> bool:
    return len(drug_norm_list) == len(expected) and all(
        drug_name_matches(query, candidate) for query, candidate in zip(expected, drug_norm_list)
    )


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
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
    for j, h in enumerate(headers):
        widths.append(max(len(h), max([len(row[j]) for row in rows], default=0)))
    header = "| " + " | ".join(h.ljust(widths[j]) for j, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[j] for j in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[j].ljust(widths[j]) for j in range(len(headers))) + " |" for row in rows]
    return "\n".join([header, sep] + body)


def resolve_artifact(run_dir: Path, name: str) -> Optional[Path]:
    candidates = [
        run_dir / name,
        run_dir / "search" / name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def infer_default_output_dir(run_dirs: Sequence[Path], drug_a: str, drug_b: str) -> Path:
    if len(run_dirs) == 1:
        return run_dirs[0] / "positive_control_search_report"
    common = Path.cwd()
    try:
        common = Path(os.path.commonpath([str(p.resolve()) for p in run_dirs]))
    except Exception:
        pass
    name = f"positive_control_search_comparison_{safe_filename(drug_a)}_{safe_filename(drug_b)}"
    return common / name


def load_results_table(results_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(results_path, sep="\t")
    if df.empty:
        raise ValueError(f"Search results are empty: {results_path}")
    meta = {"source": "results", "source_path": str(results_path), "baseline_scores": None}
    return df, meta


def load_checkpoint_table(checkpoint_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    rows = ckpt.get("results_rows", None)
    if rows:
        df = pd.DataFrame(rows)
    else:
        built_rows: List[Dict[str, Any]] = []
        for depth_key, info in (ckpt.get("depths", {}) or {}).items():
            depth = int(depth_key)
            paths = info.get("paths", []) or []
            drug_names = info.get("drug_names", []) or []
            sinkhorn = info.get("scores_sinkhorn", []) or []
            energy = info.get("scores_energy_distance", []) or []
            adjusted = info.get("adjusted_scores", []) or []
            prefilter = info.get("scores_prefilter", []) or []
            for i, path in enumerate(paths):
                drugs = drug_names[i] if i < len(drug_names) else tuple(perturbation_to_drug_name(x) for x in path)
                built_rows.append(
                    {
                        "depth": depth,
                        "rank": i + 1,
                        "num_drugs": len(path),
                        "path_json": json.dumps(list(path), ensure_ascii=False),
                        "drug_names_json": json.dumps(list(drugs), ensure_ascii=False),
                        "path_string": " -> ".join(map(str, path)),
                        "drug_name_string": " -> ".join(map(str, drugs)),
                        "score_sinkhorn_ot": float(sinkhorn[i]) if i < len(sinkhorn) else np.nan,
                        "score_energy_distance": float(energy[i]) if i < len(energy) else np.nan,
                        "score_prefilter": float(prefilter[i]) if i < len(prefilter) else np.nan,
                        "adjusted_score": float(adjusted[i]) if i < len(adjusted) else np.nan,
                    }
                )
        df = pd.DataFrame(built_rows)

    if df.empty:
        raise ValueError(f"Checkpoint has no retained search rows: {checkpoint_path}")

    meta = {
        "source": "checkpoint",
        "source_path": str(checkpoint_path),
        "baseline_scores": ckpt.get("baseline_scores", None),
        "projection_metadata": ckpt.get("projection_metadata", None),
    }
    return df, meta


def load_run_table(run_dir: Path, source: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    checkpoint_path = resolve_artifact(run_dir, "checkpoint.pt")
    results_path = resolve_artifact(run_dir, "results.tsv")

    if source == "checkpoint":
        if checkpoint_path is None:
            raise FileNotFoundError(f"Could not find checkpoint.pt in {run_dir} or {run_dir / 'search'}")
        return load_checkpoint_table(checkpoint_path)

    if source == "results":
        if results_path is None:
            raise FileNotFoundError(f"Could not find results.tsv in {run_dir} or {run_dir / 'search'}")
        return load_results_table(results_path)

    if checkpoint_path is not None:
        return load_checkpoint_table(checkpoint_path)
    if results_path is not None:
        return load_results_table(results_path)
    raise FileNotFoundError(f"Could not find checkpoint.pt or results.tsv in {run_dir} or {run_dir / 'search'}")


def prepare_search_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "path_json" in out.columns:
        out["path_list"] = out["path_json"].apply(parse_json_list)
    elif "path_string" in out.columns:
        out["path_list"] = out["path_string"].apply(parse_json_list)
    else:
        out["path_list"] = [[] for _ in range(len(out))]

    if "drug_names_json" in out.columns:
        out["drug_name_list"] = out["drug_names_json"].apply(parse_json_list)
    elif "drug_name_string" in out.columns:
        out["drug_name_list"] = out["drug_name_string"].apply(parse_json_list)
    else:
        out["drug_name_list"] = out["path_list"].apply(lambda xs: [perturbation_to_drug_name(x) for x in xs])

    out["drug_name_list"] = out.apply(
        lambda row: row["drug_name_list"]
        if len(row["drug_name_list"])
        else [perturbation_to_drug_name(x) for x in row["path_list"]],
        axis=1,
    )
    out["drug_norm_list"] = out["drug_name_list"].apply(lambda xs: [normalize_drug_name(x) for x in xs])
    out["drug_name_string_pretty"] = out["drug_name_list"].apply(lambda xs: " -> ".join(xs))
    out["path_string_pretty"] = out["path_list"].apply(lambda xs: " -> ".join(xs))

    for col in ["depth", "rank", "score_sinkhorn_ot", "score_energy_distance", "adjusted_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "rank" not in out.columns:
        out["rank"] = np.nan
    return out


def add_depth_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_by_search"] = np.nan
    for depth, idx in out.groupby("depth").groups.items():
        sub = out.loc[idx].copy()
        if sub["rank"].notna().any():
            ordered_idx = sub.sort_values(["rank"], kind="mergesort").index
        else:
            sort_col = "adjusted_score" if "adjusted_score" in sub and sub["adjusted_score"].notna().any() else "score_sinkhorn_ot"
            ordered_idx = sub.sort_values([sort_col], kind="mergesort").index
        out.loc[ordered_idx, "rank_by_search"] = np.arange(1, len(ordered_idx) + 1)

    for metric_col, _, rank_col in METRIC_SPECS:
        if metric_col not in out.columns:
            continue
        out[rank_col] = np.nan
        for depth, idx in out.groupby("depth").groups.items():
            sub = out.loc[idx]
            finite = sub[np.isfinite(pd.to_numeric(sub[metric_col], errors="coerce"))]
            if finite.empty:
                continue
            ordered_idx = finite.sort_values([metric_col], kind="mergesort").index
            out.loc[ordered_idx, rank_col] = np.arange(1, len(ordered_idx) + 1)

    return out


def add_positive_control_flags(df: pd.DataFrame, drug_a: str, drug_b: str, target_depth: int) -> pd.DataFrame:
    a = normalize_drug_name(drug_a)
    b = normalize_drug_name(drug_b)
    out = df.copy()

    out["contains_drug_a"] = out["drug_norm_list"].apply(lambda xs: drug_list_contains(xs, a))
    out["contains_drug_b"] = out["drug_norm_list"].apply(lambda xs: drug_list_contains(xs, b))
    out["contains_both"] = out["contains_drug_a"] & out["contains_drug_b"]
    out["exact_order_a_then_b"] = out["drug_norm_list"].apply(lambda xs: drug_order_matches(xs, [a, b]))
    out["exact_order_b_then_a"] = out["drug_norm_list"].apply(lambda xs: drug_order_matches(xs, [b, a]))
    out["unordered_exact_pair"] = out["drug_norm_list"].apply(
        lambda xs: len(xs) == target_depth and drug_list_contains(xs, a) and drug_list_contains(xs, b)
    )
    out["target_depth"] = out["depth"].eq(target_depth)
    return out


def depth_n(df: pd.DataFrame, depth: int) -> int:
    return int(df[df["depth"].eq(depth)].shape[0])


def best_match_row(df: pd.DataFrame, mask: pd.Series, rank_col: str = "rank_by_search") -> Optional[pd.Series]:
    sub = df[mask].copy()
    if sub.empty:
        return None
    sub = sub.sort_values([rank_col, "score_sinkhorn_ot"], na_position="last", kind="mergesort")
    return sub.iloc[0]


def metric_value(row: Optional[pd.Series], col: str) -> float:
    if row is None or col not in row or pd.isna(row[col]):
        return np.nan
    return float(row[col])


def rank_percent(rank: float, n: int) -> float:
    if not np.isfinite(rank) or n <= 0:
        return np.nan
    return 100.0 * float(rank) / float(n)


def summarize_match(
    run_label: str,
    match_type: str,
    row: Optional[pd.Series],
    n_at_depth: int,
    target_depth: int,
) -> Dict[str, Any]:
    found = row is not None
    out = {
        "run": run_label,
        "match_type": match_type,
        "found": bool(found),
        "target_depth": int(target_depth),
        "retained_paths_at_depth": int(n_at_depth),
        "rank_lower_bound_if_absent": f">{n_at_depth}" if not found else "",
    }
    if not found:
        for col in [
            "rank_by_search",
            "rank_by_sinkhorn_ot",
            "rank_by_energy_distance",
            "rank_by_adjusted_score",
            "score_sinkhorn_ot",
            "score_energy_distance",
            "adjusted_score",
            "drug_name_string",
            "path_string",
        ]:
            out[col] = np.nan if col.startswith("rank") or col.startswith("score") or col == "adjusted_score" else ""
        out["search_percentile_lower_is_better"] = np.nan
        return out

    out.update(
        {
            "depth": int(row["depth"]) if not pd.isna(row["depth"]) else np.nan,
            "rank_by_search": metric_value(row, "rank_by_search"),
            "rank_by_sinkhorn_ot": metric_value(row, "rank_by_sinkhorn_ot"),
            "rank_by_energy_distance": metric_value(row, "rank_by_energy_distance"),
            "rank_by_adjusted_score": metric_value(row, "rank_by_adjusted_score"),
            "search_percentile_lower_is_better": rank_percent(metric_value(row, "rank_by_search"), n_at_depth),
            "sinkhorn_percentile_lower_is_better": rank_percent(metric_value(row, "rank_by_sinkhorn_ot"), n_at_depth),
            "energy_percentile_lower_is_better": rank_percent(metric_value(row, "rank_by_energy_distance"), n_at_depth),
            "score_sinkhorn_ot": metric_value(row, "score_sinkhorn_ot"),
            "score_energy_distance": metric_value(row, "score_energy_distance"),
            "adjusted_score": metric_value(row, "adjusted_score"),
            "drug_name_string": row.get("drug_name_string_pretty", row.get("drug_name_string", "")),
            "path_string": row.get("path_string_pretty", row.get("path_string", "")),
        }
    )
    return out


def build_hit_summary(df: pd.DataFrame, run_label: str, target_depth: int) -> pd.DataFrame:
    n2 = depth_n(df, target_depth)
    n1 = depth_n(df, 1)
    target = df["depth"].eq(target_depth)
    depth1 = df["depth"].eq(1)

    match_defs = [
        ("contains_both", target & df["contains_both"], n2, target_depth),
        ("unordered_exact_pair", target & df["unordered_exact_pair"], n2, target_depth),
        ("exact_order_a_then_b", target & df["exact_order_a_then_b"], n2, target_depth),
        ("exact_order_b_then_a", target & df["exact_order_b_then_a"], n2, target_depth),
        ("best_contains_drug_a", target & df["contains_drug_a"], n2, target_depth),
        ("best_contains_drug_b", target & df["contains_drug_b"], n2, target_depth),
        ("drug_a_depth1", depth1 & df["contains_drug_a"], n1, 1),
        ("drug_b_depth1", depth1 & df["contains_drug_b"], n1, 1),
    ]
    rows = [
        summarize_match(run_label, name, best_match_row(df, mask), n_depth, depth)
        for name, mask, n_depth, depth in match_defs
    ]
    return pd.DataFrame(rows)


def add_score_gaps(df: pd.DataFrame, target_depth: int) -> pd.DataFrame:
    out = df.copy()
    for col in ["score_sinkhorn_ot", "score_energy_distance", "adjusted_score"]:
        if col not in out.columns:
            continue
        best = out.loc[out["depth"].eq(target_depth), col].min(skipna=True)
        if pd.isna(best):
            continue
        out[f"{col}_gap_from_best_at_depth"] = out[col].astype(float) - float(best)
    return out


def baseline_scores_from_df(df: pd.DataFrame, meta: Dict[str, Any]) -> Dict[str, float]:
    baseline = meta.get("baseline_scores") or {}
    out = {
        "sinkhorn": float(baseline["sinkhorn"]) if baseline and "sinkhorn" in baseline else np.nan,
        "energy_distance": float(baseline["energy_distance"]) if baseline and "energy_distance" in baseline else np.nan,
    }
    depth0 = df[df["depth"].eq(0)]
    if not depth0.empty:
        row = depth0.sort_values("rank_by_search").iloc[0]
        if np.isnan(out["sinkhorn"]) and "score_sinkhorn_ot" in row:
            out["sinkhorn"] = metric_value(row, "score_sinkhorn_ot")
        if np.isnan(out["energy_distance"]) and "score_energy_distance" in row:
            out["energy_distance"] = metric_value(row, "score_energy_distance")
    return out


def plot_ranked_metric(
    depth_df: pd.DataFrame,
    metric_col: str,
    metric_label: str,
    rank_col: str,
    fig_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(6.8, 4.6))
    ax = plt.gca()
    if depth_df.empty or metric_col not in depth_df.columns:
        plt.text(0.5, 0.5, f"No {metric_label} data", ha="center", va="center")
        plt.axis("off")
        savefig(fig_path)
        return

    d = depth_df[np.isfinite(pd.to_numeric(depth_df[metric_col], errors="coerce"))].copy()
    d = d.sort_values(rank_col)
    ax.plot(d[rank_col], d[metric_col], color="#8a8f98", linewidth=1.4)
    ax.scatter(d[rank_col], d[metric_col], s=18, color="#737982", alpha=0.65)

    highlights = [
        ("contains A", d["contains_drug_a"], "#2ca25f"),
        ("contains B", d["contains_drug_b"], "#1f77b4"),
        ("contains both", d["contains_both"], "#d62728"),
    ]
    for label, mask, color in highlights:
        h = d[mask]
        if not h.empty:
            marker = "*" if label == "contains both" else "o"
            size = 120 if label == "contains both" else 50
            zorder = 3 if label == "contains both" else 2
            ax.scatter(
                h[rank_col],
                h[metric_col],
                s=size,
                color=color,
                alpha=0.92,
                marker=marker,
                label=label,
                zorder=zorder,
            )

    ax.set_xlabel(f"{metric_label} rank among retained depth-2 paths")
    ax.set_ylabel(metric_axis_label(metric_label))
    ax.set_title(title)
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    if ax.get_legend_handles_labels()[0]:
        make_legend_opaque(ax, loc="best")
    polish_axes(ax)
    savefig(fig_path)


def plot_metric_scatter(depth_df: pd.DataFrame, fig_path: Path, title: str) -> None:
    plt.figure(figsize=(5.8, 5.0))
    ax = plt.gca()
    required = {"score_sinkhorn_ot", "score_energy_distance"}
    if depth_df.empty or not required.issubset(depth_df.columns):
        plt.text(0.5, 0.5, "No paired Sinkhorn/energy data", ha="center", va="center")
        plt.axis("off")
        savefig(fig_path)
        return

    d = depth_df.dropna(subset=["score_sinkhorn_ot", "score_energy_distance"]).copy()
    ax.scatter(d["score_sinkhorn_ot"], d["score_energy_distance"], s=24, color="#737982", alpha=0.58, label="retained paths")

    highlights = [
        ("contains A", d["contains_drug_a"], "#2ca25f"),
        ("contains B", d["contains_drug_b"], "#1f77b4"),
        ("contains both", d["contains_both"], "#d62728"),
    ]
    for label, mask, color in highlights:
        h = d[mask]
        if not h.empty:
            marker = "*" if label == "contains both" else "o"
            size = 140 if label == "contains both" else 58
            ax.scatter(
                h["score_sinkhorn_ot"],
                h["score_energy_distance"],
                s=size,
                color=color,
                marker=marker,
                alpha=0.92,
                label=label,
                edgecolor="white",
                linewidth=0.45,
            )

    ax.set_xlabel("Sinkhorn OT (lower is better)")
    ax.set_ylabel("Energy distance (lower is better)")
    ax.set_title(title)
    ax.grid(color="#e5e5e5", lw=0.7)
    make_legend_opaque(ax, loc="best")
    polish_axes(ax)
    savefig(fig_path)


def plot_match_score_bars(hit_df: pd.DataFrame, fig_path: Path, title: str) -> None:
    keep = [
        "contains_both",
        "drug_a_depth1",
        "drug_b_depth1",
    ]
    d = hit_df[hit_df["match_type"].isin(keep)].copy()
    labels = {
        "contains_both": "contains both",
        "drug_a_depth1": "A only",
        "drug_b_depth1": "B only",
    }
    d["label"] = d["match_type"].map(labels)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    for ax, metric_col, metric_label, color in [
        (axes[0], "score_sinkhorn_ot", "Sinkhorn OT", "#1f77b4"),
        (axes[1], "score_energy_distance", "Energy distance", "#ff7f0e"),
    ]:
        found = d[d["found"] & np.isfinite(pd.to_numeric(d[metric_col], errors="coerce"))].copy()
        if found.empty:
            ax.text(0.5, 0.5, f"No {metric_label} matches found", ha="center", va="center")
            ax.axis("off")
            continue
        ax.bar(found["label"], found[metric_col].astype(float), color=color, edgecolor="#333333", linewidth=0.7)
        ax.set_ylabel(metric_axis_label(metric_label))
        ax.set_title(metric_label)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        polish_axes(ax)
    fig.suptitle(title)
    savefig(fig_path)


def plot_prefix_scores(
    df: pd.DataFrame,
    hit_df: pd.DataFrame,
    baseline: Dict[str, float],
    fig_path: Path,
    title: str,
) -> None:
    def lookup(match_type: str, metric_col: str) -> float:
        sub = hit_df[(hit_df["match_type"] == match_type) & (hit_df["found"])]
        if sub.empty or metric_col not in sub:
            return np.nan
        return float(sub.iloc[0][metric_col])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=True)
    specs = [
        (axes[0], "score_sinkhorn_ot", "Sinkhorn OT", baseline.get("sinkhorn", np.nan)),
        (axes[1], "score_energy_distance", "Energy distance", baseline.get("energy_distance", np.nan)),
    ]
    for ax, metric_col, metric_label, base in specs:
        a_vals = [base, lookup("drug_a_depth1", metric_col), lookup("contains_both", metric_col)]
        b_vals = [base, lookup("drug_b_depth1", metric_col), lookup("contains_both", metric_col)]
        plotted = False
        if np.isfinite(a_vals).any():
            ax.plot([0, 1, 2], a_vals, marker="o", label="A then pair", color="#d62728", lw=2.4, ms=6)
            plotted = True
        if np.isfinite(b_vals).any():
            ax.plot([0, 1, 2], b_vals, marker="o", label="B then pair", color="#9467bd", lw=2.4, ms=6)
            plotted = True
        if not plotted:
            ax.text(0.5, 0.5, f"No {metric_label} prefix data", ha="center", va="center")
        ax.set_xticks([0, 1, 2], ["baseline", "one drug", "two drugs"])
        ax.set_ylabel(metric_axis_label(metric_label))
        ax.set_title(metric_label)
        ax.grid(axis="y", color="#e5e5e5", lw=0.8)
        polish_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            make_legend_opaque(ax, handles=handles, labels=labels, loc="best")
    fig.suptitle(title)
    savefig(fig_path)


def plot_depth2_top_table_image(depth_df: pd.DataFrame, fig_path: Path, top_n: int, title: str) -> None:
    cols = [
        "rank_by_search",
        "drug_name_string_pretty",
        "score_sinkhorn_ot",
        "score_energy_distance",
        "contains_drug_a",
        "contains_drug_b",
        "contains_both",
    ]
    cols = [c for c in cols if c in depth_df.columns]
    d = depth_df.sort_values("rank_by_search").head(top_n)[cols].copy()
    if d.empty:
        plt.figure(figsize=(6.8, 3.4))
        plt.text(0.5, 0.5, "No depth-2 retained paths", ha="center", va="center")
        plt.axis("off")
        savefig(fig_path)
        return
    d = d.rename(
        columns={
            "rank_by_search": "rank",
            "drug_name_string_pretty": "path",
            "score_sinkhorn_ot": "sinkhorn",
            "score_energy_distance": "energy",
            "contains_drug_a": "contains A",
            "contains_drug_b": "contains B",
            "contains_both": "A+B",
        }
    )
    for col in ["sinkhorn", "energy"]:
        if col in d:
            d[col] = d[col].map(lambda x: f"{x:.5g}" if pd.notna(x) else "")
    for col in ["contains A", "contains B", "A+B"]:
        if col in d:
            d[col] = d[col].map(lambda x: "yes" if bool(x) else "")

    fig_h = max(3.5, 0.38 * len(d) + 1.4)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    ax.axis("off")
    table = ax.table(cellText=d.values, colLabels=d.columns, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title(title, pad=12)
    savefig(fig_path)


def write_run_summary(
    report_dir: Path,
    run_dir: Path,
    run_label: str,
    source_meta: Dict[str, Any],
    hit_df: pd.DataFrame,
    depth2_df: pd.DataFrame,
    target_depth: int,
    drug_a: str,
    drug_b: str,
) -> None:
    pair = hit_df[hit_df["match_type"].eq("contains_both")]
    pair_row = pair.iloc[0] if not pair.empty else None
    lines = [
        "# Positive-Control Search Report\n",
        f"Run: `{run_label}`",
        f"Run directory: `{run_dir}`",
        f"Source: `{source_meta.get('source')}` from `{source_meta.get('source_path')}`",
        f"Known pair: `{drug_a}` + `{drug_b}` (order-insensitive)",
        f"Target depth: `{target_depth}`",
        f"Retained depth-{target_depth} paths: `{len(depth2_df)}`",
        "",
    ]
    if pair_row is not None and bool(pair_row["found"]):
        lines.append(
            "Both drugs were found together at "
            f"search rank `{int(pair_row['rank_by_search'])}` "
            f"({float(pair_row['search_percentile_lower_is_better']):.3g}% lower-is-better percentile)."
        )
    else:
        lines.append(
            f"No retained depth-{target_depth} path contained both drugs among the `{len(depth2_df)}` saved paths."
        )
    lines.extend(
        [
            "",
            "Ranks and percentiles are computed among retained search outputs at the same depth. "
            "If the pair is absent, the true rank is only known to be worse than the retained count unless candidate-pool logging was enabled during search.",
            "",
            "## Hit Summary\n",
            markdown_table(hit_df, max_rows=20),
            "",
            "## Top Retained Depth-2 Paths\n",
            markdown_table(
                depth2_df.sort_values("rank_by_search")[
                    [
                        c
                        for c in [
                            "rank_by_search",
                            "drug_name_string_pretty",
                            "score_sinkhorn_ot",
                            "score_energy_distance",
                            "adjusted_score",
                            "contains_drug_a",
                            "contains_drug_b",
                            "contains_both",
                            "exact_order_a_then_b",
                            "exact_order_b_then_a",
                        ]
                        if c in depth2_df.columns
                    ]
                ],
                max_rows=25,
            ),
            "",
            "## Figures\n",
            f"- Figure 1: depth-{target_depth} Sinkhorn rank curve",
            f"- Figure 2: depth-{target_depth} energy rank curve",
            "- Figure 3: Sinkhorn vs energy scatter",
            "- Figure 4: positive-control score bars",
            "- Figure 5: baseline/prefix trajectory",
            "- Figure 6: top retained depth-2 table snapshot",
        ]
    )
    (report_dir / "summary.md").write_text("\n".join(lines))


def process_run(
    run_dir: Path,
    run_label: str,
    report_dir: Path,
    drug_a: str,
    drug_b: str,
    target_depth: int,
    source: str,
    top_n_table: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    df_raw, meta = load_run_table(run_dir, source)
    df = prepare_search_df(df_raw)
    df = add_depth_ranks(df)
    df = add_positive_control_flags(df, drug_a, drug_b, target_depth)
    df = add_score_gaps(df, target_depth)
    df["run"] = run_label

    hit_df = build_hit_summary(df, run_label, target_depth)
    depth_df = df[df["depth"].eq(target_depth)].copy().sort_values("rank_by_search")

    table_dir = safe_mkdir(report_dir / "tables")
    fig_dir = safe_mkdir(report_dir / "figures")
    df.to_csv(table_dir / "all_retained_paths_with_positive_control_flags.tsv", sep="\t", index=False)
    depth_df.to_csv(table_dir / f"depth_{target_depth}_retained_paths.tsv", sep="\t", index=False)
    hit_df.to_csv(table_dir / "positive_control_hit_summary.tsv", sep="\t", index=False)

    plot_ranked_metric(
        depth_df,
        "score_sinkhorn_ot",
        "Sinkhorn OT",
        "rank_by_sinkhorn_ot",
        fig_dir / "01_depth2_sinkhorn_rank_curve.png",
        f"{run_label}: retained depth-{target_depth} paths by Sinkhorn",
    )
    plot_ranked_metric(
        depth_df,
        "score_energy_distance",
        "Energy distance",
        "rank_by_energy_distance",
        fig_dir / "02_depth2_energy_rank_curve.png",
        f"{run_label}: retained depth-{target_depth} paths by energy",
    )
    plot_metric_scatter(
        depth_df,
        fig_dir / "03_depth2_sinkhorn_vs_energy.png",
        f"{run_label}: Sinkhorn vs energy for retained depth-{target_depth} paths",
    )
    plot_match_score_bars(
        hit_df,
        fig_dir / "04_positive_control_match_scores.png",
        f"{run_label}: positive-control match scores",
    )
    plot_prefix_scores(
        df,
        hit_df,
        baseline_scores_from_df(df, meta),
        fig_dir / "05_positive_control_prefix_scores.png",
        f"{run_label}: baseline and positive-control prefixes",
    )
    plot_depth2_top_table_image(
        depth_df,
        fig_dir / "06_top_depth2_paths_table.png",
        top_n_table,
        f"{run_label}: top retained depth-{target_depth} paths",
    )

    write_run_summary(
        report_dir=report_dir,
        run_dir=run_dir,
        run_label=run_label,
        source_meta=meta,
        hit_df=hit_df,
        depth2_df=depth_df,
        target_depth=target_depth,
        drug_a=drug_a,
        drug_b=drug_b,
    )
    return df, hit_df, meta


def plot_comparison_bars(
    comparison_df: pd.DataFrame,
    fig_dir: Path,
    metric_col: str,
    ylabel: str,
    title: str,
    fig_name: str,
    absent_as_retained_plus_one: bool = False,
) -> None:
    d = comparison_df.copy()
    plt.figure(figsize=(max(5.8, 0.72 * len(d) + 1.6), 4.4))
    ax = plt.gca()
    vals = pd.to_numeric(d[metric_col], errors="coerce")
    labels = d["run"].astype(str).tolist()
    if absent_as_retained_plus_one:
        replacement = pd.to_numeric(d["retained_paths_at_depth"], errors="coerce") + 1
        vals = vals.fillna(replacement)
        bar_labels = [
            f">{int(n)}" if not bool(found) else f"{int(v)}"
            for v, n, found in zip(vals, d["retained_paths_at_depth"], d["found"])
        ]
    else:
        bar_labels = [f"{v:.4g}" if np.isfinite(v) else "absent" for v in vals]
    colors = ["#2f6fed" if bool(x) else "#b8bcc4" for x in d["found"]]
    ax.bar(labels, vals, color=colors, edgecolor="#333333", linewidth=0.7)
    for i, (v, label) in enumerate(zip(vals, bar_labels)):
        if np.isfinite(v):
            ax.text(i, v, label, ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(metric_axis_label(ylabel))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    polish_axes(ax)
    savefig(fig_dir / fig_name)


def write_comparison_report(
    output_dir: Path,
    all_hits: pd.DataFrame,
    target_depth: int,
    drug_a: str,
    drug_b: str,
) -> None:
    table_dir = safe_mkdir(output_dir / "tables")
    fig_dir = safe_mkdir(output_dir / "figures")
    all_hits.to_csv(table_dir / "all_run_positive_control_hit_summary.tsv", sep="\t", index=False)

    pair = all_hits[all_hits["match_type"].eq("contains_both")].copy()
    pair.to_csv(table_dir / "comparison_contains_both.tsv", sep="\t", index=False)

    if not pair.empty:
        plot_comparison_bars(
            pair,
            fig_dir,
            "rank_by_search",
            "Search rank among retained paths",
            f"Pair rank: {drug_a} + {drug_b}",
            "comparison_01_contains_both_search_rank.png",
            absent_as_retained_plus_one=True,
        )
        plot_comparison_bars(
            pair,
            fig_dir,
            "search_percentile_lower_is_better",
            "Lower-is-better percentile",
            f"Pair percentile: {drug_a} + {drug_b}",
            "comparison_02_contains_both_percentile.png",
        )
        plot_comparison_bars(
            pair,
            fig_dir,
            "score_sinkhorn_ot",
            "Sinkhorn OT",
            f"Pair Sinkhorn OT: {drug_a} + {drug_b}",
            "comparison_03_contains_both_sinkhorn.png",
        )
        plot_comparison_bars(
            pair,
            fig_dir,
            "score_energy_distance",
            "Energy distance",
            f"Pair energy distance: {drug_a} + {drug_b}",
            "comparison_04_contains_both_energy.png",
        )

    lines = [
        "# Positive-Control Search Comparison\n",
        f"Known pair: `{drug_a}` + `{drug_b}` (order-insensitive)",
        f"Target depth: `{target_depth}`",
        "",
        "Ranks and percentiles are among retained depth-matched paths for each run.",
        "",
        "## Contains Both Drugs\n",
        markdown_table(
            pair[
                [
                    "run",
                    "found",
                    "rank_by_search",
                    "rank_lower_bound_if_absent",
                    "search_percentile_lower_is_better",
                    "score_sinkhorn_ot",
                    "score_energy_distance",
                    "retained_paths_at_depth",
                    "drug_name_string",
                ]
            ]
            if not pair.empty
            else pair,
            max_rows=50,
        ),
        "",
        "## All Hit Types\n",
        markdown_table(all_hits, max_rows=80),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pharos report open-search",
        description="Generate positive-control search audit plots from cell_converter/search output directories."
    )
    p.add_argument("--run-dir", action="append", required=True, help="Search output directory. Repeat for comparisons.")
    p.add_argument("--run-label", action="append", default=None, help="Optional label for each --run-dir.")
    p.add_argument("--drug-a", required=True, help="One drug in the expected pair.")
    p.add_argument("--drug-b", required=True, help="The other drug in the expected pair.")
    p.add_argument("--target-depth", type=int, default=2, help="Depth to audit. Positive controls usually use depth 2.")
    p.add_argument(
        "--source",
        choices=["auto", "checkpoint", "results"],
        default="auto",
        help="Input source. auto prefers checkpoint.pt and falls back to results.tsv.",
    )
    p.add_argument("--output-dir", default=None, help="Report output directory.")
    p.add_argument("--top-n-table", type=int, default=25, help="Number of top paths to render in the table figure.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    ensure_report_dependencies()
    run_dirs = [Path(x) for x in args.run_dir]
    labels = args.run_label or []
    if labels and len(labels) != len(run_dirs):
        raise ValueError("--run-label must be provided the same number of times as --run-dir")
    if not labels:
        labels = [p.name if p.name != "search" else p.parent.name for p in run_dirs]

    output_dir = Path(args.output_dir) if args.output_dir else infer_default_output_dir(run_dirs, args.drug_a, args.drug_b)
    safe_mkdir(output_dir)

    all_hit_tables = []
    for run_dir, label in zip(run_dirs, labels):
        report_dir = output_dir if len(run_dirs) == 1 else output_dir / "runs" / safe_filename(label)
        safe_mkdir(report_dir)
        _, hit_df, _ = process_run(
            run_dir=run_dir,
            run_label=label,
            report_dir=report_dir,
            drug_a=args.drug_a,
            drug_b=args.drug_b,
            target_depth=args.target_depth,
            source=args.source,
            top_n_table=args.top_n_table,
        )
        all_hit_tables.append(hit_df)

    all_hits = pd.concat(all_hit_tables, ignore_index=True) if all_hit_tables else pd.DataFrame()
    if len(run_dirs) > 1:
        write_comparison_report(
            output_dir=output_dir,
            all_hits=all_hits,
            target_depth=args.target_depth,
            drug_a=args.drug_a,
            drug_b=args.drug_b,
        )

    print("\n=== Positive-control search report complete ===")
    print(f"report: {output_dir / 'summary.md'}")
    print(f"tables: {output_dir / 'tables' if len(run_dirs) > 1 else output_dir / 'tables'}")
    print(f"figures: {output_dir / 'figures' if len(run_dirs) > 1 else output_dir / 'figures'}")


if __name__ == "__main__":
    main()
