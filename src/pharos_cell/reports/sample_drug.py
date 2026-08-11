#!/usr/bin/env python
"""
make_sample_drug_report.py

Sample-specific and drug-specific report for PHAROS/ST-SE conversion searches.

Inputs:
  - A PHAROS/cell_converter output directory, or its search/ subdirectory
  - metadata/cell_line_metadata.csv
  - metadata/drug_metadata.csv

Outputs:
  <run-dir>/sample_drug_report/
    summary.md
    tables/
      starting_cell_drivers.tsv
      top_path_drug_annotations.tsv
      driver_target_frequency.tsv
      driver_target_by_step.tsv
      target_frequency.tsv
      moa_frequency.tsv
      moa_by_step.tsv
      moa_enrichment.tsv
    figures/
      01_driver_target_heatmap.png
      02_top_targets_barplot.png
      03_moa_frequency_barplot.png
      04_moa_enrichment_volcano.png
      05_moa_by_step_heatmap.png
      06_driver_target_path_matrix.png

Example:
  python make_sample_drug_report.py \
    --run-dir runs/J82_to_A172 \
    --metadata-dir metadata \
    --top-n-paths 50
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
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


def add_colorbar(im: Any, label: str) -> None:
    cbar = plt.colorbar(im)
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=11)


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


def perturbation_to_drug_name(perturbation_label: str) -> str:
    """Extract base drug name from a Tahoe label like "[('Trametinib', 0.05, 'uM')]"."""
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


def parse_targets(value) -> List[str]:
    """Parse target gene list from empty, comma/semicolon/pipe separated, or list-like cells."""
    if pd.isna(value):
        return []
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "unclear", "unknown"}:
        return []
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple, set)):
            return sorted({str(x).strip() for x in parsed if str(x).strip()})
    except Exception:
        pass
    s = s.replace(";", ",").replace("|", ",")
    return sorted({p.strip() for p in s.split(",") if p.strip()})


def normalize_gene_symbol(x: str) -> str:
    return str(x).strip().upper()


def normalize_drug_name(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip()).lower()


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Render a small markdown table without pandas.to_markdown/tabulate."""
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


def resolve_search_results(run_dir: Path) -> Path:
    if (run_dir / "results.tsv").exists():
        return run_dir / "results.tsv"
    if (run_dir / "search" / "results.tsv").exists():
        return run_dir / "search" / "results.tsv"
    raise FileNotFoundError(f"Could not find results.tsv in {run_dir} or {run_dir / 'search'}")


def infer_top_level_run_dir(run_dir: Path) -> Path:
    if run_dir.name == "search" and (run_dir / "results.tsv").exists():
        return run_dir.parent
    return run_dir


def load_results(results_path: Path) -> pd.DataFrame:
    df = pd.read_csv(results_path, sep="\t")
    if df.empty:
        raise ValueError(f"Search results are empty: {results_path}")
    df = df.copy()
    if "path_json" in df.columns:
        df["path_list"] = df["path_json"].apply(parse_json_list)
    else:
        df["path_list"] = df.get("path_string", pd.Series([""] * len(df))).apply(parse_json_list)
    if "drug_names_json" in df.columns:
        df["drug_name_list"] = df["drug_names_json"].apply(parse_json_list)
    elif "drug_name_string" in df.columns:
        df["drug_name_list"] = df["drug_name_string"].apply(parse_json_list)
    else:
        df["drug_name_list"] = df["path_list"].apply(lambda xs: [perturbation_to_drug_name(x) for x in xs])
    df["parsed_base_drug_list"] = df["path_list"].apply(lambda xs: [perturbation_to_drug_name(x) for x in xs])
    df["drug_name_list"] = df.apply(
        lambda r: r["drug_name_list"] if len(r["drug_name_list"]) else r["parsed_base_drug_list"], axis=1
    )
    return df


def selection_sort_column(df: pd.DataFrame) -> str:
    if "adjusted_score" in df.columns:
        vals = pd.to_numeric(df["adjusted_score"], errors="coerce")
        if vals.notna().any():
            return "adjusted_score"
    return "score_sinkhorn_ot"


def load_cell_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"cell_name", "Organ", "Driver_Gene_Symbol"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cell_line_metadata.csv missing required columns: {missing}")
    return df


def load_drug_metadata(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"drug", "targets", "moa-fine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"drug_metadata.csv missing required columns: {missing}")
    df = df.copy()
    df["drug_norm"] = df["drug"].apply(normalize_drug_name)
    df["target_list"] = df["targets"].apply(parse_targets)
    df["target_list_norm"] = df["target_list"].apply(lambda xs: [normalize_gene_symbol(x) for x in xs])
    df["moa_fine_clean"] = df["moa-fine"].fillna("unknown").astype(str).str.strip()
    bad = df["moa_fine_clean"].eq("") | df["moa_fine_clean"].str.lower().isin(["nan", "none", "null"])
    df.loc[bad, "moa_fine_clean"] = "unknown"
    return df


def extract_start_cell(df: pd.DataFrame, cli_start_cell: Optional[str]) -> str:
    if cli_start_cell is not None:
        return str(cli_start_cell)
    if "start_cell" in df.columns:
        vals = [str(x) for x in df["start_cell"].dropna().unique()]
        if vals:
            return vals[0]
    raise ValueError("Could not infer starting cell from results.tsv. Pass --start-cell explicitly.")


def get_starting_cell_drivers(cell_meta: pd.DataFrame, start_cell: str) -> Tuple[str, List[str], pd.DataFrame]:
    sub = cell_meta[cell_meta["cell_name"].astype(str) == str(start_cell)].copy()
    if sub.empty:
        raise ValueError(f"No rows found for start-cell={start_cell!r} in cell_line_metadata.csv")
    organ = "; ".join(sorted(set(sub["Organ"].dropna().astype(str))))
    drivers = sorted({normalize_gene_symbol(x) for x in sub["Driver_Gene_Symbol"].dropna().astype(str) if str(x).strip()})
    keep_cols = [
        "cell_name", "Organ", "Driver_Gene_Symbol", "Cell_ID_DepMap", "Cell_ID_Cellosaur",
        "Driver_VarZyg", "Driver_VarType", "Driver_ProtEffect_or_CdnaEffect",
        "Driver_Mech_InferDM", "Driver_GeneType_DM",
    ]
    keep_cols = [c for c in keep_cols if c in sub.columns]
    return organ, drivers, sub[keep_cols]


def build_drug_lookup(drug_meta: pd.DataFrame) -> Dict[str, pd.Series]:
    lookup = {}
    for _, row in drug_meta.iterrows():
        key = normalize_drug_name(row["drug"])
        if key not in lookup:
            lookup[key] = row
    return lookup


def annotate_top_paths(results: pd.DataFrame, drug_meta: pd.DataFrame, top_n_paths: int, final_depth_only: bool) -> pd.DataFrame:
    d = results.copy()
    if final_depth_only:
        d = d[d["depth"] == d["depth"].max()].copy()
    d = d.sort_values(selection_sort_column(d)).head(top_n_paths).copy()
    lookup = build_drug_lookup(drug_meta)
    rows = []
    for path_rank, (_, row) in enumerate(d.iterrows(), start=1):
        path_list = row["path_list"]
        base_drugs = row["drug_name_list"]
        if len(path_list) != len(base_drugs):
            base_drugs = [perturbation_to_drug_name(x) for x in path_list]
        for step, (pert_label, base_drug) in enumerate(zip(path_list, base_drugs), start=1):
            meta = lookup.get(normalize_drug_name(base_drug))
            if meta is None:
                targets, targets_norm, moa, found = [], [], "metadata_missing", False
            else:
                targets = list(meta["target_list"])
                targets_norm = list(meta["target_list_norm"])
                moa = str(meta["moa_fine_clean"])
                found = True
            rows.append({
                "path_rank_by_score": path_rank,
                "results_depth": int(row["depth"]),
                "results_rank": int(row["rank"]) if "rank" in row and not pd.isna(row["rank"]) else path_rank,
                "score_sinkhorn_ot": float(row["score_sinkhorn_ot"]),
                "score_energy_distance": float(row["score_energy_distance"]) if "score_energy_distance" in row else np.nan,
                "step": step,
                "perturbation_label": pert_label,
                "drug": base_drug,
                "drug_norm": normalize_drug_name(base_drug),
                "targets": ", ".join(targets),
                "target_list": targets,
                "target_list_norm": targets_norm,
                "num_targets": len(targets),
                "moa_fine": moa,
                "metadata_found": found,
                "path_string": row.get("path_string", " -> ".join(path_list)),
                "drug_name_string": row.get("drug_name_string", " -> ".join(base_drugs)),
            })
    return pd.DataFrame(rows)


def driver_target_frequency(ann: pd.DataFrame, driver_genes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for driver in [normalize_gene_symbol(x) for x in driver_genes]:
        matching = ann[ann["target_list_norm"].apply(lambda xs: driver in set(xs))]
        rows.append({
            "driver_gene": driver,
            "drug_occurrences_targeting_driver": int(len(matching)),
            "unique_drugs_targeting_driver": int(matching["drug"].nunique()) if len(matching) else 0,
            "num_top_paths_with_driver_targeting_drug": int(matching["path_rank_by_score"].nunique()) if len(matching) else 0,
            "drugs": ", ".join(sorted(matching["drug"].unique())) if len(matching) else "",
        })
    return pd.DataFrame(rows).sort_values(["drug_occurrences_targeting_driver", "unique_drugs_targeting_driver"], ascending=False)


def driver_step_matrix(ann: pd.DataFrame, driver_genes: Sequence[str]) -> pd.DataFrame:
    rows = []
    for driver in [normalize_gene_symbol(x) for x in driver_genes]:
        for _, r in ann.iterrows():
            if driver in set(r["target_list_norm"]):
                rows.append({"driver_gene": driver, "step": int(r["step"]), "count": 1})
    if not rows:
        return pd.DataFrame(columns=["driver_gene", "step", "count"])
    return pd.DataFrame(rows).groupby(["driver_gene", "step"], as_index=False)["count"].sum()


def target_frequency(ann: pd.DataFrame) -> pd.DataFrame:
    counter = Counter()
    drug_counter = defaultdict(set)
    path_counter = defaultdict(set)
    moa_counter = defaultdict(Counter)
    for _, r in ann.iterrows():
        for target in r["target_list_norm"]:
            counter[target] += 1
            drug_counter[target].add(r["drug"])
            path_counter[target].add(r["path_rank_by_score"])
            moa_counter[target][r["moa_fine"]] += 1
    rows = []
    for target, n in counter.items():
        rows.append({
            "target": target,
            "target_occurrences": int(n),
            "unique_drugs_with_target": int(len(drug_counter[target])),
            "num_top_paths_with_target": int(len(path_counter[target])),
            "drugs": ", ".join(sorted(drug_counter[target])),
            "most_common_moa_fine": moa_counter[target].most_common(1)[0][0] if moa_counter[target] else "",
        })
    if not rows:
        return pd.DataFrame(columns=["target", "target_occurrences", "unique_drugs_with_target", "num_top_paths_with_target", "drugs", "most_common_moa_fine"])
    return pd.DataFrame(rows).sort_values(["num_top_paths_with_target", "target_occurrences", "unique_drugs_with_target"], ascending=False)


def moa_frequency(ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for moa, sub in ann.groupby("moa_fine", dropna=False):
        rows.append({
            "moa_fine": str(moa),
            "drug_occurrences": int(len(sub)),
            "unique_drugs": int(sub["drug"].nunique()),
            "num_top_paths_with_moa": int(sub["path_rank_by_score"].nunique()),
            "drugs": ", ".join(sorted(sub["drug"].unique())),
        })
    return pd.DataFrame(rows).sort_values(["num_top_paths_with_moa", "drug_occurrences", "unique_drugs"], ascending=False)


def moa_by_step_table(ann: pd.DataFrame) -> pd.DataFrame:
    if ann.empty:
        return pd.DataFrame(columns=["moa_fine", "step", "count"])
    return ann.groupby(["moa_fine", "step"], as_index=False).size().rename(columns={"size": "count"})


def fisher_exact_right_tail(a: int, b: int, c: int, d: int) -> float:
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        return float(p)
    except Exception:
        pass
    n_selected = a + b
    successes = a + c
    total = a + b + c + d
    max_x = min(n_selected, successes)
    def log_choose(n, k):
        if k < 0 or k > n:
            return -math.inf
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    log_den = log_choose(total, n_selected)
    return float(min(1.0, sum(math.exp(log_choose(successes, x) + log_choose(total - successes, n_selected - x) - log_den) for x in range(a, max_x + 1))))


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, ranked[i] * n / (i + 1))
        q[order[i]] = prev
    return np.minimum(q, 1.0).tolist()


def moa_enrichment(ann: pd.DataFrame, drug_meta: pd.DataFrame, background: str, results: pd.DataFrame) -> pd.DataFrame:
    selected_drugs = set(ann["drug_norm"].dropna().astype(str))
    selected_moa = ann.drop_duplicates("drug_norm").set_index("drug_norm")["moa_fine"].to_dict()
    meta = drug_meta.copy()
    if background == "search":
        observed = set()
        for drugs in results["drug_name_list"]:
            observed.update(normalize_drug_name(d) for d in drugs)
        meta = meta[meta["drug_norm"].isin(observed)].copy()
    bg_drugs = set(meta["drug_norm"].dropna().astype(str))
    selected_drugs = selected_drugs & bg_drugs
    if not bg_drugs:
        return pd.DataFrame()
    moa_to_bg = defaultdict(set)
    for _, r in meta.iterrows():
        moa_to_bg[str(r["moa_fine_clean"])].add(str(r["drug_norm"]))
    moa_to_sel = defaultdict(set)
    for drug in selected_drugs:
        moa = selected_moa.get(drug)
        if moa is not None:
            moa_to_sel[str(moa)].add(drug)
    rows = []
    n_selected = len(selected_drugs)
    n_background = len(bg_drugs)
    for moa, bg_set in moa_to_bg.items():
        sel_set = moa_to_sel.get(moa, set())
        a = len(sel_set)
        b = n_selected - a
        c = len(bg_set - selected_drugs)
        d = n_background - n_selected - c
        p = fisher_exact_right_tail(a, b, c, d)
        rows.append({
            "moa_fine": moa,
            "selected_unique_drugs_with_moa": int(a),
            "selected_unique_drugs_total": int(n_selected),
            "background_unique_drugs_with_moa": int(len(bg_set)),
            "background_unique_drugs_total": int(n_background),
            "selected_fraction": a / max(1, n_selected),
            "background_fraction": len(bg_set) / max(1, n_background),
            "odds_ratio_haldane": ((a + 0.5) / (b + 0.5)) / ((c + 0.5) / (d + 0.5)),
            "p_value_fisher_greater": p,
            "selected_drugs": ", ".join(sorted(sel_set)),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["q_value_bh"] = benjamini_hochberg(out["p_value_fisher_greater"].values)
    return out.sort_values(["q_value_bh", "p_value_fisher_greater", "odds_ratio_haldane"], ascending=[True, True, False])


def plot_driver_target_heatmap(driver_step: pd.DataFrame, driver_freq: pd.DataFrame, fig_dir: Path):
    path = fig_dir / "01_driver_target_heatmap.png"
    if driver_step.empty:
        plt.figure(figsize=(6.4, 3.4)); plt.text(0.5, 0.5, "No selected top-path drugs target starting-cell driver genes", ha="center", va="center"); plt.axis("off"); savefig(path); return
    drivers = driver_freq["driver_gene"].tolist()
    mat = driver_step.pivot_table(index="driver_gene", columns="step", values="count", fill_value=0, aggfunc="sum").reindex(drivers).fillna(0)
    plt.figure(figsize=(6.4, max(3.6, 0.38 * len(mat) + 1.4)))
    ax = plt.gca()
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis")
    add_colorbar(im, "Drug occurrences targeting driver")
    plt.xticks(np.arange(mat.shape[1]), [str(c) for c in mat.columns])
    plt.yticks(np.arange(mat.shape[0]), mat.index)
    ax.set_xlabel("Step in selected path"); ax.set_ylabel("Starting-cell driver gene"); ax.set_title("Driver-targeting drugs in top paths")
    polish_axes(ax)
    savefig(path)


def plot_top_targets_barplot(target_df: pd.DataFrame, fig_dir: Path, top_n: int):
    path = fig_dir / "02_top_targets_barplot.png"
    d = target_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(6.6, max(4.0, 0.34 * len(d) + 1.3)))
    ax = plt.gca()
    if d.empty:
        plt.text(0.5, 0.5, "No gene targets found for selected drugs", ha="center", va="center"); plt.axis("off")
    else:
        ax.barh(d["target"], d["num_top_paths_with_target"], color="#2f6fed", edgecolor="#333333", linewidth=0.6)
        ax.set_xlabel("Number of top paths containing target"); ax.set_ylabel("Drug target"); ax.set_title("Most frequent targets among drugs in top paths")
        ax.grid(axis="x", color="#e5e5e5", lw=0.8)
        polish_axes(ax)
    savefig(path)


def plot_moa_frequency(moa_df: pd.DataFrame, fig_dir: Path, top_n: int):
    path = fig_dir / "03_moa_frequency_barplot.png"
    d = moa_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(7.0, max(4.0, 0.34 * len(d) + 1.3)))
    ax = plt.gca()
    if d.empty:
        plt.text(0.5, 0.5, "No MOA annotations found", ha="center", va="center"); plt.axis("off")
    else:
        ax.barh(d["moa_fine"], d["num_top_paths_with_moa"], color="#2ca25f", edgecolor="#333333", linewidth=0.6)
        ax.set_xlabel("Number of top paths containing MOA"); ax.set_ylabel("Mechanism of action"); ax.set_title("Most frequent mechanisms of action in top paths")
        ax.grid(axis="x", color="#e5e5e5", lw=0.8)
        polish_axes(ax)
    savefig(path)


def plot_moa_enrichment(enrich_df: pd.DataFrame, fig_dir: Path, top_n_labels: int = 12):
    path = fig_dir / "04_moa_enrichment_volcano.png"
    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    if enrich_df.empty:
        plt.text(0.5, 0.5, "No MOA enrichment results", ha="center", va="center"); plt.axis("off"); savefig(path); return
    d = enrich_df.copy()
    d["log2_or"] = np.log2(d["odds_ratio_haldane"].replace(0, np.nan))
    d["neglog10_q"] = -np.log10(d["q_value_bh"].clip(lower=1e-300))
    ax.scatter(d["log2_or"], d["neglog10_q"], s=46, alpha=0.86, color="#2f6fed", edgecolor="#333333", linewidth=0.35)
    ax.axvline(0, linestyle="--", linewidth=1.2, color="#444444"); ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1.2, color="#444444")
    ax.set_xlabel("log2 enrichment odds ratio"); ax.set_ylabel("-log10 BH q-value"); ax.set_title("MOA enrichment among selected top-path drugs")
    label_df = d.sort_values(["q_value_bh", "odds_ratio_haldane"], ascending=[True, False]).head(top_n_labels)
    for _, r in label_df.iterrows():
        ax.text(r["log2_or"], r["neglog10_q"], str(r["moa_fine"])[:35], fontsize=9)
    ax.grid(color="#e5e5e5", lw=0.7)
    polish_axes(ax)
    savefig(path)


def plot_moa_by_step_heatmap(moa_step: pd.DataFrame, moa_freq: pd.DataFrame, fig_dir: Path, top_n: int):
    path = fig_dir / "05_moa_by_step_heatmap.png"
    if moa_step.empty:
        plt.figure(figsize=(6.4, 3.4)); plt.text(0.5, 0.5, "No MOA step data", ha="center", va="center"); plt.axis("off"); savefig(path); return
    top_moas = moa_freq["moa_fine"].head(top_n).tolist()
    mat = moa_step[moa_step["moa_fine"].isin(top_moas)].pivot_table(index="moa_fine", columns="step", values="count", fill_value=0, aggfunc="sum").reindex(top_moas).fillna(0)
    plt.figure(figsize=(6.8, max(4.0, 0.36 * len(mat) + 1.4)))
    ax = plt.gca()
    im = ax.imshow(mat.values, aspect="auto", cmap="Blues")
    add_colorbar(im, "Drug occurrences")
    plt.xticks(np.arange(mat.shape[1]), [str(c) for c in mat.columns])
    plt.yticks(np.arange(mat.shape[0]), mat.index)
    ax.set_xlabel("Step in selected path"); ax.set_ylabel("Mechanism of action"); ax.set_title("Mechanisms of action by path position")
    polish_axes(ax)
    savefig(path)


def plot_driver_target_path_matrix(ann: pd.DataFrame, driver_genes: Sequence[str], fig_dir: Path, top_n_paths: int):
    path = fig_dir / "06_driver_target_path_matrix.png"
    drivers = [normalize_gene_symbol(x) for x in driver_genes]
    if not drivers:
        plt.figure(figsize=(6.4, 3.4)); plt.text(0.5, 0.5, "No starting-cell driver genes available", ha="center", va="center"); plt.axis("off"); savefig(path); return
    top_paths = sorted(ann["path_rank_by_score"].unique())[:top_n_paths]
    mat = np.zeros((len(drivers), len(top_paths)), dtype=float)
    for j, path_rank in enumerate(top_paths):
        sub = ann[ann["path_rank_by_score"] == path_rank]
        targets = set()
        for xs in sub["target_list_norm"]:
            targets.update(xs)
        for i, driver in enumerate(drivers):
            mat[i, j] = 1.0 if driver in targets else 0.0
    if mat.sum() == 0:
        plt.figure(figsize=(6.4, 3.4)); plt.text(0.5, 0.5, "No top paths include drugs targeting starting-cell driver genes", ha="center", va="center"); plt.axis("off"); savefig(path); return
    plt.figure(figsize=(max(6.8, 0.28 * len(top_paths) + 1.8), max(3.6, 0.38 * len(drivers) + 1.4)))
    ax = plt.gca()
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="Greens")
    add_colorbar(im, "Driver targeted by any drug in path")
    plt.xticks(np.arange(len(top_paths)), [str(x) for x in top_paths], rotation=90, fontsize=7)
    plt.yticks(np.arange(len(drivers)), drivers)
    ax.set_xlabel("Top path rank"); ax.set_ylabel("Starting-cell driver gene"); ax.set_title("Driver-gene targeting across top paths")
    polish_axes(ax)
    savefig(path)


def write_summary(report_dir: Path, run_dir: Path, start_cell: str, organ: str, driver_genes: List[str], ann: pd.DataFrame, driver_freq: pd.DataFrame, target_df: pd.DataFrame, moa_df: pd.DataFrame, enrich_df: pd.DataFrame, args):
    lines = []
    lines.append("# Sample and Drug Metadata Report\n")
    lines.append(f"Run directory: `{run_dir}`\n")
    lines.append(f"Starting cell type: `{start_cell}`")
    lines.append(f"Organ: `{organ}`")
    lines.append(f"Driver genes: `{', '.join(driver_genes) if driver_genes else 'none found'}`")
    lines.append(f"Top paths analyzed: `{args.top_n_paths}`")
    lines.append(f"Frequency uses final depth only: `{args.final_depth_only}`\n")
    n_occ = len(ann); n_unique_drugs = ann["drug"].nunique() if len(ann) else 0; n_missing = int((~ann["metadata_found"]).sum()) if len(ann) else 0; n_no_targets = int((ann["num_targets"] == 0).sum()) if len(ann) else 0
    driver_set = set(driver_genes)
    driver_targeting_occ = int(ann["target_list_norm"].apply(lambda xs: len(driver_set & set(xs)) > 0).sum()) if driver_genes and len(ann) else 0
    lines.append("## Executive summary\n")
    lines.append(f"- Drug occurrences analyzed across selected paths: `{n_occ}`")
    lines.append(f"- Unique selected drugs: `{n_unique_drugs}`")
    lines.append(f"- Drug occurrences missing metadata: `{n_missing}`")
    lines.append(f"- Drug occurrences with no listed target: `{n_no_targets}`")
    lines.append(f"- Drug occurrences targeting at least one starting-cell driver gene: `{driver_targeting_occ}`\n")
    lines.append("## Starting-cell driver targeting\n")
    lines.append(markdown_table(driver_freq, max_rows=30) + "\n")
    lines.append("## Top drug targets among selected paths\n")
    cols = ["target", "num_top_paths_with_target", "target_occurrences", "unique_drugs_with_target", "drugs", "most_common_moa_fine"]
    lines.append(markdown_table(target_df[cols] if not target_df.empty else target_df, max_rows=25) + "\n")
    lines.append("## Mechanisms of action among selected paths\n")
    cols = ["moa_fine", "num_top_paths_with_moa", "drug_occurrences", "unique_drugs", "drugs"]
    lines.append(markdown_table(moa_df[cols] if not moa_df.empty else moa_df, max_rows=25) + "\n")
    lines.append("## MOA enrichment\n")
    if enrich_df.empty:
        lines.append("_No MOA enrichment results._\n")
    else:
        cols = ["moa_fine", "selected_unique_drugs_with_moa", "background_unique_drugs_with_moa", "odds_ratio_haldane", "p_value_fisher_greater", "q_value_bh", "selected_drugs"]
        lines.append(markdown_table(enrich_df[cols], max_rows=25) + "\n")
    lines.append("## Figures\n")
    for fname, desc in [
        ("01_driver_target_heatmap.png", "driver genes by path step, counting drugs that target each driver"),
        ("02_top_targets_barplot.png", "most frequent drug targets among top paths"),
        ("03_moa_frequency_barplot.png", "mechanisms of action most often used in top paths"),
        ("04_moa_enrichment_volcano.png", "MOA enrichment among selected unique drugs"),
        ("05_moa_by_step_heatmap.png", "MOA usage by step position"),
        ("06_driver_target_path_matrix.png", "which top paths include driver-targeting drugs"),
    ]:
        lines.append(f"- `{fname}`: {desc}")
    (report_dir / "summary.md").write_text("\n".join(lines))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate sample-specific and drug-specific PHAROS report from search outputs.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run-dir", required=True, help="Top-level cell_converter output dir or its search/ subdirectory.")
    p.add_argument("--metadata-dir", default="metadata", help="Directory containing cell_line_metadata.csv and drug_metadata.csv.")
    p.add_argument("--cell-metadata", default=None, help="Optional explicit path to cell_line_metadata.csv.")
    p.add_argument("--drug-metadata", default=None, help="Optional explicit path to drug_metadata.csv.")
    p.add_argument("--output-dir", default=None, help="Output directory. Default: <run-dir>/sample_drug_report.")
    p.add_argument("--start-cell", default=None, help="Starting cell type. If omitted, inferred from results.tsv when possible.")
    p.add_argument("--top-n-paths", type=int, default=50, help="Number of top paths analyzed.")
    p.add_argument("--final-depth-only", action=argparse.BooleanOptionalAction, default=True, help="Analyze only top paths at deepest search depth.")
    p.add_argument("--background", choices=["metadata", "search"], default="metadata", help="Background drug universe for MOA enrichment.")
    p.add_argument("--top-n-targets", type=int, default=25, help="Number of top targets shown in plot.")
    p.add_argument("--top-n-moas", type=int, default=25, help="Number of top MOAs shown in plots.")
    p.add_argument("--top-n-path-matrix", type=int, default=50, help="Number of top paths shown in driver-target matrix.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_dir_input = Path(args.run_dir)
    run_dir = infer_top_level_run_dir(run_dir_input)
    results_path = resolve_search_results(run_dir_input)
    metadata_dir = Path(args.metadata_dir)
    cell_meta_path = Path(args.cell_metadata) if args.cell_metadata else metadata_dir / "cell_line_metadata.csv"
    drug_meta_path = Path(args.drug_metadata) if args.drug_metadata else metadata_dir / "drug_metadata.csv"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "sample_drug_report"
    table_dir = safe_mkdir(output_dir / "tables")
    fig_dir = safe_mkdir(output_dir / "figures")

    print(f"Loading search results: {results_path}")
    results = load_results(results_path)
    start_cell = extract_start_cell(results, args.start_cell)
    print(f"Loading cell metadata: {cell_meta_path}")
    cell_meta = load_cell_metadata(cell_meta_path)
    print(f"Loading drug metadata: {drug_meta_path}")
    drug_meta = load_drug_metadata(drug_meta_path)
    organ, driver_genes, start_driver_rows = get_starting_cell_drivers(cell_meta, start_cell)
    print(f"Start cell: {start_cell}")
    print(f"Organ: {organ}")
    print(f"Driver genes: {', '.join(driver_genes)}")

    ann = annotate_top_paths(results, drug_meta, top_n_paths=args.top_n_paths, final_depth_only=args.final_depth_only)
    driver_set = set(driver_genes)
    ann["targets_starting_driver"] = ann["target_list_norm"].apply(lambda xs: bool(driver_set & set(xs)))
    ann["targeted_starting_driver_genes"] = ann["target_list_norm"].apply(lambda xs: ", ".join(sorted(driver_set & set(xs))))
    driver_freq = driver_target_frequency(ann, driver_genes)
    driver_step = driver_step_matrix(ann, driver_genes)
    target_df = target_frequency(ann)
    moa_df = moa_frequency(ann)
    moa_step = moa_by_step_table(ann)
    enrich_df = moa_enrichment(ann, drug_meta, background=args.background, results=results)

    start_driver_rows.to_csv(table_dir / "starting_cell_drivers.tsv", sep="\t", index=False)
    ann_save = ann.copy()
    ann_save["target_list"] = ann_save["target_list"].apply(lambda xs: ", ".join(xs))
    ann_save["target_list_norm"] = ann_save["target_list_norm"].apply(lambda xs: ", ".join(xs))
    ann_save.to_csv(table_dir / "top_path_drug_annotations.tsv", sep="\t", index=False)
    driver_freq.to_csv(table_dir / "driver_target_frequency.tsv", sep="\t", index=False)
    driver_step.to_csv(table_dir / "driver_target_by_step.tsv", sep="\t", index=False)
    target_df.to_csv(table_dir / "target_frequency.tsv", sep="\t", index=False)
    moa_df.to_csv(table_dir / "moa_frequency.tsv", sep="\t", index=False)
    moa_step.to_csv(table_dir / "moa_by_step.tsv", sep="\t", index=False)
    enrich_df.to_csv(table_dir / "moa_enrichment.tsv", sep="\t", index=False)

    plot_driver_target_heatmap(driver_step, driver_freq, fig_dir)
    plot_top_targets_barplot(target_df, fig_dir, top_n=args.top_n_targets)
    plot_moa_frequency(moa_df, fig_dir, top_n=args.top_n_moas)
    plot_moa_enrichment(enrich_df, fig_dir)
    plot_moa_by_step_heatmap(moa_step, moa_df, fig_dir, top_n=args.top_n_moas)
    plot_driver_target_path_matrix(ann, driver_genes, fig_dir, top_n_paths=args.top_n_path_matrix)
    write_summary(output_dir, run_dir, start_cell, organ, driver_genes, ann, driver_freq, target_df, moa_df, enrich_df, args)

    print("\n=== Sample/drug report complete ===")
    print(f"summary: {output_dir / 'summary.md'}")
    print(f"tables:  {table_dir}")
    print(f"figures: {fig_dir}")


if __name__ == "__main__":
    main()
