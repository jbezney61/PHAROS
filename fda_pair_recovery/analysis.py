#!/usr/bin/env python
"""Per-run target-pair abundance and exact-pair recovery vs Model B MC null."""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SALT_WORDS = {
    "acetate",
    "besylate",
    "bromide",
    "calcium",
    "chloride",
    "citrate",
    "dihydrochloride",
    "disodium",
    "fumarate",
    "hbr",
    "hcl",
    "hemisulfate",
    "hydrate",
    "hydrobromide",
    "hydrochloride",
    "maleate",
    "mesylate",
    "monohydrate",
    "nitrate",
    "phosphate",
    "potassium",
    "sodium",
    "succinate",
    "sulfate",
    "tartrate",
    "tosylate",
}


@dataclass(frozen=True)
class TargetPair:
    pair_id: str
    drug_a: str
    drug_b: str
    key_a: str
    key_b: str
    pair_key: Tuple[str, str]


@dataclass(frozen=True)
class RunInput:
    label: str
    input_path: Path
    results_path: Path
    config_path: Optional[Path]
    allow_repeated_drug_names: bool
    allow_repeated_perturbation_labels: bool
    max_drugs_to_consider: Optional[int]


@dataclass(frozen=True)
class SearchSpace:
    num_drugs: int
    concentrations_per_drug: int
    allow_repeated_drug_names: bool = False
    allow_repeated_perturbation_labels: bool = False

    @property
    def num_labels(self) -> int:
        return self.num_drugs * self.concentrations_per_drug

    def pool_size(self, beam_size: int) -> int:
        labels = self.num_labels
        if self.allow_repeated_drug_names:
            if self.allow_repeated_perturbation_labels:
                return beam_size * labels
            return beam_size * max(labels - 1, 0)
        # Forbid same base drug on a path: C * (N - 1) legal second labels.
        return beam_size * self.concentrations_per_drug * max(self.num_drugs - 1, 0)


def normalize_drug_name(name: Any) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [tok for tok in text.split() if tok not in SALT_WORDS]
    return " ".join(tokens)


def pair_key(drug_a: Any, drug_b: Any) -> Tuple[str, str]:
    return tuple(sorted((normalize_drug_name(drug_a), normalize_drug_name(drug_b))))


def parse_jsonish_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
        if parsed is not None:
            return [str(parsed).strip()]

    if "->" in text:
        return [part.strip() for part in text.split("->") if part.strip()]
    if "+" in text:
        return [part.strip() for part in text.split("+") if part.strip()]
    return [text]


def perturbation_to_drug_name(perturbation_label: str) -> str:
    text = str(perturbation_label)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and parsed:
            first = parsed[0]
            if isinstance(first, (tuple, list)) and first:
                return str(first[0])
        if isinstance(parsed, tuple) and parsed:
            return str(parsed[0])
    except Exception:
        pass
    match = re.search(r"['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else text


def p_to_stars(p_value: Any) -> str:
    if p_value is None or (isinstance(p_value, float) and not math.isfinite(p_value)):
        return ""
    p = float(p_value)
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def format_p(p_value: Any) -> str:
    """Paper-friendly p display: 3 decimals, or (<0.001) when smaller."""
    if p_value is None or (isinstance(p_value, float) and not math.isfinite(p_value)):
        return "—"
    p = float(p_value)
    if p < 0.001:
        return "(<0.001)"
    return f"({p:.3f})"


def resolve_results_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [path / "results.tsv", path / "search" / "results.tsv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    wildcard = sorted(path.glob("*_results.tsv"))
    if len(wildcard) == 1:
        return wildcard[0]
    if len(wildcard) > 1:
        raise ValueError(
            f"{path} contains multiple *_results.tsv files. Pass each file/path separately."
        )
    raise FileNotFoundError(f"Could not find results.tsv for {path}")


def read_search_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded if isinstance(loaded, dict) else {}


def resolve_run_inputs(run_dirs: Sequence[Path], labels: Sequence[str]) -> List[RunInput]:
    if len(run_dirs) != len(labels):
        raise ValueError(
            f"--run-dirs has {len(run_dirs)} entries but --labels has {len(labels)} entries"
        )
    if len(set(labels)) != len(labels):
        raise ValueError("--labels must be unique")

    runs: List[RunInput] = []
    for run_path, label in zip(run_dirs, labels):
        results_path = resolve_results_path(Path(run_path))
        config_path = results_path.parent / "search_config.used.yaml"
        cfg = read_search_config(config_path)
        constraints = cfg.get("constraints", {}) or {}
        search_cfg = cfg.get("search", {}) or {}
        runs.append(
            RunInput(
                label=str(label),
                input_path=Path(run_path),
                results_path=results_path,
                config_path=config_path if config_path.exists() else None,
                allow_repeated_drug_names=bool(constraints.get("allow_repeated_drug_names", False)),
                allow_repeated_perturbation_labels=bool(
                    constraints.get("allow_repeated_perturbation_labels", False)
                ),
                max_drugs_to_consider=search_cfg.get("max_drugs_to_consider", None),
            )
        )
    return runs


def read_target_pairs_table(
    path: Path,
    run_key_col: str = "label",
    drug_a_col: str = "drug_a",
    drug_b_col: str = "drug_b",
    pair_id_col: str = "pair_id",
) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    required = {run_key_col, drug_a_col, drug_b_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out[run_key_col] = out[run_key_col].astype(str).str.strip()
    out[drug_a_col] = out[drug_a_col].astype(str).str.strip()
    out[drug_b_col] = out[drug_b_col].astype(str).str.strip()
    if pair_id_col in out.columns:
        out[pair_id_col] = out[pair_id_col].astype(str).str.strip()
    else:
        out[pair_id_col] = out[drug_a_col] + " + " + out[drug_b_col]

    out = out.loc[
        out[run_key_col].ne("")
        & out[drug_a_col].ne("")
        & out[drug_b_col].ne("")
        & ~out[run_key_col].str.lower().eq("nan")
        & ~out[drug_a_col].str.lower().eq("nan")
        & ~out[drug_b_col].str.lower().eq("nan")
    ].copy()
    if out.empty:
        raise ValueError(f"No valid target-pair rows found in {path}")

    dup = out[run_key_col][out[run_key_col].duplicated()].tolist()
    if dup:
        raise ValueError(f"{path} has duplicate run keys: {sorted(set(dup))}")
    return out


def assign_pairs_by_label(
    run_inputs: Sequence[RunInput],
    pairs_df: pd.DataFrame,
    run_key_col: str = "label",
    drug_a_col: str = "drug_a",
    drug_b_col: str = "drug_b",
    pair_id_col: str = "pair_id",
) -> Dict[str, TargetPair]:
    by_label = {str(row[run_key_col]): row for _, row in pairs_df.iterrows()}
    labels = [run.label for run in run_inputs]
    missing = [lab for lab in labels if lab not in by_label]
    if missing:
        raise ValueError(
            f"Target-pair table is missing rows for labels: {missing}. "
            f"Expected a row for every --labels entry via column '{run_key_col}'."
        )
    extra = sorted(set(by_label) - set(labels))
    if extra:
        raise ValueError(
            f"Target-pair table has unused labels not present in --labels: {extra}"
        )

    assigned: Dict[str, TargetPair] = {}
    for lab in labels:
        row = by_label[lab]
        drug_a = str(row[drug_a_col])
        drug_b = str(row[drug_b_col])
        pid = str(row[pair_id_col]) if str(row[pair_id_col]).lower() != "nan" else f"{drug_a} + {drug_b}"
        assigned[lab] = TargetPair(
            pair_id=pid,
            drug_a=drug_a,
            drug_b=drug_b,
            key_a=normalize_drug_name(drug_a),
            key_b=normalize_drug_name(drug_b),
            pair_key=pair_key(drug_a, drug_b),
        )
    return assigned


def depth_values(depth_mode: str) -> Tuple[int, ...]:
    if str(depth_mode) == "1":
        return (1,)
    if str(depth_mode) == "2":
        return (2,)
    if str(depth_mode).lower() == "both":
        return (1, 2)
    raise ValueError("--depth must be one of: 1, 2, both")


def load_selected_rows(results_path: Path, depth_mode: str, rank_threshold: int) -> pd.DataFrame:
    df = pd.read_csv(results_path, sep="\t")
    required = {"depth", "rank", "drug_names_json"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{results_path} is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["depth"] = pd.to_numeric(out["depth"], errors="coerce")
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["rank_for_recovery"] = out["rank"]
    out["rank_source"] = "rank"
    if "score_sinkhorn_ot" in out.columns:
        out["score_sinkhorn_ot"] = pd.to_numeric(out["score_sinkhorn_ot"], errors="coerce")
        out["rank_for_recovery"] = np.nan
        out["rank_source"] = "score_sinkhorn_ot"
        for depth, idx in out.groupby("depth").groups.items():
            sub = out.loc[idx]
            finite = sub[np.isfinite(sub["score_sinkhorn_ot"])]
            if finite.empty:
                out.loc[idx, "rank_for_recovery"] = out.loc[idx, "rank"]
                out.loc[idx, "rank_source"] = "rank"
                continue
            ordered_idx = finite.sort_values(["score_sinkhorn_ot"], kind="stable").index
            out.loc[ordered_idx, "rank_for_recovery"] = np.arange(1, len(ordered_idx) + 1)
    selected = out.loc[
        out["depth"].isin(depth_values(depth_mode)) & (out["rank_for_recovery"] <= int(rank_threshold))
    ].copy()
    return selected.sort_values(["depth", "rank_for_recovery"], kind="stable")


def row_drug_names(row: pd.Series) -> List[str]:
    names = parse_jsonish_list(row.get("drug_names_json", ""))
    if names:
        return names
    path_labels = parse_jsonish_list(row.get("path_json", ""))
    return [perturbation_to_drug_name(label) for label in path_labels]


def summarize_observed_pair(
    selected: pd.DataFrame,
    pair: TargetPair,
) -> Dict[str, Any]:
    n_a = 0
    n_b = 0
    best_exact_rank: Optional[int] = None
    exact = False

    for _, row in selected.iterrows():
        depth = int(row["depth"])
        rank = int(row["rank_for_recovery"])
        names = [normalize_drug_name(name) for name in row_drug_names(row)]
        names = [name for name in names if name]
        n_a += sum(1 for name in names if name == pair.key_a)
        n_b += sum(1 for name in names if name == pair.key_b)
        if depth == 2 and len(names) >= 2:
            if tuple(sorted((names[0], names[1]))) == pair.pair_key:
                exact = True
                if best_exact_rank is None or rank < best_exact_rank:
                    best_exact_rank = rank

    return {
        "n_A": int(n_a),
        "n_B": int(n_b),
        "exact_pair": bool(exact),
        "best_exact_rank": int(best_exact_rank) if best_exact_rank is not None else np.nan,
        "n_selected_rows": int(len(selected)),
    }


def _compact_to_label(rem: int, forbidden_base: int, conc: int, num_drugs: int) -> int:
    base = rem // conc
    dose = rem % conc
    if base >= forbidden_base:
        base += 1
    if base >= num_drugs:
        raise ValueError("compact label index out of range")
    return base * conc + dose


def _second_label_for_path(
    rng: np.random.Generator,
    first_label: int,
    space: SearchSpace,
) -> int:
    conc = space.concentrations_per_drug
    first_base = int(first_label) // conc
    n_labels = space.num_labels

    if space.allow_repeated_drug_names:
        if space.allow_repeated_perturbation_labels:
            return int(rng.integers(0, n_labels))
        second = int(rng.integers(0, n_labels - 1))
        if second >= first_label:
            second += 1
        return second

    # Distinct base drug: choose among C*(N-1) labels.
    rem = int(rng.integers(0, conc * (space.num_drugs - 1)))
    return _compact_to_label(rem, first_base, conc, space.num_drugs)


def sample_model_b_beam(
    rng: np.random.Generator,
    beam_size: int,
    space: SearchSpace,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return base-drug indices for first and second positions of x sampled paths."""
    n_labels = space.num_labels
    if beam_size > n_labels:
        raise ValueError(f"beam_size={beam_size} exceeds number of labels={n_labels}")

    b1 = rng.choice(n_labels, size=beam_size, replace=False)
    first_bases = np.empty(beam_size, dtype=np.int64)
    second_bases = np.empty(beam_size, dtype=np.int64)

    # Sample unique ordered labeled paths via rejection on (first_slot, second_label).
    selected: set[Tuple[int, int]] = set()
    conc = space.concentrations_per_drug
    max_attempts = max(1000, beam_size * 200)
    attempts = 0
    filled = 0
    while filled < beam_size:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Failed to sample {beam_size} unique Model B paths after {attempts} attempts"
            )
        first_i = int(rng.integers(0, beam_size))
        first_label = int(b1[first_i])
        second_label = _second_label_for_path(rng, first_label, space)
        key = (first_label, second_label)
        if key in selected:
            continue
        selected.add(key)
        first_bases[filled] = first_label // conc
        second_bases[filled] = second_label // conc
        filled += 1

    return first_bases, second_bases


def score_null_beam(
    first_bases: np.ndarray,
    second_bases: np.ndarray,
    drug_a_index: int = 0,
    drug_b_index: int = 1,
) -> Tuple[int, int, bool]:
    n_a = int(np.sum(first_bases == drug_a_index) + np.sum(second_bases == drug_a_index))
    n_b = int(np.sum(first_bases == drug_b_index) + np.sum(second_bases == drug_b_index))
    exact = bool(
        np.any(
            ((first_bases == drug_a_index) & (second_bases == drug_b_index))
            | ((first_bases == drug_b_index) & (second_bases == drug_a_index))
        )
    )
    return n_a, n_b, exact


def empirical_p_ge(null_values: np.ndarray, observed: float) -> float:
    return float((np.sum(null_values >= observed) + 1) / (len(null_values) + 1))


def run_model_b_mc(
    beam_size: int,
    space: SearchSpace,
    n_mc: int,
    seed: int,
    observed_n_a: int,
    observed_n_b: int,
    observed_exact: bool,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    null_a = np.empty(n_mc, dtype=np.int64)
    null_b = np.empty(n_mc, dtype=np.int64)
    null_exact = np.empty(n_mc, dtype=np.int64)

    for i in range(n_mc):
        first_bases, second_bases = sample_model_b_beam(rng, beam_size, space)
        n_a, n_b, exact = score_null_beam(first_bases, second_bases)
        null_a[i] = n_a
        null_b[i] = n_b
        null_exact[i] = int(exact)

    p_a = empirical_p_ge(null_a, observed_n_a)
    p_b = empirical_p_ge(null_b, observed_n_b)
    # Exact: report enrichment p only when a hit was observed.
    if observed_exact:
        p_exact = empirical_p_ge(null_exact.astype(float), 1.0)
    else:
        p_exact = np.nan

    return {
        "p_A": p_a,
        "p_B": p_b,
        "p_exact": p_exact,
        "null_mean_n_A": float(null_a.mean()),
        "null_mean_n_B": float(null_b.mean()),
        "null_exact_rate": float(null_exact.mean()),
    }


def validate_search_space(
    run_inputs: Sequence[RunInput],
    num_drugs: int,
    concentrations_per_drug: int,
) -> List[str]:
    notes: List[str] = []
    for run in run_inputs:
        if run.max_drugs_to_consider is not None:
            notes.append(
                f"{run.label}: search_config.used.yaml has max_drugs_to_consider="
                f"{run.max_drugs_to_consider}; verify --num-drugs/--concentrations-per-drug "
                "match that filtered search space."
            )
        if run.config_path is None:
            notes.append(
                f"{run.label}: no search_config.used.yaml found next to results.tsv; "
                "using default no-repeat drug/label constraints."
            )
    if num_drugs <= 1:
        raise ValueError("--num-drugs must be greater than 1")
    if concentrations_per_drug <= 0:
        raise ValueError("--concentrations-per-drug must be positive")
    return notes


def build_display_row(row: pd.Series) -> Dict[str, str]:
    p_a = row["p_A"]
    p_b = row["p_B"]
    exact = bool(row["exact_pair"])
    p_exact = row["p_exact"]
    best = row["best_exact_rank"]
    best_str = "—" if pd.isna(best) else str(int(best))
    drug_a = str(row["drug_a"])
    drug_b = str(row["drug_b"])
    return {
        "label": str(row["label"]),
        "pair_id": str(row["pair_id"]),
        "target_pair": f"{drug_a} + {drug_b}",
        "drug_a": drug_a,
        "drug_b": drug_b,
        "n_A": f"{int(row['n_A'])}{p_to_stars(p_a)}",
        "n_A_p": format_p(p_a),
        "n_B": f"{int(row['n_B'])}{p_to_stars(p_b)}",
        "n_B_p": format_p(p_b),
        "exact": (("yes" if exact else "no") + (p_to_stars(p_exact) if exact else "")),
        "exact_p": format_p(p_exact) if exact else "—",
        "best_exact_rank": best_str,
    }


def run_analysis(
    run_dirs: Sequence[Path],
    labels: Sequence[str],
    target_pairs_path: Path,
    depth_mode: str,
    rank_threshold: int,
    permutations: int,
    output_dir: Path,
    num_drugs: int = 379,
    concentrations_per_drug: int = 3,
    seed: int = 1,
    run_key_col: str = "label",
    drug_a_col: str = "drug_a",
    drug_b_col: str = "drug_b",
    pair_id_col: str = "pair_id",
) -> Dict[str, Any]:
    if int(rank_threshold) <= 0:
        raise ValueError("--rank-threshold must be positive")
    if int(permutations) <= 0:
        raise ValueError("--permutations must be positive")

    pairs_df = read_target_pairs_table(
        target_pairs_path,
        run_key_col=run_key_col,
        drug_a_col=drug_a_col,
        drug_b_col=drug_b_col,
        pair_id_col=pair_id_col,
    )
    run_inputs = resolve_run_inputs(run_dirs, labels)
    notes = validate_search_space(run_inputs, int(num_drugs), int(concentrations_per_drug))
    assigned = assign_pairs_by_label(
        run_inputs,
        pairs_df,
        run_key_col=run_key_col,
        drug_a_col=drug_a_col,
        drug_b_col=drug_b_col,
        pair_id_col=pair_id_col,
    )

    rows: List[Dict[str, Any]] = []
    for i, run in enumerate(run_inputs):
        pair = assigned[run.label]
        selected = load_selected_rows(run.results_path, depth_mode, int(rank_threshold))
        obs = summarize_observed_pair(selected, pair)
        space = SearchSpace(
            num_drugs=int(num_drugs),
            concentrations_per_drug=int(concentrations_per_drug),
            allow_repeated_drug_names=run.allow_repeated_drug_names,
            allow_repeated_perturbation_labels=run.allow_repeated_perturbation_labels,
        )
        if int(rank_threshold) > space.pool_size(int(rank_threshold)):
            raise ValueError(
                f"{run.label}: rank-threshold={rank_threshold} exceeds Model B pool size "
                f"{space.pool_size(int(rank_threshold))}"
            )
        mc = run_model_b_mc(
            beam_size=int(rank_threshold),
            space=space,
            n_mc=int(permutations),
            seed=int(seed) + 1009 * i,
            observed_n_a=int(obs["n_A"]),
            observed_n_b=int(obs["n_B"]),
            observed_exact=bool(obs["exact_pair"]),
        )
        rows.append(
            {
                "label": run.label,
                "pair_id": pair.pair_id,
                "drug_a": pair.drug_a,
                "drug_b": pair.drug_b,
                "input_path": str(run.input_path),
                "results_path": str(run.results_path),
                "n_A": obs["n_A"],
                "p_A": mc["p_A"],
                "n_B": obs["n_B"],
                "p_B": mc["p_B"],
                "exact_pair": obs["exact_pair"],
                "p_exact": mc["p_exact"],
                "best_exact_rank": obs["best_exact_rank"],
                "n_selected_rows": obs["n_selected_rows"],
                "n_mc": int(permutations),
                "null_mean_n_A": mc["null_mean_n_A"],
                "null_mean_n_B": mc["null_mean_n_B"],
                "null_exact_rate": mc["null_exact_rate"],
            }
        )

    per_run = pd.DataFrame(rows)
    display = pd.DataFrame([build_display_row(row) for _, row in per_run.iterrows()])

    return {
        "output_dir": Path(output_dir),
        "run_inputs": run_inputs,
        "assigned_pairs": assigned,
        "per_run_results": per_run,
        "per_run_results_display": display,
        "notes": notes,
        "params": {
            "depth": depth_mode,
            "rank_threshold": int(rank_threshold),
            "permutations": int(permutations),
            "num_drugs": int(num_drugs),
            "concentrations_per_drug": int(concentrations_per_drug),
            "seed": int(seed),
            "target_pairs_path": str(target_pairs_path),
            "run_key_col": run_key_col,
        },
    }


def save_analysis_tables(analysis: Dict[str, Any]) -> Dict[str, Path]:
    output_dir = Path(analysis["output_dir"])
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_run_results": tables_dir / "per_run_results.tsv",
        "per_run_results_display": tables_dir / "per_run_results_display.tsv",
    }
    analysis["per_run_results"].to_csv(paths["per_run_results"], sep="\t", index=False)
    analysis["per_run_results_display"].to_csv(
        paths["per_run_results_display"], sep="\t", index=False
    )
    return paths


def write_summary_markdown(
    analysis: Dict[str, Any],
    table_paths: Dict[str, Path],
    figure_paths: Dict[str, Tuple[Path, Path]],
) -> Path:
    output_dir = Path(analysis["output_dir"])
    summary_path = output_dir / "summary.md"
    params = analysis["params"]
    display = analysis["per_run_results_display"]
    per_run = analysis["per_run_results"]

    lines = [
        "# Target Pair Recovery Report",
        "",
        "## Parameters",
        "",
        f"- Depth mode: `{params['depth']}`",
        f"- Rank threshold (beam width): `{params['rank_threshold']}`",
        f"- MC replicates per run: `{params['permutations']}`",
        f"- Null search space: `{params['num_drugs']}` drugs × "
        f"`{params['concentrations_per_drug']}` concentrations",
        f"- Target pairs file: `{params['target_pairs_path']}`",
        f"- Run key column: `{params['run_key_col']}`",
        f"- Seed: `{params['seed']}`",
        "",
        "## Per-run results",
        "",
        "A and B are the first and second drugs of the target pair, respectively.",
        "",
        "| Target pair | # appearances of A | # appearances of B | Exact pair found | Exact pair OT rank |",
        "|---|---|---|---|---|",
    ]
    for _, row in display.iterrows():
        lines.append(
            f"| {row['target_pair']} | "
            f"{row['n_A']}<br>{row['n_A_p']} | {row['n_B']}<br>{row['n_B_p']} | "
            f"{row['exact']}<br>{row['exact_p']} | {row['best_exact_rank']} |"
        )

    lines.extend(["", "## Numeric details", ""])
    for _, row in per_run.iterrows():
        exact = "yes" if bool(row["exact_pair"]) else "no"
        best = "—" if pd.isna(row["best_exact_rank"]) else str(int(row["best_exact_rank"]))
        p_exact = format_p(row["p_exact"]) if bool(row["exact_pair"]) else "—"
        target_pair = f"{row['drug_a']} + {row['drug_b']}"
        lines.append(
            f"- **{row['label']}** (`{target_pair}`): "
            f"#A={int(row['n_A'])} {format_p(row['p_A'])}, "
            f"#B={int(row['n_B'])} {format_p(row['p_B'])}, "
            f"exact={exact} {p_exact}, exact_pair_ot_rank={best}"
        )

    if analysis["notes"]:
        lines.extend(["", "## Notes", ""])
        for note in analysis["notes"]:
            lines.append(f"- {note}")

    lines.extend(["", "## Outputs", ""])
    for name, path in table_paths.items():
        lines.append(f"- {name}: `{path}`")
    for name, paths in figure_paths.items():
        lines.append(f"- {name}: `{paths[0]}` and `{paths[1]}`")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path
