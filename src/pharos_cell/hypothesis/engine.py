#!/usr/bin/env python
"""
positive_control_2drug.py

Core utilities for positive-control 2-drug ST-SE conversion analysis.

This module is intentionally additive: it reuses the existing data_loader,
converter, scoring, and search helpers, but does not alter any existing files.

Workflow
--------
1. Sample start/target batches from an SE-embedded AnnData file.
2. Record baseline Sinkhorn OT between the untreated start state and target.
3. For an explicit 2-drug pair, search both drug orders and all concentration
   label pairs on batch 0, then evaluate the best ordered pair over all batches.
4. For all ordered pairs matching the two requested moa-fine terms, aligned to
   the best explicit order when available, do the same concentration selection
   and multi-batch evaluation.
5. For random controls, either match the explicit-pair order/concentration
   search on batch 0 or use the legacy direct ordered-label sampling, then
   evaluate the selected random controls over all batches.

Outputs are written under:
    output_dir/
        positive_control_config.used.json
        positive_control_checkpoint.pt
        trajectory_embeddings/
            explicit_pair_trajectory.npz
            explicit_pair_trajectory_metadata.json
        tables/
            baseline_results.tsv
            batch_metadata.tsv
            concentration_selection_scores.tsv
            selected_pairs.tsv
            random_pairs.tsv
            evaluation_results.tsv
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


CONTROL_LIKE_SUBSTRINGS = ("dmso", "control", "non-targeting")


@dataclass
class ScoringParams:
    """Parameters passed to DistributionScorer."""

    normalize: bool = True
    sinkhorn_metric: str = "cosine"
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 100
    projection_auto_epsilon: bool = True


def json_default(value: Any) -> Any:
    """JSON helper for numpy, torch, and Path values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def normalize_name(value: Any) -> str:
    """Case-insensitive key that preserves punctuation but normalizes spaces."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def loose_name(value: Any) -> str:
    """Loose key for suggestions and fallback matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def is_control_like_label(label: str) -> bool:
    lower = str(label).lower()
    return any(x in lower for x in CONTROL_LIKE_SUBSTRINGS)


def perturbation_to_drug_name(perturbation_label: str) -> str:
    """
    Extract the base drug name from a Tahoe perturbation label.

    Expected label:
        "[('Trametinib', 0.05, 'uM')]"
    """
    s = str(perturbation_label)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list) and len(parsed) > 0:
            first = parsed[0]
            if isinstance(first, (tuple, list)) and len(first) > 0:
                return str(first[0])
        if isinstance(parsed, tuple) and len(parsed) > 0:
            return str(parsed[0])
    except Exception:
        pass

    m = re.search(r"['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)
    return s


def parse_perturbation_dose(perturbation_label: str) -> Tuple[Any, str]:
    """
    Parse Tahoe-style perturbation labels.

    Expected label:
        "[('Trametinib', 0.05, 'uM')]"

    Returns:
        (0.05, "uM") when available, otherwise (nan, "").
    """
    try:
        parsed = ast.literal_eval(str(perturbation_label))
        first = parsed[0] if isinstance(parsed, list) and parsed else parsed
        if isinstance(first, (tuple, list)) and len(first) >= 3:
            return first[1], str(first[2])
    except Exception:
        pass
    return math.nan, ""


def perturbation_sort_key(label: str) -> Tuple[str, float, str, str]:
    drug = perturbation_to_drug_name(label)
    dose, unit = parse_perturbation_dose(label)
    try:
        dose_num = float(dose)
    except Exception:
        dose_num = math.inf
    return (normalize_name(drug), dose_num, str(unit), str(label))


def build_perturbation_index(converter, *, include_control: bool = False) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Build normalized base-drug -> perturbation labels from converter labels.

    Returns
    -------
    label_index:
        normalized drug name -> sorted exact perturbation labels
    canonical_names:
        normalized drug name -> canonical base drug name from the converter
    """
    labels = converter.list_perturbations(include_control=include_control)
    label_index: Dict[str, List[str]] = {}
    canonical_names: Dict[str, str] = {}

    for label in labels:
        label = str(label)
        if not include_control and is_control_like_label(label):
            continue
        drug = perturbation_to_drug_name(label)
        key = normalize_name(drug)
        label_index.setdefault(key, []).append(label)
        canonical_names.setdefault(key, drug)

    for key in list(label_index):
        label_index[key] = sorted(label_index[key], key=perturbation_sort_key)

    return label_index, canonical_names


def resolve_drug_key(
    query: str,
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
) -> str:
    """Resolve a CLI/metadata drug name to a converter drug key."""
    q = normalize_name(query)
    if q in label_index:
        return q

    q_loose = loose_name(query)
    loose_matches = [k for k in label_index if loose_name(canonical_names.get(k, k)) == q_loose]
    if len(loose_matches) == 1:
        return loose_matches[0]

    contains_matches = [
        k
        for k in label_index
        if q in k or q_loose in loose_name(canonical_names.get(k, k))
    ]
    suggestions = sorted(canonical_names.get(k, k) for k in contains_matches[:20])
    if not suggestions:
        suggestions = sorted(canonical_names.get(k, k) for k in list(label_index)[:20])
    raise KeyError(
        f"Could not resolve drug {query!r} to a converter perturbation drug. "
        f"Example available drugs: {suggestions}"
    )


def load_drug_metadata(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Drug metadata not found: {path}")

    df = pd.read_csv(path)
    required = {"drug", "moa-fine"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    df = df.copy()
    df["drug_norm"] = df["drug"].apply(normalize_name)
    df["moa_fine_clean"] = df["moa-fine"].fillna("unknown").astype(str).str.strip()
    bad = df["moa_fine_clean"].eq("") | df["moa_fine_clean"].str.lower().isin(["nan", "none", "null"])
    df.loc[bad, "moa_fine_clean"] = "unknown"
    df["moa_norm"] = df["moa_fine_clean"].apply(normalize_name)
    return df


def build_drug_to_moa_map(drug_meta: pd.DataFrame) -> Dict[str, str]:
    """Map normalized drug name to normalized moa-fine term."""
    out: Dict[str, str] = {}
    for _, row in drug_meta.iterrows():
        out.setdefault(str(row["drug_norm"]), str(row["moa_norm"]))
    return out


def build_drug_to_moa_display_map(drug_meta: pd.DataFrame) -> Dict[str, str]:
    """Map normalized drug name to display moa-fine term."""
    out: Dict[str, str] = {}
    for _, row in drug_meta.iterrows():
        out.setdefault(str(row["drug_norm"]), str(row["moa_fine_clean"]))
    return out


def moa_mask(drug_meta: pd.DataFrame, term: str, mode: str = "exact") -> pd.Series:
    term_norm = normalize_name(term)
    if mode == "exact":
        return drug_meta["moa_norm"] == term_norm
    if mode == "contains":
        return drug_meta["moa_norm"].str.contains(re.escape(term_norm), regex=True, na=False)
    raise ValueError("moa_match_mode must be 'exact' or 'contains'")


def make_pair_id(first_drug: str, second_drug: str) -> str:
    return f"{first_drug} + {second_drug}"


def metadata_moa_for_drug(drug_norm: str, display_map: Dict[str, str]) -> str:
    return display_map.get(drug_norm, "metadata_missing")


def make_ordered_pair_spec(
    *,
    group: str,
    first_key: str,
    second_key: str,
    canonical_names: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
    source: str,
) -> Dict[str, Any]:
    first_drug = canonical_names[first_key]
    second_drug = canonical_names[second_key]
    return {
        "group": group,
        "pair_id": make_pair_id(first_drug, second_drug),
        "first_drug": first_drug,
        "second_drug": second_drug,
        "first_drug_norm": first_key,
        "second_drug_norm": second_key,
        "first_moa_fine": metadata_moa_for_drug(first_key, drug_to_moa_display),
        "second_moa_fine": metadata_moa_for_drug(second_key, drug_to_moa_display),
        "source": source,
    }


def build_explicit_pair_specs(
    two_drug_pair: Sequence[str],
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
) -> List[Dict[str, Any]]:
    if len(two_drug_pair) != 2:
        raise ValueError(f"two_drug_pair must contain exactly two drug names, got {two_drug_pair!r}")
    first_key = resolve_drug_key(str(two_drug_pair[0]), label_index, canonical_names)
    second_key = resolve_drug_key(str(two_drug_pair[1]), label_index, canonical_names)
    if first_key == second_key:
        raise ValueError("two_drug_pair cannot use the same base drug twice")

    specs = []
    for order_index, (ordered_first, ordered_second) in enumerate(
        [(first_key, second_key), (second_key, first_key)],
        start=1,
    ):
        spec = make_ordered_pair_spec(
            group="explicit_pair",
            first_key=ordered_first,
            second_key=ordered_second,
            canonical_names=canonical_names,
            drug_to_moa_display=drug_to_moa_display,
            source="explicit_2drug_pair_order_search",
        )
        spec.update(
            {
                "explicit_input_first_drug_norm": first_key,
                "explicit_input_second_drug_norm": second_key,
                "explicit_order_index": int(order_index),
                "explicit_order": "input_order" if order_index == 1 else "reverse_input_order",
                "order_search_id": make_pair_id(canonical_names[first_key], canonical_names[second_key]),
            }
        )
        specs.append(spec)
    return specs


def _explicit_pair_record_value(record: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(record, dict):
        for name in names:
            if name in record and pd.notna(record[name]):
                return record[name]
    return default


def build_explicit_pair_panel_specs(
    explicit_drug_pairs: Sequence[Any],
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Build ordered candidate specs for a panel of explicit 2-drug pairs.

    Each input pair contributes two ordered candidates. The user-facing pair_id
    is kept stable across both orders so concentration/order selection retains
    one best ordered version per input pair.
    """
    specs: List[Dict[str, Any]] = []
    seen_pair_ids: set[str] = set()
    for panel_index, record in enumerate(explicit_drug_pairs, start=1):
        if isinstance(record, (list, tuple)):
            if len(record) == 2:
                pair_id_raw = None
                first_query, second_query = record
            elif len(record) >= 3:
                pair_id_raw, first_query, second_query = record[:3]
            else:
                raise ValueError(f"Explicit panel pair record must have 2 or 3 values, got {record!r}")
        else:
            pair_id_raw = _explicit_pair_record_value(record, ("pair_id", "id", "name", "label"))
            first_query = _explicit_pair_record_value(record, ("drug_a", "first_drug", "drug1", "drug_a_name", "a"))
            second_query = _explicit_pair_record_value(record, ("drug_b", "second_drug", "drug2", "drug_b_name", "b"))
        pair_group = _explicit_pair_record_value(
            record,
            ("pair_group", "pair_cohort", "cohort", "clinical_status", "category"),
        )
        if first_query is None or second_query is None:
            raise ValueError(
                "Each explicit panel pair must provide drug_a and drug_b "
                f"(or first_drug/second_drug); bad record: {record!r}"
            )

        first_key = resolve_drug_key(str(first_query), label_index, canonical_names)
        second_key = resolve_drug_key(str(second_query), label_index, canonical_names)
        if first_key == second_key:
            raise ValueError(f"Explicit panel pair {record!r} uses the same base drug twice")

        default_pair_id = make_pair_id(canonical_names[first_key], canonical_names[second_key])
        pair_id = str(pair_id_raw).strip() if pair_id_raw is not None and str(pair_id_raw).strip() else default_pair_id
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate explicit panel pair_id {pair_id!r}; pair IDs must be unique")
        seen_pair_ids.add(pair_id)

        for order_index, (ordered_first, ordered_second) in enumerate(
            [(first_key, second_key), (second_key, first_key)],
            start=1,
        ):
            spec = make_ordered_pair_spec(
                group="explicit_pair",
                first_key=ordered_first,
                second_key=ordered_second,
                canonical_names=canonical_names,
                drug_to_moa_display=drug_to_moa_display,
                source="explicit_pair_panel_order_search",
            )
            ordered_pair_id = str(spec["pair_id"])
            spec.update(
                {
                    "pair_id": pair_id,
                    "ordered_pair_id": ordered_pair_id,
                    "explicit_panel_pair_id": pair_id,
                    "explicit_panel_index": int(panel_index),
                    "explicit_input_first_drug_norm": first_key,
                    "explicit_input_second_drug_norm": second_key,
                    "explicit_input_first_drug": canonical_names[first_key],
                    "explicit_input_second_drug": canonical_names[second_key],
                    "explicit_order_index": int(order_index),
                    "explicit_order": "input_order" if order_index == 1 else "reverse_input_order",
                    "order_search_id": pair_id,
                }
            )
            if pair_group is not None and str(pair_group).strip():
                spec["pair_group"] = str(pair_group).strip()
            specs.append(spec)
    return specs


def select_best_explicit_order_and_concentration(selection_scores: pd.DataFrame) -> pd.DataFrame:
    """Choose the best explicit ordered pair and concentration combination."""
    if selection_scores.empty:
        return selection_scores.copy()

    sort_cols = ["score_sinkhorn_ot", "score_energy_distance", "explicit_order_index", "pair_id"]
    d = selection_scores.sort_values(sort_cols, ascending=[True, True, True, True]).copy()
    selected = d.head(1).copy()
    selected["n_concentration_combinations_scored"] = int(len(d))
    selected["n_ordered_drug_orders_scored"] = int(
        d[["first_drug_norm", "second_drug_norm"]].drop_duplicates().shape[0]
    )
    selected = selected.rename(
        columns={
            "score_sinkhorn_ot": "selection_score_sinkhorn_ot",
            "score_energy_distance": "selection_score_energy_distance",
            "batch_index": "selection_batch_index",
        }
    )
    selected["selected_by"] = "min_selection_batch_sinkhorn_ot_across_orders"
    return selected.reset_index(drop=True)


def select_best_explicit_panel_orders_and_concentrations(selection_scores: pd.DataFrame) -> pd.DataFrame:
    """Choose one best ordered concentration combination for each explicit panel pair."""
    if selection_scores.empty:
        return selection_scores.copy()

    group_cols = ["group", "pair_id"]
    sort_cols = group_cols + ["score_sinkhorn_ot", "score_energy_distance", "explicit_order_index", "ordered_pair_id"]
    d = selection_scores.sort_values(sort_cols, ascending=[True, True, True, True, True, True]).copy()
    selected = d.groupby(group_cols, as_index=False, sort=False).head(1).copy()

    combo_counts = d.groupby(group_cols).size().rename("n_concentration_combinations_scored").reset_index()
    order_counts = (
        d.groupby(group_cols)[["first_drug_norm", "second_drug_norm"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("n_ordered_drug_orders_scored")
        .reset_index()
    )
    selected = selected.merge(combo_counts, on=group_cols, how="left")
    selected = selected.merge(order_counts, on=group_cols, how="left")
    selected = selected.rename(
        columns={
            "score_sinkhorn_ot": "selection_score_sinkhorn_ot",
            "score_energy_distance": "selection_score_energy_distance",
            "batch_index": "selection_batch_index",
        }
    )
    selected["selected_by"] = "min_selection_batch_sinkhorn_ot_within_explicit_pair_panel"
    return selected.sort_values(["selection_score_sinkhorn_ot", "pair_id"]).reset_index(drop=True)


def select_best_random_orders_and_concentrations(selection_scores: pd.DataFrame) -> pd.DataFrame:
    """Choose one best ordered concentration combination for each random drug pair."""
    if selection_scores.empty:
        return selection_scores.copy()

    group_cols = ["group", "pair_id"]
    sort_cols = group_cols + ["score_sinkhorn_ot", "score_energy_distance", "random_order_index"]
    d = selection_scores.sort_values(sort_cols, ascending=[True, True, True, True, True]).copy()
    selected = d.groupby(group_cols, as_index=False, sort=False).head(1).copy()

    combo_counts = d.groupby(group_cols).size().rename("n_concentration_combinations_scored").reset_index()
    order_counts = (
        d.groupby(group_cols)[["first_drug_norm", "second_drug_norm"]]
        .apply(lambda x: x.drop_duplicates().shape[0])
        .rename("n_ordered_drug_orders_scored")
        .reset_index()
    )
    selected = selected.merge(combo_counts, on=group_cols, how="left")
    selected = selected.merge(order_counts, on=group_cols, how="left")
    selected = selected.rename(
        columns={
            "score_sinkhorn_ot": "selection_score_sinkhorn_ot",
            "score_energy_distance": "selection_score_energy_distance",
            "batch_index": "selection_batch_index",
        }
    )
    selected["selected_by"] = "min_selection_batch_sinkhorn_ot_within_random_pair"
    return selected.sort_values(["random_pair_index", "pair_id"]).reset_index(drop=True)


def moa_pairs_for_explicit_order(
    moa_pairs: Sequence[str],
    explicit_selected: pd.DataFrame,
) -> Tuple[List[str], str]:
    """Align the requested MOA terms to the winning explicit drug order."""
    if len(moa_pairs) != 2 or explicit_selected.empty:
        return list(map(str, moa_pairs)), "input_order"

    row = explicit_selected.iloc[0]
    input_first = str(row.get("explicit_input_first_drug_norm", ""))
    input_second = str(row.get("explicit_input_second_drug_norm", ""))
    selected_first = str(row.get("first_drug_norm", ""))
    selected_second = str(row.get("second_drug_norm", ""))

    if selected_first == input_first and selected_second == input_second:
        return [str(moa_pairs[0]), str(moa_pairs[1])], "input_order"
    if selected_first == input_second and selected_second == input_first:
        return [str(moa_pairs[1]), str(moa_pairs[0])], "reverse_input_order"
    return list(map(str, moa_pairs)), "input_order"


def available_moa_drug_keys(
    drug_meta: pd.DataFrame,
    moa_term: str,
    *,
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    moa_match_mode: str = "exact",
) -> List[str]:
    mask = moa_mask(drug_meta, moa_term, mode=moa_match_mode)
    keys: List[str] = []
    seen = set()
    for _, row in drug_meta.loc[mask].iterrows():
        key = str(row["drug_norm"])
        if key not in label_index:
            # Metadata spelling can differ slightly from the converter. Try the
            # same resolver used for CLI names before excluding it.
            try:
                key = resolve_drug_key(str(row["drug"]), label_index, canonical_names)
            except KeyError:
                continue
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return sorted(keys, key=lambda k: canonical_names.get(k, k).casefold())


def build_fixed_drug_moa_explicit_pair_specs(
    fixed_drug: str,
    partner_moa_term: str,
    drug_meta: pd.DataFrame,
    *,
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
    moa_match_mode: str = "exact",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build explicit-pair candidates by pairing one fixed drug with every
    converter-available drug matching the requested partner MOA.
    """
    fixed_key = resolve_drug_key(str(fixed_drug), label_index, canonical_names)
    partner_keys = available_moa_drug_keys(
        drug_meta,
        str(partner_moa_term),
        label_index=label_index,
        canonical_names=canonical_names,
        moa_match_mode=moa_match_mode,
    )
    partner_keys = [key for key in partner_keys if key != fixed_key]

    if not partner_keys:
        raise ValueError(
            "No converter-available partner drugs remain for fixed explicit search "
            f"after matching MOA term {partner_moa_term!r} and excluding the fixed drug {fixed_drug!r}."
        )

    specs: List[Dict[str, Any]] = []
    for candidate_index, partner_key in enumerate(partner_keys, start=1):
        for order_index, (ordered_first, ordered_second) in enumerate(
            [(fixed_key, partner_key), (partner_key, fixed_key)],
            start=1,
        ):
            spec = make_ordered_pair_spec(
                group="explicit_pair",
                first_key=ordered_first,
                second_key=ordered_second,
                canonical_names=canonical_names,
                drug_to_moa_display=drug_to_moa_display,
                source="fixed_drug_moa_explicit_pair_search",
            )
            spec.update(
                {
                    "explicit_search_mode": "fixed_drug_second_moa_scan",
                    "explicit_fixed_drug_norm": fixed_key,
                    "explicit_fixed_drug": canonical_names[fixed_key],
                    "explicit_partner_moa_term": str(partner_moa_term),
                    "explicit_partner_candidate_index": int(candidate_index),
                    "explicit_input_first_drug_norm": fixed_key,
                    "explicit_input_second_drug_norm": partner_key,
                    "explicit_order_index": int(order_index),
                    "explicit_order": "input_order" if order_index == 1 else "reverse_input_order",
                    "order_search_id": make_pair_id(canonical_names[fixed_key], canonical_names[partner_key]),
                }
            )
            specs.append(spec)

    metadata = {
        "fixed_drug": canonical_names[fixed_key],
        "fixed_drug_norm": fixed_key,
        "partner_moa_term": str(partner_moa_term),
        "partner_moa_drugs_available": [canonical_names[k] for k in partner_keys],
        "n_partner_moa_drugs_available": len(partner_keys),
        "n_ordered_explicit_pairs_used": len(specs),
        "moa_match_mode": moa_match_mode,
    }
    return specs, metadata


def build_moa_pair_specs(
    moa_pairs: Sequence[str],
    drug_meta: pd.DataFrame,
    *,
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
    explicit_pair_keys: Sequence[str] = (),
    include_explicit_in_moa: bool = False,
    moa_match_mode: str = "exact",
    max_moa_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(moa_pairs) != 2:
        raise ValueError(f"moa_pairs must contain exactly two moa-fine terms, got {moa_pairs!r}")

    first_keys = available_moa_drug_keys(
        drug_meta,
        str(moa_pairs[0]),
        label_index=label_index,
        canonical_names=canonical_names,
        moa_match_mode=moa_match_mode,
    )
    second_keys = available_moa_drug_keys(
        drug_meta,
        str(moa_pairs[1]),
        label_index=label_index,
        canonical_names=canonical_names,
        moa_match_mode=moa_match_mode,
    )

    if not first_keys:
        raise ValueError(f"No converter-available drugs found for first MOA term: {moa_pairs[0]!r}")
    if not second_keys:
        raise ValueError(f"No converter-available drugs found for second MOA term: {moa_pairs[1]!r}")

    specs: List[Dict[str, Any]] = []
    explicit_pair_keys = explicit_pair_keys or ()
    explicit_tuple = tuple(explicit_pair_keys)
    explicit_tuple = explicit_tuple if len(explicit_tuple) == 2 else None
    for first_key in first_keys:
        for second_key in second_keys:
            if first_key == second_key:
                continue
            if explicit_tuple is not None and not include_explicit_in_moa and (first_key, second_key) == explicit_tuple:
                continue
            specs.append(
                make_ordered_pair_spec(
                    group="moa_pair",
                    first_key=first_key,
                    second_key=second_key,
                    canonical_names=canonical_names,
                    drug_to_moa_display=drug_to_moa_display,
                    source="moa_pair",
                )
            )

    full_count = len(specs)
    if max_moa_pairs is not None and len(specs) > int(max_moa_pairs):
        if rng is None:
            rng = np.random.default_rng(0)
        chosen = rng.choice(len(specs), size=int(max_moa_pairs), replace=False)
        specs = [specs[int(i)] for i in sorted(chosen)]

    if not specs:
        raise ValueError("No MOA pair candidates remain after filtering.")

    metadata = {
        "first_moa_term": str(moa_pairs[0]),
        "second_moa_term": str(moa_pairs[1]),
        "first_moa_drugs_available": [canonical_names[k] for k in first_keys],
        "second_moa_drugs_available": [canonical_names[k] for k in second_keys],
        "n_first_moa_drugs_available": len(first_keys),
        "n_second_moa_drugs_available": len(second_keys),
        "n_ordered_moa_pairs_full": full_count,
        "n_ordered_moa_pairs_used": len(specs),
        "include_explicit_in_moa": bool(include_explicit_in_moa),
        "moa_match_mode": moa_match_mode,
    }
    return specs, metadata


def expand_concentration_candidates(
    pair_specs: Sequence[Dict[str, Any]],
    label_index: Dict[str, List[str]],
) -> pd.DataFrame:
    """Expand base-drug pair specs into all ordered concentration-label pairs."""
    rows: List[Dict[str, Any]] = []
    for spec in pair_specs:
        first_labels = label_index[str(spec["first_drug_norm"])]
        second_labels = label_index[str(spec["second_drug_norm"])]
        for first_label in first_labels:
            first_dose, first_unit = parse_perturbation_dose(first_label)
            for second_label in second_labels:
                second_dose, second_unit = parse_perturbation_dose(second_label)
                row = dict(spec)
                row.update(
                    {
                        "first_perturbation": str(first_label),
                        "second_perturbation": str(second_label),
                        "first_dose": first_dose,
                        "first_dose_unit": first_unit,
                        "second_dose": second_dose,
                        "second_dose_unit": second_unit,
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


# setup_projection_and_pools now lives in projections.py (a neutral module) so the
# search workflow can reuse it. Keep this lazy wrapper for backward compatibility
# without importing the heavier projection stack during table-only tests.
def setup_projection_and_pools(*args, **kwargs):
    from ..projections import setup_projection_and_pools as _setup_projection_and_pools

    return _setup_projection_and_pools(*args, **kwargs)


def _make_scorer(target_embeddings, scoring_params: ScoringParams, device: str | torch.device, projection=None):
    from ..scoring import DistributionScorer

    return DistributionScorer(
        target_state=target_embeddings,
        device=device,
        normalize=scoring_params.normalize,
        sinkhorn_metric=scoring_params.sinkhorn_metric,
        sinkhorn_epsilon=scoring_params.sinkhorn_epsilon,
        sinkhorn_iters=scoring_params.sinkhorn_iters,
        projection=projection,
        projection_auto_metric=True,
        projection_auto_epsilon=bool(projection is not None and scoring_params.projection_auto_epsilon),
    )


def score_ordered_label_pairs(
    *,
    converter,
    scorer,
    start_embeddings,
    pair_df: pd.DataFrame,
    chunk_size: int = 16,
    batch_index: int = 0,
    score_context: str = "evaluation",
) -> pd.DataFrame:
    """
    Apply first perturbation then second perturbation for each row in pair_df.

    pair_df must include first_perturbation and second_perturbation columns.
    Other columns are copied through to the returned score table.
    """
    required = {"first_perturbation", "second_perturbation"}
    missing = required - set(pair_df.columns)
    if missing:
        raise ValueError(f"pair_df missing required columns: {sorted(missing)}")
    if pair_df.empty:
        return pair_df.copy()

    rows: List[Dict[str, Any]] = []
    chunk_size = max(1, int(chunk_size))

    # Preserve first-label encounter order for reproducible progress and output.
    first_labels = list(dict.fromkeys(pair_df["first_perturbation"].astype(str).tolist()))

    for first_i, first_label in enumerate(first_labels, start=1):
        sub = pair_df[pair_df["first_perturbation"].astype(str) == first_label].copy()
        print(
            f"  {score_context}: first drug label {first_i}/{len(first_labels)} "
            f"with {len(sub)} second-label candidates"
        )

        first_state = converter.convert_one(start_embeddings, first_label, return_cpu=False)
        sub_indices = sub.index.tolist()
        second_labels = sub["second_perturbation"].astype(str).tolist()
        offset = 0

        for labels, pred_batch in converter.convert_many_iter(
            first_state,
            perturbations=second_labels,
            chunk_size=chunk_size,
            return_cpu=False,
        ):
            n = len(labels)
            row_indices = sub_indices[offset : offset + n]
            offset += n

            sinkhorn_scores = scorer.sinkhorn(pred_batch).detach().cpu().numpy()
            energy_scores = scorer.energy_distance(pred_batch).detach().cpu().numpy()

            for row_idx, second_label, sinkhorn_score, energy_score in zip(
                row_indices,
                labels,
                sinkhorn_scores,
                energy_scores,
            ):
                base = pair_df.loc[row_idx].to_dict()
                base.update(
                    {
                        "batch_index": int(batch_index),
                        "score_context": score_context,
                        "first_perturbation": str(first_label),
                        "second_perturbation": str(second_label),
                        "score_sinkhorn_ot": float(sinkhorn_score),
                        "score_energy_distance": float(energy_score),
                    }
                )
                rows.append(base)

            del pred_batch

        del first_state

    return pd.DataFrame(rows)


def select_best_concentrations(selection_scores: pd.DataFrame) -> pd.DataFrame:
    """Choose the minimum Sinkhorn concentration pair per ordered base-drug pair."""
    if selection_scores.empty:
        return selection_scores.copy()

    sort_cols = ["group", "pair_id", "score_sinkhorn_ot", "score_energy_distance"]
    d = selection_scores.sort_values(sort_cols, ascending=[True, True, True, True]).copy()
    selected = d.groupby(["group", "pair_id"], as_index=False, sort=False).head(1).copy()
    combo_counts = d.groupby(["group", "pair_id"]).size().rename("n_concentration_combinations_scored").reset_index()
    selected = selected.merge(combo_counts, on=["group", "pair_id"], how="left")
    selected = selected.rename(
        columns={
            "score_sinkhorn_ot": "selection_score_sinkhorn_ot",
            "score_energy_distance": "selection_score_energy_distance",
            "batch_index": "selection_batch_index",
        }
    )
    selected["selected_by"] = "min_selection_batch_sinkhorn_ot"
    selected = selected.sort_values(["group", "selection_score_sinkhorn_ot", "pair_id"]).reset_index(drop=True)
    return selected


def resolve_control_perturbation(converter) -> Optional[str]:
    """
    Find a vehicle/DMSO control perturbation label in the converter.

    Prefers a control-like label (dmso/control/non-targeting) with zero numeric
    dose. Returns None if no control label is available.
    """
    labels = [str(p) for p in converter.list_perturbations(include_control=True)]
    control_like = [p for p in labels if is_control_like_label(p)]
    if not control_like:
        return None

    def _has_zero_dose(label: str) -> bool:
        m = re.search(r",\s*([0-9eE+\-.]+)\s*,", label)
        if not m:
            return False
        try:
            return float(m.group(1)) == 0.0
        except ValueError:
            return False

    zero_dose = [p for p in control_like if _has_zero_dose(p)]
    if zero_dose:
        return sorted(zero_dose, key=perturbation_sort_key)[0]
    return sorted(control_like, key=perturbation_sort_key)[0]


def compute_baselines(
    batches: Sequence[Any],
    *,
    scoring_params: ScoringParams,
    device: str | torch.device,
    projection=None,
) -> pd.DataFrame:
    """Raw baseline: real (untransformed) start embeddings vs target, per batch."""
    rows: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        scorer = _make_scorer(batch.target_embeddings, scoring_params, device, projection=projection)
        start_state = torch.as_tensor(batch.start_embeddings, dtype=torch.float32, device=scorer.device)
        sinkhorn = float(scorer.sinkhorn(start_state).item())
        energy = float(scorer.energy_distance(start_state).item())
        rows.append(
            {
                "group": "baseline",
                "pair_id": "start_vs_target",
                "batch_index": int(batch_index),
                "seed": int(batch.seed),
                "score_sinkhorn_ot": sinkhorn,
                "score_energy_distance": energy,
                "start_n_available": int(batch.start_n_available),
                "target_n_available": int(batch.target_n_available),
                "start_n_sampled": int(batch.start_n_sampled),
                "target_n_sampled": int(batch.target_n_sampled),
                "replace_start": bool(batch.replace_start),
                "replace_target": bool(batch.replace_target),
            }
        )
    return pd.DataFrame(rows)


def compute_control_baselines(
    batches: Sequence[Any],
    *,
    converter,
    control_label: str,
    scoring_params: ScoringParams,
    device: str | torch.device,
    projection=None,
) -> pd.DataFrame:
    """
    "Null perturbation" baselines passed through ST-SE with the control label:
      - baseline_stse_control_1pass: ST-SE(start, control)
      - baseline_stse_control_2pass: ST-SE(ST-SE(start, control), control)

    The 2-pass variant matches the two sequential applications used for scored
    drug pairs, isolating any systematic ST-SE pass-through shift from the drug
    effect itself. Diagnostic only: not used for gain-vs-baseline lookups.
    """
    rows: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        scorer = _make_scorer(batch.target_embeddings, scoring_params, device, projection=projection)
        start_state = torch.as_tensor(batch.start_embeddings, dtype=torch.float32, device=converter.device)

        common = {
            "pair_id": "start_vs_target",
            "batch_index": int(batch_index),
            "seed": int(batch.seed),
            "start_n_sampled": int(batch.start_n_sampled),
            "target_n_sampled": int(batch.target_n_sampled),
        }

        ctrl_1 = converter.convert_one(start_state, control_label, return_cpu=False)
        rows.append(
            {
                "group": "baseline_stse_control_1pass",
                "score_sinkhorn_ot": float(scorer.sinkhorn(ctrl_1).item()),
                "score_energy_distance": float(scorer.energy_distance(ctrl_1).item()),
                "control_label": str(control_label),
                **common,
            }
        )
        ctrl_2 = converter.convert_one(ctrl_1, control_label, return_cpu=False)
        rows.append(
            {
                "group": "baseline_stse_control_2pass",
                "score_sinkhorn_ot": float(scorer.sinkhorn(ctrl_2).item()),
                "score_energy_distance": float(scorer.energy_distance(ctrl_2).item()),
                "control_label": str(control_label),
                **common,
            }
        )
        del ctrl_1, ctrl_2

    return pd.DataFrame(rows)


def selected_pairs_for_scoring(selected_pairs: pd.DataFrame) -> pd.DataFrame:
    """Rename selected score columns away from the active score columns."""
    keep = selected_pairs.copy()
    drop_cols = [c for c in ["score_context"] if c in keep.columns]
    keep = keep.drop(columns=drop_cols, errors="ignore")
    return keep


def evaluate_selected_pairs_across_batches(
    *,
    converter,
    batches: Sequence[Any],
    selected_pairs: pd.DataFrame,
    baseline_df: pd.DataFrame,
    scoring_params: ScoringParams,
    device: str | torch.device,
    chunk_size: int,
    projection=None,
    score_context: str = "selected_pair_evaluation",
    progress_label: str = "selected explicit/MOA pairs",
) -> pd.DataFrame:
    eval_rows: List[pd.DataFrame] = []
    base_by_batch = baseline_df.set_index("batch_index")
    pair_df = selected_pairs_for_scoring(selected_pairs)

    for batch_index, batch in enumerate(batches):
        print(f"Evaluating {progress_label} on batch {batch_index + 1}/{len(batches)}")
        scorer = _make_scorer(batch.target_embeddings, scoring_params, device, projection=projection)
        scored = score_ordered_label_pairs(
            converter=converter,
            scorer=scorer,
            start_embeddings=batch.start_embeddings,
            pair_df=pair_df,
            chunk_size=chunk_size,
            batch_index=batch_index,
            score_context=score_context,
        )
        baseline_sinkhorn = float(base_by_batch.loc[batch_index, "score_sinkhorn_ot"])
        baseline_energy = float(base_by_batch.loc[batch_index, "score_energy_distance"])
        scored["baseline_sinkhorn_ot"] = baseline_sinkhorn
        scored["baseline_energy_distance"] = baseline_energy
        scored["delta_sinkhorn_from_baseline"] = scored["score_sinkhorn_ot"].astype(float) - baseline_sinkhorn
        scored["delta_energy_from_baseline"] = scored["score_energy_distance"].astype(float) - baseline_energy
        eval_rows.append(scored)

    if not eval_rows:
        return pd.DataFrame()
    return pd.concat(eval_rows, ignore_index=True)


def evaluate_explicit_pair_additive_interaction(
    *,
    converter,
    batches: Sequence[Any],
    explicit_selected: pd.DataFrame,
    baseline_df: pd.DataFrame,
    scoring_params: ScoringParams,
    device: str | torch.device,
    projection=None,
) -> pd.DataFrame:
    """
    Evaluate additive-delta and order-sensitive sequential variants for the
    explicit 2-drug pair using its selected concentration labels.

    Additive-delta state:
        x_additive = x_start + (x_A - x_start) + (x_B - x_start)
                   = x_A + x_B - x_start
    """
    explicit = explicit_selected[explicit_selected["group"] == "explicit_pair"].copy()
    pair_source = "explicit_pair"
    if explicit.empty:
        explicit = explicit_selected[explicit_selected["group"] == "moa_pair"].copy()
        pair_source = "best_moa_pair"
        if not explicit.empty:
            sort_col = "eval_mean_sinkhorn_ot" if "eval_mean_sinkhorn_ot" in explicit.columns else "selection_score_sinkhorn_ot"
            explicit = explicit.sort_values([sort_col, "pair_id"], ascending=[True, True])

    if explicit.empty:
        return pd.DataFrame()

    row = explicit.iloc[0]
    first_label = str(row["first_perturbation"])
    second_label = str(row["second_perturbation"])
    first_drug = str(row["first_drug"])
    second_drug = str(row["second_drug"])
    pair_id = str(row["pair_id"])
    base_by_batch = baseline_df.set_index("batch_index")

    rows: List[Dict[str, Any]] = []

    def append_score(
        *,
        batch_index: int,
        seed: int,
        scorer,
        state: torch.Tensor,
        mode: str,
        mode_label: str,
        mode_order: int,
    ) -> None:
        baseline_sinkhorn = float(base_by_batch.loc[batch_index, "score_sinkhorn_ot"])
        baseline_energy = float(base_by_batch.loc[batch_index, "score_energy_distance"])
        sinkhorn = float(scorer.sinkhorn(state).item())
        energy = float(scorer.energy_distance(state).item())
        rows.append(
            {
                "group": "explicit_pair_additive",
                "pair_id": pair_id,
                "batch_index": int(batch_index),
                "seed": int(seed),
                "mode": mode,
                "mode_label": mode_label,
                "mode_order": int(mode_order),
                "first_drug": first_drug,
                "second_drug": second_drug,
                "first_perturbation": first_label,
                "second_perturbation": second_label,
                "source_pair_group": str(row.get("group", "")),
                "source_pair_selection": pair_source,
                "score_sinkhorn_ot": sinkhorn,
                "score_energy_distance": energy,
                "baseline_sinkhorn_ot": baseline_sinkhorn,
                "baseline_energy_distance": baseline_energy,
                "delta_sinkhorn_from_baseline": sinkhorn - baseline_sinkhorn,
                "delta_energy_from_baseline": energy - baseline_energy,
                "gain_sinkhorn_vs_baseline": baseline_sinkhorn - sinkhorn,
                "gain_energy_vs_baseline": baseline_energy - energy,
            }
        )

    for batch_index, batch in enumerate(batches):
        print(f"Evaluating explicit additive/interaction variants on batch {batch_index + 1}/{len(batches)}")
        scorer = _make_scorer(batch.target_embeddings, scoring_params, device, projection=projection)
        start_state = torch.as_tensor(batch.start_embeddings, dtype=torch.float32, device=scorer.device)

        state_a = converter.convert_one(start_state, first_label, return_cpu=False)
        state_b = converter.convert_one(start_state, second_label, return_cpu=False)
        state_additive = state_a + state_b - start_state
        state_seq_ab = converter.convert_one(state_a, second_label, return_cpu=False)
        state_seq_ba = converter.convert_one(state_b, first_label, return_cpu=False)

        append_score(
            batch_index=batch_index,
            seed=batch.seed,
            scorer=scorer,
            state=state_a,
            mode="single_A",
            mode_label=f"Single A\n{first_drug}",
            mode_order=1,
        )
        append_score(
            batch_index=batch_index,
            seed=batch.seed,
            scorer=scorer,
            state=state_b,
            mode="single_B",
            mode_label=f"Single B\n{second_drug}",
            mode_order=2,
        )
        append_score(
            batch_index=batch_index,
            seed=batch.seed,
            scorer=scorer,
            state=state_additive,
            mode="additive_A_plus_B",
            mode_label=f"Additive\n{first_drug} + {second_drug}",
            mode_order=3,
        )
        append_score(
            batch_index=batch_index,
            seed=batch.seed,
            scorer=scorer,
            state=state_seq_ab,
            mode="sequential_A_to_B",
            mode_label=f"Sequential\n{first_drug} -> {second_drug}",
            mode_order=4,
        )
        append_score(
            batch_index=batch_index,
            seed=batch.seed,
            scorer=scorer,
            state=state_seq_ba,
            mode="sequential_B_to_A",
            mode_label=f"Sequential\n{second_drug} -> {first_drug}",
            mode_order=5,
        )

        del state_a, state_b, state_additive, state_seq_ab, state_seq_ba, start_state

    return pd.DataFrame(rows)


def random_pair_candidates(
    *,
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    drug_to_moa_norm: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
    explicit_pair_keys: Sequence[str] = (),
    moa_terms: Sequence[str],
    n_pairs: int,
    rng: np.random.Generator,
    allow_random_metadata_missing: bool = False,
    max_enumerate_pairs: int = 200_000,
) -> pd.DataFrame:
    """
    Sample ordered random base-drug pairs, then one random concentration label
    for each drug in the pair.

    Exclusions:
      - same base drug twice
      - either base drug in the explicit 2-drug pair
      - either drug has moa-fine matching one of the requested MOA terms
      - metadata-missing drugs, unless allow_random_metadata_missing=True
    """
    if n_pairs <= 0:
        return pd.DataFrame()

    explicit_set = set(explicit_pair_keys or [])
    excluded_moas = {normalize_name(x) for x in moa_terms}

    eligible_drugs: List[str] = []
    for drug_key, labels in label_index.items():
        if drug_key in explicit_set:
            continue
        if not labels:
            continue
        moa = drug_to_moa_norm.get(drug_key)
        if moa is None and not allow_random_metadata_missing:
            continue
        if moa is not None and moa in excluded_moas:
            continue
        eligible_drugs.append(drug_key)

    if len(eligible_drugs) < 2:
        raise ValueError("Fewer than two random-control drugs are eligible after filtering.")

    n_drugs = len(eligible_drugs)
    possible_pairs = n_drugs * (n_drugs - 1)
    if possible_pairs <= 0:
        raise ValueError("No random-control pairs are possible after filtering.")

    if n_pairs > possible_pairs:
        print(f"Requested {n_pairs} random drug pairs but only {possible_pairs} are possible; using all possible pairs.")
        n_pairs = int(possible_pairs)

    chosen_base_pairs: List[Tuple[str, str]] = []
    seen = set()

    if possible_pairs <= max_enumerate_pairs:
        all_pairs: List[Tuple[str, str]] = []
        for first_key in eligible_drugs:
            for second_key in eligible_drugs:
                if first_key == second_key:
                    continue
                all_pairs.append((first_key, second_key))
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        chosen_base_pairs = [all_pairs[int(i)] for i in idx]
    else:
        max_attempts = max(10_000, int(n_pairs) * 200)
        attempts = 0
        while len(chosen_base_pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            first_i = int(rng.integers(0, n_drugs))
            second_i = int(rng.integers(0, n_drugs))
            first_key = eligible_drugs[first_i]
            second_key = eligible_drugs[second_i]
            if first_key == second_key:
                continue
            pair_key = (first_key, second_key)
            if pair_key in seen:
                continue
            seen.add(pair_key)
            chosen_base_pairs.append(pair_key)

        if len(chosen_base_pairs) < n_pairs:
            raise RuntimeError(
                f"Could only sample {len(chosen_base_pairs)} unique random drug pairs after {attempts} attempts; "
                "reduce --random-pairs or loosen random-control filters."
            )

    rows: List[Dict[str, Any]] = []
    for i, (first_key, second_key) in enumerate(chosen_base_pairs, start=1):
        first_label = str(rng.choice(label_index[first_key]))
        second_label = str(rng.choice(label_index[second_key]))
        first_drug = canonical_names[first_key]
        second_drug = canonical_names[second_key]
        first_dose, first_unit = parse_perturbation_dose(first_label)
        second_dose, second_unit = parse_perturbation_dose(second_label)
        rows.append(
            {
                "group": "random_pair",
                "pair_id": f"random_{i:05d}: {make_pair_id(first_drug, second_drug)}",
                "random_pair_index": int(i),
                "first_drug": first_drug,
                "second_drug": second_drug,
                "first_drug_norm": first_key,
                "second_drug_norm": second_key,
                "first_moa_fine": metadata_moa_for_drug(first_key, drug_to_moa_display),
                "second_moa_fine": metadata_moa_for_drug(second_key, drug_to_moa_display),
                "first_perturbation": first_label,
                "second_perturbation": second_label,
                "first_dose": first_dose,
                "first_dose_unit": first_unit,
                "second_dose": second_dose,
                "second_dose_unit": second_unit,
                "source": "random_control",
            }
        )

    return pd.DataFrame(rows)


def matched_random_pair_specs(
    *,
    label_index: Dict[str, List[str]],
    canonical_names: Dict[str, str],
    drug_to_moa_display: Dict[str, str],
    n_pairs: int,
    rng: np.random.Generator,
    max_enumerate_pairs: int = 200_000,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Sample random base-drug pairs, then emit both drug orders for each pair.

    Unlike the legacy random controls, this intentionally applies no
    explicit-pair, MOA, or metadata-missing drug exclusions. The only base-drug
    constraints are converter availability and two distinct drugs per pair.
    """
    eligible_drugs = [drug_key for drug_key, labels in label_index.items() if labels]
    if len(eligible_drugs) < 2:
        raise ValueError("Fewer than two converter-available drugs are eligible for matched random controls.")

    n_drugs = len(eligible_drugs)
    possible_pairs = n_drugs * (n_drugs - 1) // 2
    if n_pairs <= 0:
        return [], {
            "random_type": "matched",
            "n_random_base_drugs_eligible": len(eligible_drugs),
            "n_random_unordered_pairs_possible": int(possible_pairs),
            "n_random_unordered_pairs_sampled": 0,
            "n_random_ordered_pair_specs": 0,
            "random_drug_filtering": "none_except_distinct_converter_available_drugs",
        }

    if n_pairs > possible_pairs:
        print(f"Requested {n_pairs} random drug pairs but only {possible_pairs} are possible; using all possible pairs.")
        n_pairs = int(possible_pairs)

    chosen_base_pairs: List[Tuple[str, str]] = []
    seen: set[Tuple[str, str]] = set()

    if possible_pairs <= max_enumerate_pairs:
        all_pairs: List[Tuple[str, str]] = []
        for first_i, first_key in enumerate(eligible_drugs):
            for second_key in eligible_drugs[first_i + 1 :]:
                all_pairs.append((first_key, second_key))
        idx = rng.choice(len(all_pairs), size=n_pairs, replace=False)
        chosen_base_pairs = [all_pairs[int(i)] for i in idx]
    else:
        max_attempts = max(10_000, int(n_pairs) * 200)
        attempts = 0
        while len(chosen_base_pairs) < n_pairs and attempts < max_attempts:
            attempts += 1
            first_i = int(rng.integers(0, n_drugs))
            second_i = int(rng.integers(0, n_drugs))
            if first_i == second_i:
                continue
            low_i, high_i = sorted((first_i, second_i))
            pair_key = (eligible_drugs[low_i], eligible_drugs[high_i])
            if pair_key in seen:
                continue
            seen.add(pair_key)
            chosen_base_pairs.append(pair_key)

        if len(chosen_base_pairs) < n_pairs:
            raise RuntimeError(
                f"Could only sample {len(chosen_base_pairs)} unique matched random drug pairs after {attempts} attempts; "
                "reduce --random-pairs."
            )

    specs: List[Dict[str, Any]] = []
    for random_pair_index, (first_key, second_key) in enumerate(chosen_base_pairs, start=1):
        first_drug = canonical_names[first_key]
        second_drug = canonical_names[second_key]
        pair_id = f"random_{random_pair_index:05d}: {make_pair_id(first_drug, second_drug)}"
        ordered_pair_id = make_pair_id(first_drug, second_drug)
        for order_index, (ordered_first, ordered_second) in enumerate(
            [(first_key, second_key), (second_key, first_key)],
            start=1,
        ):
            spec = make_ordered_pair_spec(
                group="random_pair",
                first_key=ordered_first,
                second_key=ordered_second,
                canonical_names=canonical_names,
                drug_to_moa_display=drug_to_moa_display,
                source="random_control_matched_order_concentration_search",
            )
            spec.update(
                {
                    "pair_id": pair_id,
                    "ordered_pair_id": str(spec["pair_id"]),
                    "random_pair_index": int(random_pair_index),
                    "random_pair_drug_a": first_drug,
                    "random_pair_drug_b": second_drug,
                    "random_pair_drug_a_norm": first_key,
                    "random_pair_drug_b_norm": second_key,
                    "random_order_index": int(order_index),
                    "random_order": "sampled_order" if order_index == 1 else "reverse_sampled_order",
                    "order_search_id": ordered_pair_id,
                    "random_type": "matched",
                }
            )
            specs.append(spec)

    metadata = {
        "random_type": "matched",
        "n_random_base_drugs_eligible": len(eligible_drugs),
        "n_random_unordered_pairs_possible": int(possible_pairs),
        "n_random_unordered_pairs_sampled": int(len(chosen_base_pairs)),
        "n_random_ordered_pair_specs": int(len(specs)),
        "random_drug_filtering": "none_except_distinct_converter_available_drugs",
    }
    return specs, metadata


def selected_random_pairs_for_evaluation(selected_random_pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Expose matched random selection-batch scores through evaluation columns.

    Retained for table-only callers. The full analysis now evaluates the
    selected random controls across all batches after batch-0 selection.
    """
    if selected_random_pairs.empty:
        return selected_random_pairs.copy()

    required = {
        "selection_score_sinkhorn_ot",
        "selection_score_energy_distance",
        "selection_batch_index",
    }
    missing = required - set(selected_random_pairs.columns)
    if missing:
        raise ValueError(f"selected_random_pairs missing required columns: {sorted(missing)}")

    out = selected_random_pairs.copy()
    out["score_sinkhorn_ot"] = out["selection_score_sinkhorn_ot"].astype(float)
    out["score_energy_distance"] = out["selection_score_energy_distance"].astype(float)
    out["batch_index"] = out["selection_batch_index"].astype(int)
    out["score_context"] = "random_control_evaluation"
    return out


def attach_evaluation_stats(selected_pairs: pd.DataFrame, evaluation_df: pd.DataFrame) -> pd.DataFrame:
    """Attach mean/std/n evaluation scores to selected pair rows."""
    if selected_pairs.empty or evaluation_df.empty:
        return selected_pairs.copy()

    d = evaluation_df[evaluation_df["group"].isin(["explicit_pair", "moa_pair"])].copy()
    stats = (
        d.groupby(["group", "pair_id"], as_index=False)
        .agg(
            eval_n_batches=("score_sinkhorn_ot", "size"),
            eval_mean_sinkhorn_ot=("score_sinkhorn_ot", "mean"),
            eval_std_sinkhorn_ot=("score_sinkhorn_ot", "std"),
            eval_min_sinkhorn_ot=("score_sinkhorn_ot", "min"),
            eval_max_sinkhorn_ot=("score_sinkhorn_ot", "max"),
            eval_mean_energy_distance=("score_energy_distance", "mean"),
            eval_std_energy_distance=("score_energy_distance", "std"),
            eval_mean_delta_sinkhorn_from_baseline=("delta_sinkhorn_from_baseline", "mean"),
        )
    )
    stats["eval_std_sinkhorn_ot"] = stats["eval_std_sinkhorn_ot"].fillna(0.0)
    stats["eval_std_energy_distance"] = stats["eval_std_energy_distance"].fillna(0.0)
    out = selected_pairs.merge(stats, on=["group", "pair_id"], how="left")
    out["rank_within_group_by_eval_mean"] = (
        out.groupby("group")["eval_mean_sinkhorn_ot"].rank(method="first", ascending=True).astype("Int64")
    )
    return out.sort_values(["group", "rank_within_group_by_eval_mean", "pair_id"]).reset_index(drop=True)


def build_batch_metadata(batches: Sequence[Any]) -> pd.DataFrame:
    rows = []
    for batch_index, batch in enumerate(batches):
        meta = batch.metadata()
        meta.pop("start_obs_names", None)
        meta.pop("target_obs_names", None)
        meta["batch_index"] = int(batch_index)
        rows.append(meta)
    return pd.DataFrame(rows)


def _stack_embedding_batches(name: str, arrays: Sequence[np.ndarray]) -> np.ndarray:
    if not arrays:
        raise ValueError(f"No arrays available for {name}")
    shapes = {tuple(np.asarray(x).shape) for x in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Cannot stack {name}; batch shapes differ: {sorted(shapes)}")
    return np.stack([np.asarray(x, dtype=np.float32) for x in arrays], axis=0)


def save_explicit_pair_trajectory_embeddings(
    *,
    output_dir: Path,
    converter,
    batches: Sequence[Any],
    selected_pairs: pd.DataFrame,
) -> Dict[str, str]:
    """
    Save explicit --2drug-pair trajectory embeddings for downstream plotting.

    The saved sequence is:
        WT/start -> WT + first drug -> WT + first drug + second drug -> target
    using the selected concentration labels from batch-0 concentration search.
    """
    explicit = selected_pairs[selected_pairs["group"] == "explicit_pair"].copy()
    if explicit.empty:
        print("explicit trajectory embeddings skipped: no resolved explicit --2drug-pair was selected")
        return {}

    if "rank_within_group_by_eval_mean" in explicit.columns:
        explicit = explicit.sort_values(["rank_within_group_by_eval_mean", "pair_id"], ascending=[True, True])
    else:
        explicit = explicit.sort_values(["selection_score_sinkhorn_ot", "pair_id"], ascending=[True, True])

    row = explicit.iloc[0]
    first_label = str(row["first_perturbation"])
    second_label = str(row["second_perturbation"])

    trajectory_dir = output_dir / "trajectory_embeddings"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    npz_path = trajectory_dir / "explicit_pair_trajectory.npz"
    metadata_path = trajectory_dir / "explicit_pair_trajectory_metadata.json"

    starts: List[np.ndarray] = []
    drug1s: List[np.ndarray] = []
    drug2s: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    batch_rows: List[Dict[str, Any]] = []
    start_obs_names: List[List[str]] = []
    target_obs_names: List[List[str]] = []

    for batch_index, batch in enumerate(batches):
        print(f"Saving explicit trajectory embeddings for batch {batch_index + 1}/{len(batches)}")
        start_state = torch.as_tensor(batch.start_embeddings, dtype=torch.float32, device=converter.device)
        state_drug1 = converter.convert_one(start_state, first_label, return_cpu=False)
        state_drug2 = converter.convert_one(state_drug1, second_label, return_cpu=False)

        starts.append(np.asarray(batch.start_embeddings, dtype=np.float32))
        drug1s.append(state_drug1.detach().cpu().numpy().astype(np.float32, copy=False))
        drug2s.append(state_drug2.detach().cpu().numpy().astype(np.float32, copy=False))
        targets.append(np.asarray(batch.target_embeddings, dtype=np.float32))
        start_obs_names.append(list(map(str, batch.start_obs_names)))
        target_obs_names.append(list(map(str, batch.target_obs_names)))

        meta = batch.metadata()
        meta.pop("start_obs_names", None)
        meta.pop("target_obs_names", None)
        meta["batch_index"] = int(batch_index)
        batch_rows.append(meta)

        del state_drug1, state_drug2, start_state

    start_arr = _stack_embedding_batches("start_embeddings", starts)
    drug1_arr = _stack_embedding_batches("drug1_embeddings", drug1s)
    drug2_arr = _stack_embedding_batches("drug2_embeddings", drug2s)
    target_arr = _stack_embedding_batches("target_embeddings", targets)

    np.savez_compressed(
        npz_path,
        start_embeddings=start_arr,
        drug1_embeddings=drug1_arr,
        drug2_embeddings=drug2_arr,
        target_embeddings=target_arr,
        batch_index=np.arange(len(batches), dtype=np.int64),
        seed=np.asarray([int(batch.seed) for batch in batches], dtype=np.int64),
        start_obs_names=np.asarray(start_obs_names, dtype=object),
        target_obs_names=np.asarray(target_obs_names, dtype=object),
        state_keys=np.asarray(["start", "drug1", "drug2", "target"], dtype=object),
        state_labels=np.asarray(
            [
                "WT",
                f"WT + {row['first_drug']}",
                f"WT + {row['first_drug']} + {row['second_drug']}",
                "Target",
            ],
            dtype=object,
        ),
    )

    metadata = {
        "pair_id": str(row["pair_id"]),
        "first_drug": str(row["first_drug"]),
        "second_drug": str(row["second_drug"]),
        "first_perturbation": first_label,
        "second_perturbation": second_label,
        "first_dose": row.get("first_dose"),
        "first_dose_unit": row.get("first_dose_unit"),
        "second_dose": row.get("second_dose"),
        "second_dose_unit": row.get("second_dose_unit"),
        "source": "explicit_2drug_pair",
        "n_batches": int(len(batches)),
        "embedding_shapes": {
            "start_embeddings": list(start_arr.shape),
            "drug1_embeddings": list(drug1_arr.shape),
            "drug2_embeddings": list(drug2_arr.shape),
            "target_embeddings": list(target_arr.shape),
        },
        "batch_metadata": batch_rows,
        "selected_pair": row.to_dict(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=json_default))

    return {
        "explicit_pair_trajectory_embeddings": str(npz_path),
        "explicit_pair_trajectory_metadata": str(metadata_path),
    }


def write_outputs(
    *,
    output_dir: Path,
    config: Dict[str, Any],
    baseline_df: pd.DataFrame,
    batch_metadata_df: pd.DataFrame,
    concentration_scores: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    random_pairs: pd.DataFrame,
    evaluation_df: pd.DataFrame,
    explicit_pair_additive_df: pd.DataFrame,
    metadata: Dict[str, Any],
    batch_selection_candidates: Optional[pd.DataFrame] = None,
    projection_component_selection: Optional[pd.DataFrame] = None,
    trajectory_paths: Optional[Dict[str, str]] = None,
    projection_cache_path: Optional[str] = None,
) -> Dict[str, str]:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    trajectory_paths = trajectory_paths or {}

    baseline_path = tables_dir / "baseline_results.tsv"
    batch_path = tables_dir / "batch_metadata.tsv"
    concentration_path = tables_dir / "concentration_selection_scores.tsv"
    selected_path = tables_dir / "selected_pairs.tsv"
    random_path = tables_dir / "random_pairs.tsv"
    evaluation_path = tables_dir / "evaluation_results.tsv"
    explicit_additive_path = tables_dir / "explicit_pair_additive_results.tsv"
    batch_selection_candidates_path = tables_dir / "batch_selection_candidates.tsv"
    projection_component_selection_path = tables_dir / "projection_component_selection.tsv"
    config_path = output_dir / "positive_control_config.used.json"
    checkpoint_path = output_dir / "positive_control_checkpoint.pt"

    baseline_df.to_csv(baseline_path, sep="\t", index=False)
    batch_metadata_df.to_csv(batch_path, sep="\t", index=False)
    concentration_scores.to_csv(concentration_path, sep="\t", index=False)
    selected_pairs.to_csv(selected_path, sep="\t", index=False)
    random_pairs.to_csv(random_path, sep="\t", index=False)
    evaluation_df.to_csv(evaluation_path, sep="\t", index=False)
    explicit_pair_additive_df.to_csv(explicit_additive_path, sep="\t", index=False)
    has_batch_selection_candidates = (
        batch_selection_candidates is not None
        and hasattr(batch_selection_candidates, "empty")
        and not batch_selection_candidates.empty
    )
    if has_batch_selection_candidates:
        batch_selection_candidates.to_csv(batch_selection_candidates_path, sep="\t", index=False)
    has_projection_component_selection = (
        projection_component_selection is not None
        and hasattr(projection_component_selection, "empty")
        and not projection_component_selection.empty
    )
    if has_projection_component_selection:
        projection_component_selection.to_csv(projection_component_selection_path, sep="\t", index=False)

    payload = {
        "config": config,
        "metadata": metadata,
        "paths": {
            "baseline_results": str(baseline_path),
            "batch_metadata": str(batch_path),
            "concentration_selection_scores": str(concentration_path),
            "selected_pairs": str(selected_path),
            "random_pairs": str(random_path),
            "evaluation_results": str(evaluation_path),
            "explicit_pair_additive_results": str(explicit_additive_path),
            **(
                {"batch_selection_candidates": str(batch_selection_candidates_path)}
                if has_batch_selection_candidates
                else {}
            ),
            **(
                {"projection_component_selection": str(projection_component_selection_path)}
                if has_projection_component_selection
                else {}
            ),
            **trajectory_paths,
            **({"projection_cache": str(projection_cache_path)} if projection_cache_path else {}),
        },
    }
    config_path.write_text(json.dumps(payload, indent=2, default=json_default))

    torch.save(
        {
            "config": config,
            "metadata": metadata,
            "baseline_results": baseline_df.to_dict(orient="records"),
            "selected_pairs": selected_pairs.to_dict(orient="records"),
            "evaluation_results": evaluation_df.to_dict(orient="records"),
            "explicit_pair_additive_results": explicit_pair_additive_df.to_dict(orient="records"),
            "batch_selection_candidates": batch_selection_candidates.to_dict(orient="records")
            if has_batch_selection_candidates
            else [],
            "projection_component_selection": projection_component_selection.to_dict(orient="records")
            if has_projection_component_selection
            else [],
            "trajectory_paths": trajectory_paths,
            "projection_cache_path": str(projection_cache_path) if projection_cache_path else None,
        },
        checkpoint_path,
    )

    return {
        "baseline_results": str(baseline_path),
        "batch_metadata": str(batch_path),
        "concentration_selection_scores": str(concentration_path),
        "selected_pairs": str(selected_path),
        "random_pairs": str(random_path),
        "evaluation_results": str(evaluation_path),
        "explicit_pair_additive_results": str(explicit_additive_path),
        **(
            {"batch_selection_candidates": str(batch_selection_candidates_path)}
            if has_batch_selection_candidates
            else {}
        ),
        **(
            {"projection_component_selection": str(projection_component_selection_path)}
            if has_projection_component_selection
            else {}
        ),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        **({"projection_cache": str(projection_cache_path)} if projection_cache_path else {}),
        **trajectory_paths,
    }


def run_positive_control_2drug_analysis(
    *,
    adata: str | Path,
    start_cell: str,
    target_cell: str,
    cell_col: str,
    embed_key: str,
    model_dir: str | Path,
    checkpoint: Optional[str | Path],
    output_dir: str | Path,
    two_drug_pair: Optional[Sequence[str]] = None,
    explicit_drug_pairs: Optional[Sequence[Any]] = None,
    moa_pairs: Sequence[str] = (),
    random_pairs: int = 1000,
    random_type: str = "matched",
    n_batches: int = 5,
    start_sample: str | int = "256",
    target_sample: str | int = "256",
    batch_selection: str = "standard",
    batch_candidates: int = 300,
    batch_overlap_penalty: float = 0.02,
    batch_selection_score_chunk_size: int = 8,
    seed: int = 42,
    batch_seed_offset: int = 0,
    replace_if_needed: bool = True,
    converter_chunk_size: int = 16,
    device: Optional[str] = None,
    max_set_len: int = 256,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16",
    normalize_embeddings: bool = True,
    sinkhorn_metric: str = "cosine",
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
    metadata_dir: str | Path = "metadata",
    drug_metadata: Optional[str | Path] = None,
    include_explicit_in_moa: bool = False,
    moa_match_mode: str = "exact",
    max_moa_pairs: Optional[int] = None,
    evaluate_moa_pairs: bool = True,
    allow_random_metadata_missing: bool = False,
    projection_method: str = "none",
    projection_components: int = 128,
    projection_whiten: bool = False,
    projection_fit_cap: Optional[int] = 4000,
    projection_pca_prefilter: int = 256,
    projection_target_split: str = "auto",
    projection_split_frac: float = 0.5,
    projection_small_dataset_threshold: int = 512,
    projection_auto_epsilon: bool = True,
    projection_auto_select_components: bool = False,
    projection_selection_pca_grid: str = "96,128,192,256",
    projection_selection_pls_grid: str = "32,64,96,128,192",
    projection_selection_fit_frac: float = 0.5,
    projection_selection_repeats: int = 10,
    projection_selection_small_cell_threshold: int = 150,
    projection_selection_fallback_pca: int = 128,
    projection_selection_fallback_pls: int = 64,
    projection_selection_rule: str = "one_se",
    save_trajectory_embeddings: bool = True,
    evaluate_additive_interaction: bool = True,
) -> Dict[str, Any]:
    """
    Run the full positive-control 2-drug analysis and write output tables.
    """
    if int(n_batches) <= 0:
        raise ValueError("n_batches must be positive")
    if int(converter_chunk_size) <= 0:
        raise ValueError("converter_chunk_size must be positive")
    batch_selection = str(batch_selection)
    if batch_selection not in {"standard", "high-sensitivity"}:
        raise ValueError("batch_selection must be 'standard' or 'high-sensitivity'")
    if int(batch_candidates) <= 0:
        raise ValueError("batch_candidates must be positive")
    if int(batch_selection_score_chunk_size) <= 0:
        raise ValueError("batch_selection_score_chunk_size must be positive")
    random_type = str(random_type).replace("-", "_").casefold()
    if random_type not in {"matched", "legacy"}:
        raise ValueError("random_type must be 'matched' or 'legacy'")
    if bool(projection_auto_select_components) and str(projection_method) == "none":
        raise ValueError("projection_auto_select_components requires projection_method != 'none'")
    if int(projection_selection_repeats) <= 0:
        raise ValueError("projection_selection_repeats must be positive")
    if not (0.0 < float(projection_selection_fit_frac) < 1.0):
        raise ValueError("projection_selection_fit_frac must be between 0 and 1")
    if int(projection_selection_small_cell_threshold) <= 1:
        raise ValueError("projection_selection_small_cell_threshold must be > 1")
    projection_selection_rule = str(projection_selection_rule).replace("-", "_").casefold()
    if projection_selection_rule not in {"one_se", "best"}:
        raise ValueError("projection_selection_rule must be 'one_se' or 'best'")

    t_start = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    metadata_dir = Path(metadata_dir)
    drug_metadata_path = Path(drug_metadata) if drug_metadata else metadata_dir / "drug_metadata.csv"
    rng = np.random.default_rng(int(seed))
    explicit_pair_panel_provided = bool(explicit_drug_pairs)
    if explicit_pair_panel_provided and bool(two_drug_pair):
        raise ValueError("Pass either two_drug_pair or explicit_drug_pairs, not both.")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    scoring_params = ScoringParams(
        normalize=bool(normalize_embeddings),
        sinkhorn_metric=str(sinkhorn_metric),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iters=int(sinkhorn_iters),
        projection_auto_epsilon=bool(projection_auto_epsilon),
    )

    need_projection_setup = (str(projection_method) != "none") or (str(projection_target_split) in {"holdout", "auto"})
    projection = None
    start_eval_pool = None
    target_eval_pool = None
    projection_split_info: Optional[Dict[str, Any]] = None
    projection_cache_path: Optional[str] = None

    print("\n=== Positive-control 2-drug analysis ===")
    print(f"start cell:       {start_cell}")
    print(f"target cell:      {target_cell}")
    explicit_pair_provided = bool(two_drug_pair) or explicit_pair_panel_provided
    if explicit_pair_panel_provided:
        explicit_display = "(explicit pair panel)"
    elif two_drug_pair:
        explicit_display = list(two_drug_pair)
    else:
        explicit_display = "(not provided; using MOA fallback)"
    print(f"explicit pair:    {explicit_display}")
    if explicit_pair_panel_provided:
        print(f"explicit panel:   {len(explicit_drug_pairs or [])} pairs")
    print(f"MOA pair terms:   {list(moa_pairs)}")
    print(f"evaluate MOA:     {bool(evaluate_moa_pairs)}")
    print(f"batches:          {n_batches}")
    print(f"batch selection:  {batch_selection}")
    print(f"random pairs:     {random_pairs}")
    print(f"random type:      {random_type}")
    print(f"device:           {device}")

    if need_projection_setup:
        print("\n[0/8] Setting up scoring projection / eval split")
        projection, start_eval_pool, target_eval_pool, projection_split_info = setup_projection_and_pools(
            adata=adata,
            start_cell=start_cell,
            target_cell=target_cell,
            cell_col=cell_col,
            embed_key=embed_key,
            method=str(projection_method),
            n_components=int(projection_components),
            whiten=bool(projection_whiten),
            fit_cap=projection_fit_cap,
            pca_prefilter=int(projection_pca_prefilter),
            split_mode=str(projection_target_split),
            split_frac=float(projection_split_frac),
            small_dataset_threshold=int(projection_small_dataset_threshold),
            seed=int(seed),
            auto_select_components=bool(projection_auto_select_components),
            selection_pca_grid=str(projection_selection_pca_grid),
            selection_pls_grid=str(projection_selection_pls_grid),
            selection_fit_frac=float(projection_selection_fit_frac),
            selection_repeats=int(projection_selection_repeats),
            selection_small_cell_threshold=int(projection_selection_small_cell_threshold),
            selection_fallback_pca=int(projection_selection_fallback_pca),
            selection_fallback_pls=int(projection_selection_fallback_pls),
            selection_rule=str(projection_selection_rule),
        )
        if projection is not None:
            projection_cache = output_dir / "cache" / "projection.npz"
            projection.save(projection_cache)
            projection_cache_path = str(projection_cache)
            print(f"saved scoring projection cache: {projection_cache}")

    print("\n[1/8] Loading start/target embedding batches")
    batch_selection_candidates = pd.DataFrame()
    if batch_selection == "high-sensitivity":
        from ..data_loader import load_high_sensitivity_start_target_embedding_batches

        print(
            "high-sensitivity batch search: "
            f"{batch_candidates} local candidates, overlap penalty={batch_overlap_penalty}"
        )
        batches, batch_selection_candidates = load_high_sensitivity_start_target_embedding_batches(
            h5ad_path=adata,
            start_cell=start_cell,
            target_cell=target_cell,
            cell_col=cell_col,
            embed_key=embed_key,
            start_sample=start_sample,
            target_sample=target_sample,
            n_batches=int(n_batches),
            n_candidates=int(batch_candidates),
            overlap_penalty=float(batch_overlap_penalty),
            score_chunk_size=int(batch_selection_score_chunk_size),
            seed=int(seed),
            seed_offset=int(batch_seed_offset),
            replace_if_needed=bool(replace_if_needed),
            start_index_pool=start_eval_pool,
            target_index_pool=target_eval_pool,
            normalize=bool(normalize_embeddings),
            sinkhorn_metric=str(sinkhorn_metric),
            sinkhorn_epsilon=float(sinkhorn_epsilon),
            sinkhorn_iters=int(sinkhorn_iters),
            device=device,
            projection=projection,
            projection_auto_epsilon=bool(scoring_params.projection_auto_epsilon),
        )
    else:
        from ..data_loader import load_start_target_embedding_batches

        batches = load_start_target_embedding_batches(
            h5ad_path=adata,
            start_cell=start_cell,
            target_cell=target_cell,
            cell_col=cell_col,
            embed_key=embed_key,
            start_sample=start_sample,
            target_sample=target_sample,
            n_batches=int(n_batches),
            seed=int(seed),
            seed_offset=int(batch_seed_offset),
            replace_if_needed=bool(replace_if_needed),
            start_index_pool=start_eval_pool,
            target_index_pool=target_eval_pool,
        )
    batch_metadata_df = build_batch_metadata(batches)
    print(f"loaded {len(batches)} batches")
    for i, batch in enumerate(batches, start=1):
        msg = (
            f"  batch {i}: start {batch.start_embeddings.shape}, target {batch.target_embeddings.shape}, "
            f"seed={batch.seed}"
        )
        if batch.batch_selection == "high-sensitivity":
            msg += (
                f", candidate={batch.batch_selection_candidate_index}, "
                f"baseline_OT={batch.batch_selection_score_sinkhorn_ot:.6g}"
            )
        print(msg)

    print("\n[2/8] Loading ST-SE converter and perturbation index")
    from ..converter import StateSEConverter

    amp_dtype_torch = torch.bfloat16 if str(amp_dtype) == "bfloat16" else torch.float16
    converter = StateSEConverter(
        model_dir=str(model_dir),
        checkpoint=str(checkpoint) if checkpoint else None,
        device=device,
        max_set_len=int(max_set_len),
        use_amp=bool(use_amp),
        amp_dtype=amp_dtype_torch,
    )
    label_index, canonical_names = build_perturbation_index(converter, include_control=False)
    print(f"converter drugs with non-control labels: {len(label_index)}")

    control_label = resolve_control_perturbation(converter)
    if control_label is not None:
        print(f"control perturbation for ST-SE baseline: {control_label!r}")
    else:
        print("no control-like perturbation found; ST-SE control baselines will be skipped")

    print("\n[3/8] Loading drug metadata")
    drug_meta = load_drug_metadata(drug_metadata_path)
    drug_to_moa_norm = build_drug_to_moa_map(drug_meta)
    drug_to_moa_display = build_drug_to_moa_display_map(drug_meta)
    print(f"drug metadata rows: {len(drug_meta)}")

    explicit_pair_error = None
    explicit_pair_search_mode = "none"
    fixed_drug_moa_search_metadata: Optional[Dict[str, Any]] = None
    if explicit_pair_provided:
        try:
            if explicit_pair_panel_provided:
                explicit_pair_search_mode = "explicit_pair_panel_order_search"
                explicit_specs = build_explicit_pair_panel_specs(
                    explicit_drug_pairs or [],
                    label_index,
                    canonical_names,
                    drug_to_moa_display,
                )
                explicit_pair_keys = tuple(
                    sorted(
                        {
                            str(spec["explicit_input_first_drug_norm"])
                            for spec in explicit_specs
                        }
                        | {
                            str(spec["explicit_input_second_drug_norm"])
                            for spec in explicit_specs
                        }
                    )
                )
                print(
                    "explicit pair panel search: "
                    f"{len(set(str(spec['pair_id']) for spec in explicit_specs))} input pairs, "
                    f"{len(explicit_specs)} ordered drug orders"
                )
            elif len(two_drug_pair) == 1:
                if len(moa_pairs) != 2:
                    raise ValueError(
                        "One-drug --2drug-pair mode requires exactly two --MOA-pairs terms; "
                        "the second term is used for the partner-drug scan."
                    )
                explicit_pair_search_mode = "fixed_drug_second_moa_scan"
                explicit_specs, fixed_drug_moa_search_metadata = build_fixed_drug_moa_explicit_pair_specs(
                    str(two_drug_pair[0]),
                    str(moa_pairs[1]),
                    drug_meta,
                    label_index=label_index,
                    canonical_names=canonical_names,
                    drug_to_moa_display=drug_to_moa_display,
                    moa_match_mode=moa_match_mode,
                )
                explicit_pair_keys = (str(explicit_specs[0]["explicit_fixed_drug_norm"]),)
                print(
                    "fixed-drug explicit search: "
                    f"{fixed_drug_moa_search_metadata['fixed_drug']} + "
                    f"{fixed_drug_moa_search_metadata['n_partner_moa_drugs_available']} drugs matching "
                    f"{fixed_drug_moa_search_metadata['partner_moa_term']!r}"
                )
            elif len(two_drug_pair) == 2:
                explicit_pair_search_mode = "explicit_2drug_pair_order_search"
                explicit_specs = build_explicit_pair_specs(
                    two_drug_pair,
                    label_index,
                    canonical_names,
                    drug_to_moa_display,
                )
                explicit_pair_keys = (
                    str(explicit_specs[0]["first_drug_norm"]),
                    str(explicit_specs[0]["second_drug_norm"]),
                )
            else:
                raise ValueError(f"two_drug_pair must contain one or two drug names, got {two_drug_pair!r}")
        except (KeyError, ValueError) as exc:
            explicit_pair_error = str(exc)
            print(
                "Warning: explicit pair input could not be resolved in the converter; "
                "skipping blue explicit-pair outputs and using the best MOA pair for additive/sequential analysis."
            )
            explicit_specs = []
            explicit_pair_keys = tuple()
            explicit_pair_provided = False
            explicit_pair_search_mode = "none"
    else:
        explicit_specs = []
        explicit_pair_keys = tuple()

    print("\n[4/8] Computing baseline OT distances")
    baseline_df = compute_baselines(batches, scoring_params=scoring_params, device=device, projection=projection)
    if control_label is not None:
        control_baseline_df = compute_control_baselines(
            batches,
            converter=converter,
            control_label=control_label,
            scoring_params=scoring_params,
            device=device,
            projection=projection,
        )
    else:
        control_baseline_df = pd.DataFrame()
    baseline_mean = float(baseline_df["score_sinkhorn_ot"].mean())
    print(f"baseline mean Sinkhorn OT: {baseline_mean:.6g}")

    projection_component_selection_df = pd.DataFrame()
    if projection_split_info and projection_split_info.get("projection_component_selection"):
        selection_info = projection_split_info["projection_component_selection"] or {}
        selection_summary = selection_info.get("summary") or []
        if selection_summary:
            projection_component_selection_df = pd.DataFrame(selection_summary)

    print("\n[5/8] Selecting best concentration labels for explicit and MOA pairs on batch 0")
    batch0 = batches[0]
    scorer0 = _make_scorer(batch0.target_embeddings, scoring_params, device, projection=projection)

    if explicit_specs:
        explicit_candidates = expand_concentration_candidates(explicit_specs, label_index)
        explicit_order_count = explicit_candidates[["first_drug_norm", "second_drug_norm"]].drop_duplicates().shape[0]
        print(
            f"explicit ordered drug orders: {explicit_order_count}; "
            f"concentration/order combinations: {len(explicit_candidates)}"
        )
        explicit_selection_scores = score_ordered_label_pairs(
            converter=converter,
            scorer=scorer0,
            start_embeddings=batch0.start_embeddings,
            pair_df=explicit_candidates,
            chunk_size=int(converter_chunk_size),
            batch_index=0,
            score_context="explicit_concentration_selection",
        )
        if explicit_pair_panel_provided:
            explicit_selected = select_best_explicit_panel_orders_and_concentrations(explicit_selection_scores)
        else:
            explicit_selected = select_best_explicit_order_and_concentration(explicit_selection_scores)
        if not explicit_selected.empty:
            if explicit_pair_panel_provided:
                print(f"selected explicit panel pairs: {len(explicit_selected)}")
                for _, chosen in explicit_selected.sort_values(["selection_score_sinkhorn_ot", "pair_id"]).iterrows():
                    print(
                        "  panel pair: "
                        f"{chosen['pair_id']} => {chosen['first_drug']} -> {chosen['second_drug']} "
                        f"({chosen.get('explicit_order', 'selected_order')}); "
                        f"Sinkhorn={float(chosen['selection_score_sinkhorn_ot']):.6g}"
                    )
            else:
                chosen = explicit_selected.iloc[0]
                print(
                    "best explicit order: "
                    f"{chosen['first_drug']} -> {chosen['second_drug']} "
                    f"({chosen.get('explicit_order', 'selected_order')}); "
                    f"Sinkhorn={float(chosen['selection_score_sinkhorn_ot']):.6g}"
                )
                if explicit_pair_search_mode == "fixed_drug_second_moa_scan":
                    print(
                        "best fixed-drug MOA partner: "
                        f"{chosen['first_drug']} + {chosen['second_drug']} "
                        f"from {chosen.get('order_search_id', chosen['pair_id'])}"
                    )
    else:
        print("explicit pair not provided; skipping explicit concentration selection")
        explicit_selection_scores = pd.DataFrame()
        explicit_selected = pd.DataFrame()

    effective_moa_pairs = list(map(str, moa_pairs))
    moa_order_source = "input_order"
    selected_explicit_pair_keys: Sequence[str] = explicit_pair_keys
    moa_explicit_pair_keys: Sequence[str] = explicit_pair_keys
    if explicit_pair_panel_provided and not explicit_selected.empty:
        selected_explicit_pair_keys = tuple(
            sorted(
                set(explicit_selected["first_drug_norm"].astype(str))
                | set(explicit_selected["second_drug_norm"].astype(str))
            )
        )
        moa_explicit_pair_keys = selected_explicit_pair_keys
    elif not explicit_selected.empty:
        selected_explicit_pair_keys = (
            str(explicit_selected.iloc[0]["first_drug_norm"]),
            str(explicit_selected.iloc[0]["second_drug_norm"]),
        )
        moa_explicit_pair_keys = selected_explicit_pair_keys

    if evaluate_moa_pairs:
        effective_moa_pairs, moa_order_source = moa_pairs_for_explicit_order(moa_pairs, explicit_selected)
        if list(map(str, effective_moa_pairs)) != list(map(str, moa_pairs)):
            print(f"MOA terms reordered to match best explicit order: {effective_moa_pairs}")

        moa_specs, moa_metadata = build_moa_pair_specs(
            effective_moa_pairs,
            drug_meta,
            label_index=label_index,
            canonical_names=canonical_names,
            drug_to_moa_display=drug_to_moa_display,
            explicit_pair_keys=moa_explicit_pair_keys,
            include_explicit_in_moa=include_explicit_in_moa,
            moa_match_mode=moa_match_mode,
            max_moa_pairs=max_moa_pairs,
            rng=rng,
        )
        moa_metadata["requested_moa_pairs"] = list(map(str, moa_pairs))
        moa_metadata["effective_moa_pairs"] = list(map(str, effective_moa_pairs))
        moa_metadata["moa_order_source"] = str(moa_order_source)
        print(f"MOA ordered pairs used: {len(moa_specs)}")

        moa_candidates = expand_concentration_candidates(moa_specs, label_index)
        print(f"MOA concentration combinations: {len(moa_candidates)}")
        moa_selection_scores = score_ordered_label_pairs(
            converter=converter,
            scorer=scorer0,
            start_embeddings=batch0.start_embeddings,
            pair_df=moa_candidates,
            chunk_size=int(converter_chunk_size),
            batch_index=0,
            score_context="moa_concentration_selection",
        )
        moa_selected = select_best_concentrations(moa_selection_scores)
    else:
        moa_order_source = "skipped"
        moa_metadata = {
            "requested_moa_pairs": list(map(str, moa_pairs)),
            "effective_moa_pairs": list(map(str, effective_moa_pairs)),
            "moa_order_source": str(moa_order_source),
            "n_ordered_moa_pairs_full": 0,
            "n_ordered_moa_pairs_used": 0,
            "include_explicit_in_moa": bool(include_explicit_in_moa),
            "moa_match_mode": moa_match_mode,
            "skipped": True,
        }
        print("MOA pair scoring skipped")
        moa_selection_scores = pd.DataFrame()
        moa_selected = pd.DataFrame()

    concentration_scores = pd.concat(
        [explicit_selection_scores, moa_selection_scores],
        ignore_index=True,
        sort=False,
    )
    selected_pairs = pd.concat([explicit_selected, moa_selected], ignore_index=True, sort=False)

    print("\n[6/8] Selecting and evaluating random 2-drug controls across all batches")
    random_selection_scores = pd.DataFrame()
    random_pairs_for_eval = pd.DataFrame()
    if random_type == "matched":
        print("matched random controls: selecting best order and concentration per random drug pair")
        random_specs, random_metadata = matched_random_pair_specs(
            label_index=label_index,
            canonical_names=canonical_names,
            drug_to_moa_display=drug_to_moa_display,
            n_pairs=int(random_pairs),
            rng=rng,
        )
        random_candidate_df = expand_concentration_candidates(random_specs, label_index)
        print(
            "matched random search: "
            f"{random_metadata.get('n_random_unordered_pairs_sampled', 0)} drug pairs, "
            f"{len(random_specs)} ordered drug orders, "
            f"{len(random_candidate_df)} concentration/order combinations"
        )
        if random_candidate_df.empty:
            random_selected = pd.DataFrame()
        else:
            random_selection_scores = score_ordered_label_pairs(
                converter=converter,
                scorer=scorer0,
                start_embeddings=batch0.start_embeddings,
                pair_df=random_candidate_df,
                chunk_size=int(converter_chunk_size),
                batch_index=0,
                score_context="random_control_concentration_selection",
            )
            random_selected = select_best_random_orders_and_concentrations(random_selection_scores)
            random_pairs_for_eval = random_selected
        random_metadata["n_random_concentration_combinations_scored"] = int(len(random_selection_scores))
        random_metadata["n_random_pairs_selected"] = int(len(random_selected))
    else:
        print("legacy random controls: sampling ordered concentration labels with existing filters")
        random_candidate_df = random_pair_candidates(
            label_index=label_index,
            canonical_names=canonical_names,
            drug_to_moa_norm=drug_to_moa_norm,
            drug_to_moa_display=drug_to_moa_display,
            explicit_pair_keys=selected_explicit_pair_keys,
            moa_terms=moa_pairs,
            n_pairs=int(random_pairs),
            rng=rng,
            allow_random_metadata_missing=allow_random_metadata_missing,
        )
        random_metadata = {
            "random_type": "legacy",
            "n_random_label_pairs_requested": int(random_pairs),
            "n_random_label_pairs_sampled": int(len(random_candidate_df)),
            "allow_random_metadata_missing": bool(allow_random_metadata_missing),
            "random_drug_filtering": "exclude_explicit_drugs_requested_moas_and_metadata_missing_by_default",
        }
        random_pairs_for_eval = random_candidate_df
        random_metadata["n_random_pairs_selected"] = int(len(random_pairs_for_eval))

    if random_pairs_for_eval.empty:
        random_scored = pd.DataFrame()
    else:
        random_scored = evaluate_selected_pairs_across_batches(
            converter=converter,
            batches=batches,
            selected_pairs=random_pairs_for_eval,
            baseline_df=baseline_df,
            scoring_params=scoring_params,
            device=device,
            chunk_size=int(converter_chunk_size),
            projection=projection,
            score_context="random_control_evaluation",
            progress_label="random controls",
        )
    random_metadata["n_random_pairs_evaluated"] = (
        int(random_scored["pair_id"].nunique()) if not random_scored.empty else 0
    )
    random_metadata["n_random_evaluation_rows"] = int(len(random_scored))
    random_metadata["n_random_batches_evaluated"] = (
        int(random_scored["batch_index"].nunique()) if not random_scored.empty else 0
    )

    print("\n[7/8] Evaluating selected explicit and MOA pairs across all batches")
    selected_eval = evaluate_selected_pairs_across_batches(
        converter=converter,
        batches=batches,
        selected_pairs=selected_pairs,
        baseline_df=baseline_df,
        scoring_params=scoring_params,
        device=device,
        chunk_size=int(converter_chunk_size),
        projection=projection,
    )

    selected_pairs_with_stats = attach_evaluation_stats(selected_pairs, selected_eval)

    trajectory_paths: Dict[str, str] = {}
    if save_trajectory_embeddings and not explicit_pair_panel_provided:
        print("\n[7b/8] Saving explicit 2-drug trajectory embeddings")
        trajectory_paths = save_explicit_pair_trajectory_embeddings(
            output_dir=output_dir,
            converter=converter,
            batches=batches,
            selected_pairs=selected_pairs_with_stats,
        )
    elif save_trajectory_embeddings and explicit_pair_panel_provided:
        print("\n[7b/8] Explicit trajectory embeddings skipped for explicit-pair panel mode")

    print("\n[8/8] Evaluating additive and order-interaction variants")
    if evaluate_additive_interaction and not explicit_pair_panel_provided:
        explicit_pair_additive_df = evaluate_explicit_pair_additive_interaction(
            converter=converter,
            batches=batches,
            explicit_selected=selected_pairs_with_stats,
            baseline_df=baseline_df,
            scoring_params=scoring_params,
            device=device,
            projection=projection,
        )
    else:
        reason = "explicit-pair panel mode" if explicit_pair_panel_provided else "--no-evaluate-additive-interaction"
        print(f"additive/order-interaction variants skipped: {reason}")
        explicit_pair_additive_df = pd.DataFrame()

    baseline_for_eval = baseline_df.copy()
    baseline_for_eval["score_context"] = "baseline"
    baseline_for_eval["baseline_sinkhorn_ot"] = baseline_for_eval["score_sinkhorn_ot"]
    baseline_for_eval["baseline_energy_distance"] = baseline_for_eval["score_energy_distance"]
    baseline_for_eval["delta_sinkhorn_from_baseline"] = 0.0
    baseline_for_eval["delta_energy_from_baseline"] = 0.0

    control_for_eval = control_baseline_df.copy()
    if not control_for_eval.empty:
        # Diagnostics expressed relative to the raw baseline of the same batch.
        raw_by_batch = baseline_df.set_index("batch_index")
        control_for_eval["score_context"] = "baseline_stse_control"
        control_for_eval["baseline_sinkhorn_ot"] = control_for_eval["batch_index"].map(
            raw_by_batch["score_sinkhorn_ot"]
        )
        control_for_eval["baseline_energy_distance"] = control_for_eval["batch_index"].map(
            raw_by_batch["score_energy_distance"]
        )
        control_for_eval["delta_sinkhorn_from_baseline"] = (
            control_for_eval["score_sinkhorn_ot"].astype(float)
            - control_for_eval["baseline_sinkhorn_ot"].astype(float)
        )
        control_for_eval["delta_energy_from_baseline"] = (
            control_for_eval["score_energy_distance"].astype(float)
            - control_for_eval["baseline_energy_distance"].astype(float)
        )

    evaluation_df = pd.concat(
        [baseline_for_eval, control_for_eval, random_scored, selected_eval],
        ignore_index=True,
        sort=False,
    )

    config = {
        "adata": str(adata),
        "start_cell": str(start_cell),
        "target_cell": str(target_cell),
        "cell_col": str(cell_col),
        "embed_key": str(embed_key),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "output_dir": str(output_dir),
        "two_drug_pair": list(map(str, two_drug_pair or [])),
        "explicit_drug_pairs": list(explicit_drug_pairs or []),
        "explicit_pair_panel_provided": bool(explicit_pair_panel_provided),
        "explicit_pair_provided": bool(explicit_pair_provided),
        "explicit_pair_search_mode": str(explicit_pair_search_mode),
        "explicit_pair_error": explicit_pair_error,
        "moa_pairs": list(map(str, moa_pairs)),
        "effective_moa_pairs": list(map(str, effective_moa_pairs)),
        "moa_order_source": str(moa_order_source),
        "evaluate_moa_pairs": bool(evaluate_moa_pairs),
        "random_pairs": int(random_pairs),
        "random_type": str(random_type),
        "n_batches": int(n_batches),
        "start_sample": str(start_sample),
        "target_sample": str(target_sample),
        "batch_selection": str(batch_selection),
        "batch_candidates": int(batch_candidates),
        "batch_overlap_penalty": float(batch_overlap_penalty),
        "batch_selection_score_chunk_size": int(batch_selection_score_chunk_size),
        "seed": int(seed),
        "batch_seed_offset": int(batch_seed_offset),
        "replace_if_needed": bool(replace_if_needed),
        "converter_chunk_size": int(converter_chunk_size),
        "device": str(device),
        "max_set_len": int(max_set_len),
        "use_amp": bool(use_amp),
        "amp_dtype": str(amp_dtype),
        "scoring": asdict(scoring_params),
        "metadata_dir": str(metadata_dir),
        "drug_metadata": str(drug_metadata_path),
        "include_explicit_in_moa": bool(include_explicit_in_moa),
        "moa_match_mode": str(moa_match_mode),
        "max_moa_pairs": max_moa_pairs,
        "allow_random_metadata_missing": bool(allow_random_metadata_missing),
        "control_label": str(control_label) if control_label is not None else None,
        "projection_method": str(projection_method),
        "projection_components": int(projection_components),
        "projection_whiten": bool(projection_whiten),
        "projection_target_split": str(projection_target_split),
        "projection_split_frac": float(projection_split_frac),
        "projection_small_dataset_threshold": int(projection_small_dataset_threshold),
        "projection_auto_epsilon": bool(projection_auto_epsilon),
        "projection_auto_select_components": bool(projection_auto_select_components),
        "projection_selection_pca_grid": str(projection_selection_pca_grid),
        "projection_selection_pls_grid": str(projection_selection_pls_grid),
        "projection_selection_fit_frac": float(projection_selection_fit_frac),
        "projection_selection_repeats": int(projection_selection_repeats),
        "projection_selection_small_cell_threshold": int(projection_selection_small_cell_threshold),
        "projection_selection_fallback_pca": int(projection_selection_fallback_pca),
        "projection_selection_fallback_pls": int(projection_selection_fallback_pls),
        "projection_selection_rule": str(projection_selection_rule),
        "projection_cache_path": projection_cache_path,
        "projection_info": projection_split_info,
        "save_trajectory_embeddings": bool(save_trajectory_embeddings),
        "evaluate_additive_interaction": bool(evaluate_additive_interaction),
    }

    random_eval_rows = evaluation_df[evaluation_df["group"] == "random_pair"].copy()

    metadata = {
        "elapsed_seconds": float(time.perf_counter() - t_start),
        "n_converter_drugs": int(len(label_index)),
        "n_batches": int(len(batches)),
        "batch_selection": str(batch_selection),
        "n_batch_selection_candidates": int(len(batch_selection_candidates))
        if hasattr(batch_selection_candidates, "__len__")
        else 0,
        "baseline_mean_sinkhorn_ot": baseline_mean,
        "baseline_std_sinkhorn_ot": float(baseline_df["score_sinkhorn_ot"].std(ddof=1))
        if len(baseline_df) > 1
        else 0.0,
        "n_explicit_selected_pairs": int((selected_pairs_with_stats["group"] == "explicit_pair").sum()),
        "n_moa_selected_pairs": int((selected_pairs_with_stats["group"] == "moa_pair").sum()),
        "n_random_pairs_scored": int(random_eval_rows["pair_id"].nunique()) if not random_eval_rows.empty else 0,
        "n_random_pair_evaluation_rows": int(len(random_eval_rows)),
        "n_explicit_pair_additive_rows": int(len(explicit_pair_additive_df)),
        "trajectory_embeddings_saved": bool(trajectory_paths),
        "moa_metadata": moa_metadata,
        "random_metadata": random_metadata,
        "fixed_drug_moa_search_metadata": fixed_drug_moa_search_metadata,
        "explicit_pair_panel_provided": bool(explicit_pair_panel_provided),
        "n_explicit_input_pairs": int(len(explicit_drug_pairs or [])) if explicit_pair_panel_provided else 0,
        "explicit_pair_panel_selected": (
            selected_pairs_with_stats[selected_pairs_with_stats["group"] == "explicit_pair"].to_dict(orient="records")
            if explicit_pair_panel_provided
            else []
        ),
        "explicit_order_selected": explicit_selected.iloc[0].to_dict() if not explicit_selected.empty else None,
        "control_label": str(control_label) if control_label is not None else None,
        "baseline_stse_control_1pass_mean_sinkhorn_ot": float(
            control_baseline_df.loc[
                control_baseline_df["group"] == "baseline_stse_control_1pass", "score_sinkhorn_ot"
            ].mean()
        )
        if not control_baseline_df.empty
        else None,
        "baseline_stse_control_2pass_mean_sinkhorn_ot": float(
            control_baseline_df.loc[
                control_baseline_df["group"] == "baseline_stse_control_2pass", "score_sinkhorn_ot"
            ].mean()
        )
        if not control_baseline_df.empty
        else None,
    }

    paths = write_outputs(
        output_dir=output_dir,
        config=config,
        baseline_df=baseline_df,
        batch_metadata_df=batch_metadata_df,
        concentration_scores=concentration_scores,
        selected_pairs=selected_pairs_with_stats,
        random_pairs=random_scored,
        evaluation_df=evaluation_df,
        explicit_pair_additive_df=explicit_pair_additive_df,
        metadata=metadata,
        batch_selection_candidates=batch_selection_candidates,
        projection_component_selection=projection_component_selection_df,
        trajectory_paths=trajectory_paths,
        projection_cache_path=projection_cache_path,
    )

    print("\n=== Positive-control analysis complete ===")
    print(f"output:             {output_dir}")
    print(f"evaluation results: {paths['evaluation_results']}")
    print(f"selected pairs:     {paths['selected_pairs']}")
    if paths.get("explicit_pair_trajectory_embeddings"):
        print(f"trajectory cache:   {paths['explicit_pair_trajectory_embeddings']}")
    print(f"checkpoint:         {paths['checkpoint']}")

    return {
        "output_dir": str(output_dir),
        "paths": paths,
        "config": config,
        "metadata": metadata,
        "baseline_results": baseline_df,
        "selected_pairs": selected_pairs_with_stats,
        "evaluation_results": evaluation_df,
        "explicit_pair_additive_results": explicit_pair_additive_df,
        "batch_selection_candidates": batch_selection_candidates,
        "projection_component_selection": projection_component_selection_df,
    }
