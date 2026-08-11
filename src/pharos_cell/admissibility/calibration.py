#!/usr/bin/env python
"""
target_calibration_qc.py

Core analysis for ST-SE target calibration QC.

For each cell line and each observed 5.0 uM perturbation, this analysis:
  1. Samples one WT/control batch.
  2. Applies the matching ST-SE 5.0 uM perturbation label.
  3. Samples the actual target cells for that cell line/drug.
  4. Scores predicted-vs-actual target with Sinkhorn OT using search defaults.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


LOGGER_NAME = "target_calibration_qc"
CONTROL_LIKE_SUBSTRINGS = ("dmso", "control", "non-targeting")
CALIBRATION_MODES = ("raw", "dmso_start_only", "dmso_adapter")
CALIBRATION_MODE_SEED_OFFSETS = {"raw": 0, "dmso_start_only": 1_000_000, "dmso_adapter": 2_000_000}
PROJECTION_METHODS = ("none", "pls_da", "pca_pls_da", "pca")


@dataclass
class TargetCalibrationQCParams:
    input_h5ad: str
    model_dir: str
    output_dir: str
    checkpoint: Optional[str] = None
    cell_col: str = "cell_type"
    perturbation_col: str = "drugname_drugconc"
    control_label: str = "DMSO"
    target_calibration_mode: str = "all"
    dmso_adapter_label: Optional[str] = None
    embed_key: str = "X_state"
    cells_per_state: int = 100
    drug_concentration: float = 5.0
    drug_unit: str = "uM"
    seed: int = 42
    replace_if_needed: bool = True
    cell_types: Optional[Sequence[str]] = None
    max_cell_types: Optional[int] = None
    max_drugs: Optional[int] = None
    device: Optional[str] = None
    max_set_len: int = 100
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    converter_chunk_size: int = 8
    normalize_embeddings: bool = True
    sinkhorn_metric: str = "cosine"
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 100
    projection_method: str = "none"
    projection_components: int = 128
    projection_whiten: bool = False
    projection_fit_cap: Optional[int] = 4000
    projection_target_split: str = "auto"
    projection_split_frac: float = 0.5
    projection_small_dataset_threshold: int = 512
    projection_auto_epsilon: bool = True
    projection_pca_prefilter: int = 256
    projection_auto_select_components: bool = False
    projection_selection_pca_grid: str = "96,128,192,256"
    projection_selection_pls_grid: str = "32,64,96,128,192"
    overwrite: bool = False


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def setup_logger(output_dir: str | Path, log_name: str = "target_calibration_qc.log") -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / log_name, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def prepare_output_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is not empty: {path}\n"
                "Use --overwrite or choose a new --output-dir."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    return path


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    return path


def parse_perturbation_label(label: str) -> Tuple[str, float, str]:
    """Parse Tahoe-style labels such as "[('Trametinib', 5.0, 'uM')]". """
    try:
        parsed = ast.literal_eval(str(label))
        first = parsed[0] if isinstance(parsed, list) and parsed else parsed
        if isinstance(first, (tuple, list)) and len(first) >= 3:
            return str(first[0]), float(first[1]), str(first[2])
        if isinstance(first, (tuple, list)) and len(first) >= 1:
            return str(first[0]), math.nan, ""
    except Exception:
        pass

    try:
        from ..search import perturbation_to_drug_name

        drug = perturbation_to_drug_name(str(label))
    except Exception:
        drug = str(label)
    return drug, math.nan, ""


def drug_key(value: Any) -> str:
    return " ".join(str(value).strip().casefold().split())


def is_control_like_label(label: str) -> bool:
    lower = str(label).lower()
    return any(x in lower for x in CONTROL_LIKE_SUBSTRINGS)


def concentration_matches(label: str, *, concentration: float, unit: str) -> bool:
    _, dose, dose_unit = parse_perturbation_label(label)
    if not np.isfinite(dose):
        return False
    return math.isclose(float(dose), float(concentration), rel_tol=1e-6, abs_tol=1e-8) and str(dose_unit).casefold() == str(unit).casefold()


def perturbation_sort_key(label: str) -> Tuple[str, float, str, str]:
    drug, dose, unit = parse_perturbation_label(label)
    dose_value = float(dose) if np.isfinite(dose) else math.inf
    return (drug.casefold(), dose_value, str(unit).casefold(), str(label))


def normalize_calibration_mode(mode: str) -> str:
    mode = str(mode).strip().casefold().replace("-", "_")
    if mode not in {"raw", "dmso_start_only", "dmso_adapter", "both", "all"}:
        raise ValueError("target_calibration_mode must be one of: raw, dmso_start_only, dmso_adapter, both, all")
    return mode


def resolve_calibration_modes(mode: str) -> List[str]:
    mode = normalize_calibration_mode(mode)
    if mode == "both":
        return ["raw", "dmso_adapter"]
    if mode == "all":
        return list(CALIBRATION_MODES)
    return [mode]


def normalize_projection_method(method: str) -> str:
    method = str(method).strip().casefold().replace("-", "_")
    if method not in PROJECTION_METHODS:
        raise ValueError(f"projection_method must be one of: {', '.join(PROJECTION_METHODS)}")
    return method


def normalize_projection_split(split: str) -> str:
    split = str(split).strip().casefold().replace("-", "_")
    if split not in {"auto", "none", "holdout"}:
        raise ValueError("projection_target_split must be one of: auto, none, holdout")
    return split


def mode_uses_dmso_start(mode: str) -> bool:
    return str(mode) in {"dmso_start_only", "dmso_adapter"}


def mode_uses_dmso_target(mode: str) -> bool:
    return str(mode) == "dmso_adapter"


def make_control_mask(perturbation_values: Sequence[str], control_label: str) -> Tuple[np.ndarray, str, List[str]]:
    values = np.asarray(perturbation_values, dtype=str)
    query = str(control_label).strip()
    exact = values == query
    if np.any(exact):
        labels = sorted(pd.unique(values[exact]).astype(str).tolist())
        return exact, "exact", labels

    query_drug, query_dose, query_unit = parse_perturbation_label(query)
    query_norm = query.casefold()
    query_drug_norm = str(query_drug).casefold()
    query_unit_norm = str(query_unit).casefold()
    matched = np.zeros(values.shape[0], dtype=bool)

    for i, value in enumerate(values):
        drug, dose, unit = parse_perturbation_label(str(value))
        value_norm = str(value).casefold()
        drug_norm = str(drug).casefold()
        name_matches = (
            drug_norm == query_drug_norm
            or drug_norm.startswith(query_drug_norm)
            or query_drug_norm in drug_norm
            or query_norm in value_norm
        )
        if not name_matches:
            continue
        if np.isfinite(query_dose):
            if not np.isfinite(dose) or not math.isclose(float(dose), float(query_dose), rel_tol=1e-6, abs_tol=1e-8):
                continue
            if query_unit_norm and str(unit).casefold() != query_unit_norm:
                continue
        elif np.isfinite(dose) and not math.isclose(float(dose), 0.0, rel_tol=0.0, abs_tol=1e-12):
            continue
        matched[i] = True

    labels = sorted(pd.unique(values[matched]).astype(str).tolist()) if np.any(matched) else []
    return matched, "parsed_control_name", labels


def resolve_dmso_adapter_label(
    converter: Any,
    *,
    control_label: str,
    matched_control_labels: Sequence[str],
    requested_label: Optional[str],
    logger: logging.Logger,
) -> str:
    converter_labels = [str(x) for x in converter.list_perturbations(include_control=True)]
    converter_label_set = set(converter_labels)

    if requested_label:
        requested = str(requested_label)
        if requested not in converter_label_set:
            examples = [x for x in converter_labels if is_control_like_label(x)][:20] or converter_labels[:20]
            raise KeyError(
                f"Requested dmso_adapter_label={requested!r} was not found in the converter perturbation map. "
                f"Control-like examples: {examples}"
            )
        logger.info("Using requested DMSO adapter perturbation label: %s", requested)
        return requested

    for label in matched_control_labels:
        label = str(label)
        if label in converter_label_set:
            logger.info("Using h5ad-matched control label as DMSO adapter perturbation: %s", label)
            return label

    query_drug, _, _ = parse_perturbation_label(str(control_label))
    query_norm = drug_key(control_label)
    query_drug_norm = drug_key(query_drug)
    candidates: List[Tuple[int, str]] = []
    for label in converter_labels:
        if not is_control_like_label(label):
            continue
        drug, dose, _ = parse_perturbation_label(label)
        raw_norm = drug_key(label)
        drug_norm = drug_key(drug)
        query_terms = [x for x in {query_norm, query_drug_norm} if x]
        zero_dose = np.isfinite(dose) and math.isclose(float(dose), 0.0, rel_tol=0.0, abs_tol=1e-12)
        name_matches = (
            any(term in raw_norm for term in query_terms)
            or any(term in drug_norm for term in query_terms)
            or drug_norm in set(query_terms)
        )
        if zero_dose and name_matches:
            priority = 0
        elif zero_dose:
            priority = 1
        else:
            priority = 2
        candidates.append((priority, label))

    if not candidates:
        examples = [x for x in converter_labels if is_control_like_label(x)][:20] or converter_labels[:20]
        raise KeyError(
            "Could not resolve a DMSO adapter perturbation label from the converter. "
            "Pass --dmso-adapter-label with the exact converter label. "
            f"Control-like examples: {examples}"
        )

    candidates = sorted(set(candidates), key=lambda x: (x[0], perturbation_sort_key(x[1])))
    selected = candidates[0][1]
    logger.info("Resolved DMSO adapter perturbation label: %s", selected)
    return selected


def sample_indices(
    available_indices: np.ndarray,
    *,
    sample: int,
    rng: np.random.Generator,
    replace_if_needed: bool,
    label: str,
) -> Tuple[np.ndarray, bool]:
    available_indices = np.asarray(available_indices, dtype=np.int64)
    if len(available_indices) == 0:
        raise ValueError(f"No cells available for {label}")
    if sample <= 0:
        raise ValueError(f"sample must be positive for {label}, got {sample}")
    if len(available_indices) >= int(sample):
        return rng.choice(available_indices, size=int(sample), replace=False).astype(np.int64), False
    if not replace_if_needed:
        raise ValueError(
            f"Requested {sample} cells for {label}, but only {len(available_indices)} are available. "
            "Use --replace-if-needed or reduce --cells-per-state."
        )
    return rng.choice(available_indices, size=int(sample), replace=True).astype(np.int64), True


def selected_cell_types(
    labels: np.ndarray,
    control_mask: np.ndarray,
    requested: Optional[Sequence[str]],
    max_cell_types: Optional[int],
) -> List[str]:
    if requested:
        values = [str(x) for x in requested]
    else:
        values = sorted(pd.unique(labels[control_mask]).astype(str).tolist())
    missing = [x for x in values if not np.any((labels == x) & control_mask)]
    if missing:
        examples = sorted(pd.unique(labels[control_mask]).astype(str).tolist())[:20]
        raise ValueError(f"Requested cell types missing from control cells: {missing}. Examples: {examples}")
    if max_cell_types is not None:
        values = values[: int(max_cell_types)]
    return values


def build_converter_5um_index(converter: Any, *, concentration: float, unit: str) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for label in converter.list_perturbations(include_control=False):
        label = str(label)
        if is_control_like_label(label):
            continue
        if not concentration_matches(label, concentration=concentration, unit=unit):
            continue
        drug, _, _ = parse_perturbation_label(label)
        index.setdefault(drug_key(drug), []).append(label)
    for key in list(index):
        index[key] = sorted(set(index[key]), key=perturbation_sort_key)
    return index


def build_observed_5um_targets(
    perturbation_values: Sequence[str],
    *,
    concentration: float,
    unit: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for label in sorted(pd.unique(np.asarray(perturbation_values, dtype=str)).astype(str).tolist(), key=perturbation_sort_key):
        if is_control_like_label(label):
            continue
        if not concentration_matches(label, concentration=concentration, unit=unit):
            continue
        drug, dose, dose_unit = parse_perturbation_label(label)
        rows.append(
            {
                "target_perturbation_label": str(label),
                "drug_name": drug,
                "drug_key": drug_key(drug),
                "dose": dose,
                "dose_unit": dose_unit,
            }
        )
    return pd.DataFrame(rows)


def match_observed_targets_to_converter(
    observed_targets: pd.DataFrame,
    converter_index: Dict[str, List[str]],
    *,
    max_drugs: Optional[int],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for _, row in observed_targets.iterrows():
        key = str(row["drug_key"])
        labels = converter_index.get(key, [])
        if not labels:
            missing.append(str(row["target_perturbation_label"]))
            continue
        exact = str(row["target_perturbation_label"])
        model_label = exact if exact in labels else labels[0]
        out = row.to_dict()
        out["model_perturbation_label"] = str(model_label)
        out["n_model_labels_for_drug"] = int(len(labels))
        rows.append(out)

    if missing:
        logger.warning("Skipped %d observed 5uM target labels not found in converter map. Examples: %s", len(missing), missing[:10])

    matched = pd.DataFrame(rows)
    if matched.empty:
        raise ValueError("No observed 5uM target perturbations could be matched to converter perturbation labels.")
    matched = matched.sort_values(["drug_name", "target_perturbation_label"]).reset_index(drop=True)
    if max_drugs is not None:
        matched = matched.head(int(max_drugs)).copy()
    return matched


def state_to_numpy_2d(state: Any) -> np.ndarray:
    if torch.is_tensor(state):
        arr = state.detach().cpu().numpy()
    else:
        arr = np.asarray(state)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected state with shape [n_cells, emb_dim], got {arr.shape}")
    return arr


def slice_state_rows(state: Any, row_idx: np.ndarray) -> Any:
    row_idx = np.asarray(row_idx, dtype=np.int64)
    if torch.is_tensor(state):
        return state[torch.as_tensor(row_idx, device=state.device, dtype=torch.long)]
    return np.asarray(state)[row_idx]


def projection_default_metadata(
    *,
    projection_method: str,
    projection_components: int,
    projection_whiten: bool,
    projection_target_split: str,
    projection_split_frac: float,
    projection_small_dataset_threshold: int,
    projection_auto_epsilon: bool,
    projection_pca_prefilter: int,
    projection_auto_select_components: bool,
    projection_selection_pca_grid: str,
    projection_selection_pls_grid: str,
    n_start: int,
    n_target: int,
) -> Dict[str, Any]:
    return {
        "scoring_space": "raw_embedding" if projection_method == "none" else "projected",
        "projection_method": projection_method,
        "projection_components_requested": int(projection_components),
        "projection_components_effective": math.nan,
        "projection_whiten": bool(projection_whiten),
        "projection_target_split": projection_target_split,
        "projection_resolved_split": "none",
        "projection_split_frac": float(projection_split_frac),
        "projection_small_dataset_threshold": int(projection_small_dataset_threshold),
        "projection_auto_epsilon": bool(projection_auto_epsilon),
        "projection_auto_select_components": bool(projection_auto_select_components),
        "projection_selection_pca_grid": str(projection_selection_pca_grid),
        "projection_selection_pls_grid": str(projection_selection_pls_grid),
        "projection_pca_prefilter": int(projection_pca_prefilter),
        "projection_pca_prefilter_requested": int(projection_pca_prefilter),
        "projection_component_selection_selected_by": "",
        "projection_component_selection_skipped_reason": "",
        "projection_fit_n_start": int(n_start),
        "projection_fit_n_target": int(n_target),
        "projection_eval_n_start": int(n_start),
        "projection_eval_n_target": int(n_target),
        "projection_rotation_hash": "",
        "projection_var_recovered": math.nan,
        "projection_var_recovered_bio": math.nan,
        "projection_var_recovered_tail": math.nan,
        "projection_sinkhorn_metric": "",
        "projection_sinkhorn_epsilon": math.nan,
    }


def setup_pair_projection(
    source_state: Any,
    target_state: Any,
    *,
    method: str,
    n_components: int,
    whiten: bool,
    fit_cap: Optional[int],
    split_mode: str,
    split_frac: float,
    seed: int,
    small_dataset_threshold: int,
    auto_epsilon: bool,
    pca_prefilter: int,
    auto_select_components: bool = False,
    selection_pca_grid: str = "96,128,192,256",
    selection_pls_grid: str = "32,64,96,128,192",
) -> Tuple[Optional[Any], np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit a per-pair scoring projection and return row indices used for scoring."""
    source_np = state_to_numpy_2d(source_state)
    target_np = state_to_numpy_2d(target_state)
    if source_np.shape[1] != target_np.shape[1]:
        raise ValueError(
            f"source and target embedding dimensions differ: {source_np.shape[1]} vs {target_np.shape[1]}"
        )

    method = normalize_projection_method(method)
    split_mode = normalize_projection_split(split_mode)
    n_start = int(source_np.shape[0])
    n_target = int(target_np.shape[0])
    metadata = projection_default_metadata(
        projection_method=method,
        projection_components=int(n_components),
        projection_whiten=bool(whiten),
        projection_target_split=split_mode,
        projection_split_frac=float(split_frac),
        projection_small_dataset_threshold=int(small_dataset_threshold),
        projection_auto_epsilon=bool(auto_epsilon),
        projection_pca_prefilter=int(pca_prefilter),
        projection_auto_select_components=bool(auto_select_components),
        projection_selection_pca_grid=str(selection_pca_grid),
        projection_selection_pls_grid=str(selection_pls_grid),
        n_start=n_start,
        n_target=n_target,
    )

    all_start = np.arange(n_start, dtype=np.int64)
    all_target = np.arange(n_target, dtype=np.int64)
    if method == "none":
        return None, all_start, all_target, metadata

    resolved_split = split_mode
    if split_mode == "auto":
        resolved_split = "none" if min(n_start, n_target) <= int(small_dataset_threshold) else "holdout"

    rng = np.random.default_rng(int(seed))
    if resolved_split == "holdout":
        if not (0.0 < float(split_frac) < 1.0):
            raise ValueError("projection_split_frac must be between 0 and 1 for holdout projection splits")
        s_perm = rng.permutation(n_start)
        t_perm = rng.permutation(n_target)
        s_cut = max(1, int(round(n_start * float(split_frac))))
        t_cut = max(1, int(round(n_target * float(split_frac))))
        start_fit_idx, start_eval_idx = s_perm[:s_cut], s_perm[s_cut:]
        target_fit_idx, target_eval_idx = t_perm[:t_cut], t_perm[t_cut:]
        if len(start_eval_idx) == 0 or len(target_eval_idx) == 0:
            raise ValueError("Projection holdout split left an empty evaluation set; adjust --projection-split-frac.")
    elif resolved_split == "none":
        start_fit_idx, target_fit_idx = all_start, all_target
        start_eval_idx, target_eval_idx = all_start, all_target
    else:
        raise ValueError(f"Unknown projection split mode: {split_mode!r}")

    from ..projections import LinearProjection, select_projection_components

    effective_components = int(n_components)
    effective_pca_prefilter = int(pca_prefilter)
    component_selection: Optional[Dict[str, Any]] = None
    if bool(auto_select_components):
        component_selection = select_projection_components(
            source_np[start_fit_idx],
            target_np[target_fit_idx],
            method=method,
            pca_grid=selection_pca_grid,
            pls_grid=selection_pls_grid,
            fit_frac=0.5,
            repeats=10,
            small_cell_threshold=150,
            fallback_pca=128,
            fallback_pls=64,
            fit_cap=fit_cap,
            seed=int(seed),
            whiten=bool(whiten),
            selection_rule="one_se",
        )
        effective_components = int(component_selection["selected_projection_components"])
        effective_pca_prefilter = int(component_selection["selected_pca_components"])

    projection = LinearProjection.fit(
        source_np[start_fit_idx],
        target_np[target_fit_idx],
        method=method,
        n_components=effective_components,
        fit_cap_per_class=fit_cap,
        seed=int(seed),
        whiten_components=bool(whiten),
        pca_prefilter_components=effective_pca_prefilter,
    )
    if component_selection is not None:
        projection.fit_metadata["component_selection"] = {
            key: value for key, value in component_selection.items() if key not in {"results", "summary"}
        }
    projection_meta = projection.metadata_dict()
    fit_meta = projection_meta.get("fit_metadata", {}) or {}
    metadata.update(
        {
            "projection_resolved_split": resolved_split,
            "projection_components_effective": int(projection_meta.get("n_components", projection.output_dim)),
            "projection_fit_n_start": int(len(start_fit_idx)),
            "projection_fit_n_target": int(len(target_fit_idx)),
            "projection_eval_n_start": int(len(start_eval_idx)),
            "projection_eval_n_target": int(len(target_eval_idx)),
            "projection_rotation_hash": str(projection_meta.get("rotation_hash", "")),
            "projection_var_recovered": float(fit_meta.get("var_recovered", math.nan)),
            "projection_var_recovered_bio": float(fit_meta.get("var_recovered_bio", math.nan)),
            "projection_var_recovered_tail": float(fit_meta.get("var_recovered_tail", math.nan)),
            "projection_pca_prefilter": int(effective_pca_prefilter),
        }
    )
    if component_selection is not None:
        metadata.update(
            {
                "projection_component_selection_selected_by": str(component_selection.get("selected_by", "")),
                "projection_component_selection_skipped_reason": str(
                    component_selection.get("selection_skipped_reason") or ""
                ),
                "projection_component_selection_mean_score": component_selection.get("selected_mean_score", math.nan),
                "projection_component_selection_sem_score": component_selection.get("selected_sem_score", math.nan),
            }
        )
    return projection, start_eval_idx.astype(np.int64), target_eval_idx.astype(np.int64), metadata


def make_distribution_scorer_silent_auto_epsilon(scorer_cls: Any, **kwargs: Any) -> Any:
    if kwargs.get("projection") is not None and kwargs.get("projection_auto_epsilon"):
        with contextlib.redirect_stdout(io.StringIO()):
            return scorer_cls(**kwargs)
    return scorer_cls(**kwargs)


def summarize_scores(scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    if scores.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    d = scores.copy()
    if "calibration_mode" not in d.columns:
        d["calibration_mode"] = "raw"
    d["sinkhorn_ot"] = pd.to_numeric(d["sinkhorn_ot"], errors="coerce")
    d["baseline_wt_to_target_sinkhorn_ot"] = pd.to_numeric(d["baseline_wt_to_target_sinkhorn_ot"], errors="coerce")
    d["improvement_over_wt"] = pd.to_numeric(d["improvement_over_wt"], errors="coerce")
    d["percent_total_ot_closed"] = pd.to_numeric(d.get("percent_total_ot_closed", np.nan), errors="coerce")
    d["percent_total_ot_closed_clipped_0_100"] = pd.to_numeric(
        d.get("percent_total_ot_closed_clipped_0_100", np.nan),
        errors="coerce",
    )

    def q(x, value):
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(np.quantile(arr, value)) if len(arr) else math.nan

    cell_summary = (
        d.groupby(["calibration_mode", "cell_type"], as_index=False)
        .agg(
            n_drugs_scored=("sinkhorn_ot", "size"),
            mean_sinkhorn_ot=("sinkhorn_ot", "mean"),
            median_sinkhorn_ot=("sinkhorn_ot", "median"),
            p05_sinkhorn_ot=("sinkhorn_ot", lambda x: q(x, 0.05)),
            p95_sinkhorn_ot=("sinkhorn_ot", lambda x: q(x, 0.95)),
            mean_improvement_over_wt=("improvement_over_wt", "mean"),
            mean_percent_total_ot_closed=("percent_total_ot_closed", "mean"),
            median_percent_total_ot_closed=("percent_total_ot_closed", "median"),
        )
        .sort_values(["calibration_mode", "mean_sinkhorn_ot"])
    )
    drug_summary = (
        d.groupby(
            ["calibration_mode", "drug_name", "target_perturbation_label", "model_perturbation_label"],
            as_index=False,
        )
        .agg(
            n_cell_types_scored=("sinkhorn_ot", "size"),
            mean_sinkhorn_ot=("sinkhorn_ot", "mean"),
            median_sinkhorn_ot=("sinkhorn_ot", "median"),
            p05_sinkhorn_ot=("sinkhorn_ot", lambda x: q(x, 0.05)),
            p95_sinkhorn_ot=("sinkhorn_ot", lambda x: q(x, 0.95)),
            mean_improvement_over_wt=("improvement_over_wt", "mean"),
            mean_percent_total_ot_closed=("percent_total_ot_closed", "mean"),
            median_percent_total_ot_closed=("percent_total_ot_closed", "median"),
        )
        .sort_values(["calibration_mode", "mean_sinkhorn_ot"])
    )

    def global_stats(frame: pd.DataFrame) -> Dict[str, Any]:
        vals = np.asarray(frame["sinkhorn_ot"], dtype=float)
        vals = vals[np.isfinite(vals)]
        pct = np.asarray(frame["percent_total_ot_closed"], dtype=float)
        pct = pct[np.isfinite(pct)]
        return {
            "n_scores": int(len(frame)),
            "n_finite_scores": int(len(vals)),
            "mean_sinkhorn_ot": float(np.mean(vals)) if len(vals) else math.nan,
            "median_sinkhorn_ot": float(np.median(vals)) if len(vals) else math.nan,
            "p05_sinkhorn_ot": float(np.quantile(vals, 0.05)) if len(vals) else math.nan,
            "p95_sinkhorn_ot": float(np.quantile(vals, 0.95)) if len(vals) else math.nan,
            "min_sinkhorn_ot": float(np.min(vals)) if len(vals) else math.nan,
            "max_sinkhorn_ot": float(np.max(vals)) if len(vals) else math.nan,
            "mean_percent_total_ot_closed": float(np.mean(pct)) if len(pct) else math.nan,
            "median_percent_total_ot_closed": float(np.median(pct)) if len(pct) else math.nan,
            "p05_percent_total_ot_closed": float(np.quantile(pct, 0.05)) if len(pct) else math.nan,
            "p95_percent_total_ot_closed": float(np.quantile(pct, 0.95)) if len(pct) else math.nan,
        }

    global_summary = global_stats(d)
    global_summary["by_calibration_mode"] = {
        str(mode): global_stats(frame)
        for mode, frame in d.groupby("calibration_mode", sort=True)
    }
    return cell_summary, drug_summary, global_summary


def run_target_calibration_qc(
    *,
    input_h5ad: str | Path,
    model_dir: str | Path,
    output_dir: str | Path,
    checkpoint: Optional[str | Path] = None,
    cell_col: str = "cell_type",
    perturbation_col: str = "drugname_drugconc",
    control_label: str = "DMSO",
    target_calibration_mode: str = "all",
    dmso_adapter_label: Optional[str] = None,
    embed_key: str = "X_state",
    cells_per_state: int = 100,
    drug_concentration: float = 5.0,
    drug_unit: str = "uM",
    seed: int = 42,
    replace_if_needed: bool = True,
    cell_types: Optional[Sequence[str]] = None,
    max_cell_types: Optional[int] = None,
    max_drugs: Optional[int] = None,
    device: Optional[str] = None,
    max_set_len: int = 100,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16",
    converter_chunk_size: int = 8,
    normalize_embeddings: bool = True,
    sinkhorn_metric: str = "cosine",
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
    projection_method: str = "none",
    projection_components: int = 128,
    projection_whiten: bool = False,
    projection_fit_cap: Optional[int] = 4000,
    projection_target_split: str = "auto",
    projection_split_frac: float = 0.5,
    projection_small_dataset_threshold: int = 512,
    projection_auto_epsilon: bool = True,
    projection_pca_prefilter: int = 256,
    projection_auto_select_components: bool = False,
    projection_selection_pca_grid: str = "96,128,192,256",
    projection_selection_pls_grid: str = "32,64,96,128,192",
    overwrite: bool = False,
) -> Dict[str, Any]:
    target_calibration_mode = normalize_calibration_mode(target_calibration_mode)
    active_calibration_modes = resolve_calibration_modes(target_calibration_mode)
    projection_method = normalize_projection_method(projection_method)
    projection_target_split = normalize_projection_split(projection_target_split)
    if bool(projection_auto_select_components) and projection_method == "none":
        raise ValueError("projection_auto_select_components requires projection_method != 'none'")
    params = TargetCalibrationQCParams(
        input_h5ad=str(input_h5ad),
        model_dir=str(model_dir),
        output_dir=str(output_dir),
        checkpoint=str(checkpoint) if checkpoint else None,
        cell_col=cell_col,
        perturbation_col=perturbation_col,
        control_label=control_label,
        target_calibration_mode=target_calibration_mode,
        dmso_adapter_label=str(dmso_adapter_label) if dmso_adapter_label else None,
        embed_key=embed_key,
        cells_per_state=int(cells_per_state),
        drug_concentration=float(drug_concentration),
        drug_unit=str(drug_unit),
        seed=int(seed),
        replace_if_needed=bool(replace_if_needed),
        cell_types=list(cell_types) if cell_types else None,
        max_cell_types=max_cell_types,
        max_drugs=max_drugs,
        device=device,
        max_set_len=int(max_set_len),
        use_amp=bool(use_amp),
        amp_dtype=str(amp_dtype),
        converter_chunk_size=int(converter_chunk_size),
        normalize_embeddings=bool(normalize_embeddings),
        sinkhorn_metric=str(sinkhorn_metric),
        sinkhorn_epsilon=float(sinkhorn_epsilon),
        sinkhorn_iters=int(sinkhorn_iters),
        projection_method=str(projection_method),
        projection_components=int(projection_components),
        projection_whiten=bool(projection_whiten),
        projection_fit_cap=projection_fit_cap,
        projection_target_split=str(projection_target_split),
        projection_split_frac=float(projection_split_frac),
        projection_small_dataset_threshold=int(projection_small_dataset_threshold),
        projection_auto_epsilon=bool(projection_auto_epsilon),
        projection_pca_prefilter=int(projection_pca_prefilter),
        projection_auto_select_components=bool(projection_auto_select_components),
        projection_selection_pca_grid=str(projection_selection_pca_grid),
        projection_selection_pls_grid=str(projection_selection_pls_grid),
        overwrite=bool(overwrite),
    )

    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    start_time = time.time()
    logger.info("Starting target calibration QC.")
    logger.info("Input h5ad: %s", input_h5ad)
    logger.info("Output dir: %s", output_dir)
    logger.info("Calibration modes: %s", active_calibration_modes)
    logger.info("Scoring projection method: %s", projection_method)
    if projection_method != "none":
        logger.info(
            "Projection config: components=%d, whiten=%s, split=%s, split_frac=%.3g, auto_epsilon=%s, auto_select=%s",
            int(projection_components),
            bool(projection_whiten),
            projection_target_split,
            float(projection_split_frac),
            bool(projection_auto_epsilon),
            bool(projection_auto_select_components),
        )
    write_json(output_dir / "target_calibration_qc_config.used.json", {"config": asdict(params)})

    import scanpy as sc
    from ..converter import StateSEConverter
    from ..scoring import DistributionScorer

    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if torch.cuda.is_available():
        logger.info("PyTorch CUDA device 0: %s", torch.cuda.get_device_name(0))

    logger.info("Loading h5ad: %s", input_h5ad)
    adata = sc.read_h5ad(input_h5ad)
    if embed_key not in adata.obsm:
        raise KeyError(f"embed_key={embed_key!r} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    if cell_col not in adata.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    if perturbation_col not in adata.obs:
        raise KeyError(
            f"perturbation_col={perturbation_col!r} not found in adata.obs. Available columns: {list(adata.obs.columns)}"
        )

    X_state = np.asarray(adata.obsm[embed_key], dtype=np.float32)
    labels = adata.obs[cell_col].astype(str).to_numpy()
    perturbation_values = adata.obs[perturbation_col].astype(str).to_numpy()
    control_mask, control_match_mode, matched_control_labels = make_control_mask(perturbation_values, str(control_label))
    if not np.any(control_mask):
        examples = adata.obs[perturbation_col].astype(str).value_counts().head(20).index.astype(str).tolist()
        raise ValueError(
            f"No control cells found in {perturbation_col!r} for control_label={control_label!r}. "
            f"Example labels: {examples}"
        )
    logger.info(
        "Control selection matched %d cells across %d label(s) by %s: %s",
        int(np.sum(control_mask)),
        len(matched_control_labels),
        control_match_mode,
        matched_control_labels[:10],
    )

    selected_cells = selected_cell_types(labels, control_mask, cell_types, max_cell_types)
    logger.info("Selected %d cell lines.", len(selected_cells))

    amp_torch_dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    logger.info("Loading ST-SE converter from %s", model_dir)
    converter = StateSEConverter(
        model_dir=str(model_dir),
        checkpoint=str(checkpoint) if checkpoint else None,
        device=device,
        max_set_len=int(max_set_len),
        use_amp=bool(use_amp),
        amp_dtype=amp_torch_dtype,
        verbose=False,
    )
    resolved_dmso_adapter_label = None
    if any(mode_uses_dmso_start(mode) for mode in active_calibration_modes):
        resolved_dmso_adapter_label = resolve_dmso_adapter_label(
            converter,
            control_label=str(control_label),
            matched_control_labels=matched_control_labels,
            requested_label=dmso_adapter_label,
            logger=logger,
        )

    converter_index = build_converter_5um_index(converter, concentration=float(drug_concentration), unit=str(drug_unit))
    observed_targets = build_observed_5um_targets(
        perturbation_values,
        concentration=float(drug_concentration),
        unit=str(drug_unit),
    )
    matched_targets = match_observed_targets_to_converter(
        observed_targets,
        converter_index,
        max_drugs=max_drugs,
        logger=logger,
    )
    logger.info("Matched %d observed 5uM drug targets to converter labels.", len(matched_targets))

    rows: List[Dict[str, Any]] = []
    chunk_size = max(1, int(converter_chunk_size))
    for cell_i, cell_type in enumerate(selected_cells):
        logger.info("Scoring cell line %s (%d/%d).", cell_type, cell_i + 1, len(selected_cells))
        start_available = np.flatnonzero((labels == str(cell_type)) & control_mask).astype(np.int64)
        start_seed = int(seed) + 10_000 * cell_i
        start_idx, replace_start = sample_indices(
            start_available,
            sample=int(cells_per_state),
            rng=np.random.default_rng(start_seed),
            replace_if_needed=bool(replace_if_needed),
            label=f"{cell_type}/control",
        )
        start_embeddings = np.asarray(X_state[start_idx], dtype=np.float32).copy()

        cell_targets: List[Dict[str, Any]] = []
        for drug_i, target in matched_targets.iterrows():
            target_label = str(target["target_perturbation_label"])
            target_available = np.flatnonzero((labels == str(cell_type)) & (perturbation_values == target_label)).astype(np.int64)
            if len(target_available) == 0:
                continue
            target_seed = int(seed) + 10_000 * cell_i + 100 + int(drug_i)
            target_idx, replace_target = sample_indices(
                target_available,
                sample=int(cells_per_state),
                rng=np.random.default_rng(target_seed),
                replace_if_needed=bool(replace_if_needed),
                label=f"{cell_type}/{target_label}",
            )
            target_embeddings = np.asarray(X_state[target_idx], dtype=np.float32).copy()
            cell_targets.append(
                {
                    "target": target,
                    "target_embeddings": target_embeddings,
                    "target_n_available": int(len(target_available)),
                    "target_n_sampled": int(len(target_idx)),
                    "target_seed": int(target_seed),
                    "replace_target": bool(replace_target),
                }
            )

        if not cell_targets:
            logger.warning("No matched 5uM targets found for cell line %s.", cell_type)
            continue

        mode_start_embeddings: Dict[str, Any] = {"raw": start_embeddings}
        if any(mode_uses_dmso_start(mode) for mode in active_calibration_modes):
            logger.info("  %s [dmso_start]: adapting WT/control start cells with %s", cell_type, resolved_dmso_adapter_label)
            dmso_start_embeddings = converter.convert_one(
                start_embeddings,
                str(resolved_dmso_adapter_label),
                return_cpu=False,
            )
            mode_start_embeddings["dmso_start_only"] = dmso_start_embeddings
            mode_start_embeddings["dmso_adapter"] = dmso_start_embeddings

        raw_source_modes = [mode for mode in active_calibration_modes if not mode_uses_dmso_start(mode)]
        dmso_source_modes = [mode for mode in active_calibration_modes if mode_uses_dmso_start(mode)]
        source_groups: List[Tuple[str, Any, List[str]]] = []
        if raw_source_modes:
            source_groups.append(("raw_start", mode_start_embeddings["raw"], raw_source_modes))
        if dmso_source_modes:
            source_groups.append(("dmso_start", mode_start_embeddings["dmso_start_only"], dmso_source_modes))

        for chunk_start in range(0, len(cell_targets), chunk_size):
            chunk = cell_targets[chunk_start : chunk_start + chunk_size]
            model_labels = [str(item["target"]["model_perturbation_label"]) for item in chunk]
            raw_target_states = [item["target_embeddings"] for item in chunk]
            dmso_target_states = None
            if any(mode_uses_dmso_target(mode) for mode in active_calibration_modes):
                dmso_target_states = [
                    converter.convert_one(
                        item["target_embeddings"],
                        str(resolved_dmso_adapter_label),
                        return_cpu=False,
                    )
                    for item in chunk
                ]

            for source_name, source_embeddings, scoring_modes in source_groups:
                logger.info(
                    "  %s [%s -> %s]: converting/scoring drugs %d-%d/%d",
                    cell_type,
                    source_name,
                    ",".join(scoring_modes),
                    chunk_start + 1,
                    chunk_start + len(chunk),
                    len(cell_targets),
                )
                source_projection_state = (
                    state_to_numpy_2d(source_embeddings) if projection_method != "none" else source_embeddings
                )
                for out_labels, pred_batch in converter.convert_many_iter(
                    source_embeddings,
                    perturbations=model_labels,
                    chunk_size=chunk_size,
                    return_cpu=False,
                ):
                    for j, model_label in enumerate(out_labels):
                        item = chunk[j]
                        target = item["target"]
                        for calibration_mode in scoring_modes:
                            target_states = dmso_target_states if mode_uses_dmso_target(calibration_mode) else raw_target_states
                            if target_states is None:
                                raise RuntimeError(f"Missing target states for calibration mode {calibration_mode}")
                            start_state_label = "raw_control" if calibration_mode == "raw" else "dmso_adapter_control"
                            target_state_label = (
                                "dmso_adapter_observed_target"
                                if mode_uses_dmso_target(calibration_mode)
                                else "raw_observed_target"
                            )
                            row_dmso_adapter_label = "" if calibration_mode == "raw" else str(resolved_dmso_adapter_label)
                            target_state_full = target_states[j]
                            projection_seed = int(item["target_seed"]) + int(CALIBRATION_MODE_SEED_OFFSETS.get(calibration_mode, 0))
                            projection, source_eval_idx, target_eval_idx, projection_meta = setup_pair_projection(
                                source_projection_state,
                                target_state_full,
                                method=projection_method,
                                n_components=int(projection_components),
                                whiten=bool(projection_whiten),
                                fit_cap=projection_fit_cap,
                                split_mode=projection_target_split,
                                split_frac=float(projection_split_frac),
                                seed=projection_seed,
                                small_dataset_threshold=int(projection_small_dataset_threshold),
                                auto_epsilon=bool(projection_auto_epsilon),
                                pca_prefilter=int(projection_pca_prefilter),
                                auto_select_components=bool(projection_auto_select_components),
                                selection_pca_grid=str(projection_selection_pca_grid),
                                selection_pls_grid=str(projection_selection_pls_grid),
                            )
                            score_target_state = slice_state_rows(target_state_full, target_eval_idx)
                            score_source_embeddings = slice_state_rows(source_embeddings, source_eval_idx)
                            pred_state = slice_state_rows(pred_batch[j], source_eval_idx)
                            scorer = make_distribution_scorer_silent_auto_epsilon(
                                DistributionScorer,
                                target_state=score_target_state,
                                device=device,
                                normalize=bool(normalize_embeddings) and projection is None,
                                sinkhorn_metric=sinkhorn_metric,
                                sinkhorn_epsilon=float(sinkhorn_epsilon),
                                sinkhorn_iters=int(sinkhorn_iters),
                                projection=projection,
                                projection_auto_metric=True,
                                projection_auto_epsilon=bool(projection is not None and projection_auto_epsilon),
                            )
                            sinkhorn_score = float(scorer.sinkhorn(pred_state).detach().cpu().item())
                            baseline_score = float(scorer.sinkhorn(score_source_embeddings).detach().cpu().item())
                            scorer_projection_meta = scorer.projection_metadata()
                            if scorer_projection_meta:
                                projection_meta["projection_sinkhorn_metric"] = str(
                                    scorer_projection_meta.get("sinkhorn_metric", scorer.sinkhorn_metric)
                                )
                                projection_meta["projection_sinkhorn_epsilon"] = float(
                                    scorer_projection_meta.get("sinkhorn_epsilon", scorer.sinkhorn_epsilon)
                                )
                                projection_meta["normalize_embeddings"] = bool(scorer_projection_meta.get("normalize", scorer.normalize))
                            else:
                                projection_meta["projection_sinkhorn_metric"] = str(scorer.sinkhorn_metric)
                                projection_meta["projection_sinkhorn_epsilon"] = float(scorer.sinkhorn_epsilon)
                                projection_meta["normalize_embeddings"] = bool(scorer.normalize)
                            improvement = baseline_score - sinkhorn_score
                            if baseline_score > 1e-12:
                                remaining_ot_percent = 100.0 * sinkhorn_score / baseline_score
                                percent_total_ot_closed = 100.0 * improvement / baseline_score
                            else:
                                remaining_ot_percent = math.nan
                                percent_total_ot_closed = math.nan
                            rows.append(
                                {
                                    "calibration_mode": str(calibration_mode),
                                    "start_state_label": start_state_label,
                                    "target_state_label": target_state_label,
                                    "dmso_adapter_label": row_dmso_adapter_label,
                                    "cell_type": str(cell_type),
                                    "drug_name": str(target["drug_name"]),
                                    "target_perturbation_label": str(target["target_perturbation_label"]),
                                    "model_perturbation_label": str(model_label),
                                    "dose": float(target["dose"]),
                                    "dose_unit": str(target["dose_unit"]),
                                    "sinkhorn_ot": sinkhorn_score,
                                    "baseline_start_to_target_sinkhorn_ot": baseline_score,
                                    "baseline_wt_to_target_sinkhorn_ot": baseline_score,
                                    "improvement_over_start": improvement,
                                    "improvement_over_wt": improvement,
                                    "remaining_ot_percent_of_baseline": remaining_ot_percent,
                                    "percent_total_ot_closed": percent_total_ot_closed,
                                    "percent_total_ot_closed_clipped_0_100": (
                                        float(np.clip(percent_total_ot_closed, 0.0, 100.0))
                                        if np.isfinite(percent_total_ot_closed)
                                        else math.nan
                                    ),
                                    "target_self_term": scorer.target_self_term,
                                    "n_start_available": int(len(start_available)),
                                    "n_start_sampled": int(len(start_idx)),
                                    "replace_start": bool(replace_start),
                                    "start_seed": int(start_seed),
                                    "n_target_available": int(item["target_n_available"]),
                                    "n_target_sampled": int(item["target_n_sampled"]),
                                    "replace_target": bool(item["replace_target"]),
                                    "target_seed": int(item["target_seed"]),
                                    "projection_seed": int(projection_seed),
                                    "sinkhorn_metric": str(scorer.sinkhorn_metric),
                                    "sinkhorn_metric_requested": str(sinkhorn_metric),
                                    "sinkhorn_epsilon": float(scorer.sinkhorn_epsilon),
                                    "sinkhorn_epsilon_requested": float(sinkhorn_epsilon),
                                    "sinkhorn_iters": int(sinkhorn_iters),
                                    "normalize_embeddings_requested": bool(normalize_embeddings),
                                    **projection_meta,
                                }
                            )
                    del pred_batch
            del raw_target_states
            if dmso_target_states is not None:
                del dmso_target_states
        del mode_start_embeddings

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise RuntimeError("No target calibration scores were generated.")
    cell_summary, drug_summary, global_summary = summarize_scores(scores)

    paths_out: Dict[str, str] = {
        "config": str(output_dir / "target_calibration_qc_config.used.json"),
        "log": str(output_dir / "target_calibration_qc.log"),
        "scores": str(write_table(scores, table_dir / "target_calibration_scores.tsv")),
        "cell_line_summary": str(write_table(cell_summary, table_dir / "cell_line_summary.tsv")),
        "drug_summary": str(write_table(drug_summary, table_dir / "drug_summary.tsv")),
        "matched_targets": str(write_table(matched_targets, table_dir / "matched_5um_targets.tsv")),
    }
    metadata = {
        "n_scores": int(len(scores)),
        "n_cell_types": int(scores["cell_type"].nunique()),
        "n_drugs": int(scores["drug_name"].nunique()),
        "calibration_modes": active_calibration_modes,
        "projection_method": projection_method,
        "projection_components": int(projection_components),
        "projection_whiten": bool(projection_whiten),
        "projection_target_split": projection_target_split,
        "projection_split_frac": float(projection_split_frac),
        "projection_small_dataset_threshold": int(projection_small_dataset_threshold),
        "projection_auto_epsilon": bool(projection_auto_epsilon),
        "projection_auto_select_components": bool(projection_auto_select_components),
        "projection_selection_pca_grid": str(projection_selection_pca_grid),
        "projection_selection_pls_grid": str(projection_selection_pls_grid),
        "resolved_dmso_adapter_label": resolved_dmso_adapter_label,
        "control_match_mode": control_match_mode,
        "matched_control_labels": matched_control_labels,
        "elapsed_minutes": (time.time() - start_time) / 60.0,
        "global_summary": global_summary,
    }
    manifest = {"config": asdict(params), "metadata": metadata, "paths": paths_out}
    manifest_path = write_json(output_dir / "target_calibration_qc_manifest.json", manifest)
    paths_out["manifest"] = str(manifest_path)
    logger.info("Target calibration QC complete in %.2f minutes.", metadata["elapsed_minutes"])
    logger.info("Scores: %s", paths_out["scores"])
    logger.info("Manifest: %s", manifest_path)
    return {"output_dir": str(output_dir), "metadata": metadata, "paths": paths_out}
