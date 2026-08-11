#!/usr/bin/env python
"""
search.py

Beam-search algorithms for sequential ST-SE perturbation search.

This module assumes you already have:

    converter: StateSEConverter
        - converter.convert_many_iter(state, perturbations, chunk_size)
        - converter.list_perturbations(include_control=False)

    scorer: DistributionScorer
        - scorer.energy_distance(candidate_batch)
        - scorer.sinkhorn(candidate_batch)

    start_embeddings:
        - np.ndarray or torch.Tensor [n_cells, emb_dim], usually [256, 2058]

Core algorithms
---------------
1. deterministic_beam_search()
   - no repeated drug names by default
   - configurable prefilter metric
   - Sinkhorn OT rerank
   - saves results.tsv and checkpoint.pt

2. diverse_beam_search()
   - same as deterministic beam search
   - adds path-overlap penalty
   - optional state-similarity penalty

Optional robust reranking
-------------------------
If cfg["robustness"]["enabled"] is true and robust_samples are provided,
each depth reranks the post-Sinkhorn candidate pool by replaying every
candidate path on additional sampled start/target batches before retaining
the beam. The robust aggregate is stored in adjusted_score.

Important concentration handling
--------------------------------
Tahoe perturbation labels often contain the same drug at multiple concentrations, e.g.

    "[('18β-Glycyrrhetinic acid', 0.5, 'uM')]"
    "[('18β-Glycyrrhetinic acid', 0.05, 'uM')]"
    "[('18β-Glycyrrhetinic acid', 5.0, 'uM')]"

By default, the search may choose any concentration at a given step, but once a
drug name has been used, no other concentration of the same drug can be selected
later in the same path.

Output
------
output_dir/
    results.tsv
    checkpoint.pt

checkpoint.pt contains:
    {
        "config": config dictionary,
        "baseline_scores": {
            "sinkhorn": float,             # Sinkhorn OT(start_state, target_state)
            "energy_distance": float,      # Energy distance(start_state, target_state)
            "target_self_term": float,     # Cached E||Y-Y'|| for the target
        } or None,
        "depths": {
            depth: {
                "paths": list[tuple[str]],
                "drug_names": list[tuple[str]],
                "scores_sinkhorn": list[float],
                "scores_energy_distance": list[float],
                "adjusted_scores": list[float],
                "states": torch.Tensor [beam_size, n_cells, emb_dim] on CPU float16/float32,
            }
        }
    }

Depth 0
-------
Depth 0 records the un-perturbed starting state. Its score_sinkhorn and
score_energy_distance are the baseline distances start_state -> target_state,
so depth-1 rows in results.tsv carry a meaningful delta_sinkhorn_from_parent
(score_after_first_drug - baseline).
"""

from __future__ import annotations

import ast
import heapq
import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import yaml


# -----------------------------
# Config / dataclasses
# -----------------------------


@dataclass
class SearchNode:
    path: Tuple[str, ...]
    drug_names: Tuple[str, ...]
    state: torch.Tensor
    score_sinkhorn: float = math.inf
    score_energy_distance: float = math.inf
    score_prefilter: float = math.inf
    prefilter_metric: str = "energy"
    adjusted_score: float = math.inf
    parent_score_sinkhorn: Optional[float] = None
    robust_score_mean: float = math.nan
    robust_score_median: float = math.nan
    robust_score_std: float = math.nan
    robust_score_worst: float = math.nan
    robust_score_aggregate: float = math.nan
    robust_n_samples: int = 0

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def last_drug(self) -> str:
        return self.path[-1] if self.path else ""

    @property
    def last_drug_name(self) -> str:
        return self.drug_names[-1] if self.drug_names else ""


@dataclass
class Candidate:
    path: Tuple[str, ...]
    drug_names: Tuple[str, ...]
    state: torch.Tensor
    parent_score_sinkhorn: Optional[float]
    score_energy_distance: float
    score_prefilter: float = math.inf
    prefilter_metric: str = "energy"
    score_sinkhorn: float = math.inf
    adjusted_score: float = math.inf
    robust_score_mean: float = math.nan
    robust_score_median: float = math.nan
    robust_score_std: float = math.nan
    robust_score_worst: float = math.nan
    robust_score_aggregate: float = math.nan
    robust_n_samples: int = 0


def load_search_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def save_search_config(cfg: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def get_nested(cfg: Dict[str, Any], keys: Sequence[str], default=None):
    d = cfg
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


# -----------------------------
# Perturbation parsing / constraints
# -----------------------------


def perturbation_to_drug_name(perturbation_label: str) -> str:
    """
    Extract base drug name from a Tahoe perturbation label.

    Expected label:
        "[('Trametinib', 0.05, 'uM')]"

    Returns:
        "Trametinib"

    Fallbacks are intentionally permissive because model maps sometimes contain
    labels with slightly different formatting.
    """
    s = str(perturbation_label)

    try:
        parsed = ast.literal_eval(s)
        # Common Tahoe format: list of tuples
        if isinstance(parsed, list) and len(parsed) > 0:
            first = parsed[0]
            if isinstance(first, (tuple, list)) and len(first) > 0:
                return str(first[0])
        # Sometimes just tuple
        if isinstance(parsed, tuple) and len(parsed) > 0:
            return str(parsed[0])
    except Exception:
        pass

    # Fallback: try to capture first quoted string.
    m = re.search(r"['\"]([^'\"]+)['\"]", s)
    if m:
        return m.group(1)

    # Last fallback: strip concentration-ish suffixes poorly but safely.
    return s


def filter_allowed_perturbations(
    perturbations: Sequence[str],
    cfg: Dict[str, Any],
) -> List[str]:
    """
    Apply global perturbation filters before search.

    This does not handle per-path repeated-drug constraints; that is handled
    during expansion.
    """
    constraints = cfg.get("constraints", {}) or {}
    search_cfg = cfg.get("search", {}) or {}

    banned_drug_names = set(map(str, constraints.get("banned_drug_names", []) or []))
    banned_perturbation_labels = set(map(str, constraints.get("banned_perturbation_labels", []) or []))

    allowed_drug_names = constraints.get("allowed_drug_names", None)
    if allowed_drug_names is not None:
        allowed_drug_names = set(map(str, allowed_drug_names))

    allowed_name_contains = constraints.get("allowed_drug_name_contains", None)
    if allowed_name_contains is not None:
        allowed_name_contains = [str(x).lower() for x in allowed_name_contains]

    filtered = []
    for p in map(str, perturbations):
        drug_name = perturbation_to_drug_name(p)

        if p in banned_perturbation_labels:
            continue
        if drug_name in banned_drug_names:
            continue
        if allowed_drug_names is not None and drug_name not in allowed_drug_names:
            continue
        if allowed_name_contains is not None:
            if not any(substr in drug_name.lower() for substr in allowed_name_contains):
                continue

        # Usually skip DMSO/control-like perturbations in search.
        if not constraints.get("allow_control_like_drugs", False):
            lower = p.lower()
            if "dmso" in lower or "control" in lower or "non-targeting" in lower:
                continue

        filtered.append(p)

    # Useful for short smoke tests.
    max_drugs_to_consider = search_cfg.get("max_drugs_to_consider", None)
    if max_drugs_to_consider is not None:
        filtered = filtered[: int(max_drugs_to_consider)]

    return filtered


def path_allows_perturbation(
    node: SearchNode,
    perturbation: str,
    cfg: Dict[str, Any],
) -> bool:
    """
    Return whether perturbation is allowed to extend this node.

    Main rule:
        no repeated base drug names by default, even at different concentrations.
    """
    constraints = cfg.get("constraints", {}) or {}

    allow_repeated_labels = bool(constraints.get("allow_repeated_perturbation_labels", False))
    allow_repeated_drug_names = bool(constraints.get("allow_repeated_drug_names", False))

    p = str(perturbation)
    drug_name = perturbation_to_drug_name(p)

    if not allow_repeated_labels and p in node.path:
        return False

    if not allow_repeated_drug_names and drug_name in set(node.drug_names):
        return False

    return True


# -----------------------------
# Scoring helpers
# -----------------------------


def _to_device_state(x, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32, non_blocking=True)
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def _state_similarity_penalty(candidate_state: torch.Tensor, selected_states: List[torch.Tensor]) -> float:
    """
    Cosine similarity penalty between candidate centroid and selected centroids.

    Returns max cosine similarity to selected states in [roughly -1, 1].
    Higher means more redundant. If no selected states, returns 0.
    """
    if not selected_states:
        return 0.0

    c = candidate_state.mean(dim=0)
    c = c / (torch.linalg.norm(c) + 1e-8)

    vals = []
    for s in selected_states:
        ss = s.to(candidate_state.device)
        m = ss.mean(dim=0)
        m = m / (torch.linalg.norm(m) + 1e-8)
        vals.append(torch.dot(c, m))

    return float(torch.stack(vals).max().detach().cpu())


def _path_overlap_fraction(path_a: Sequence[str], path_b: Sequence[str]) -> float:
    """
    Fractional base-drug-name overlap between two paths.
    """
    a = {perturbation_to_drug_name(p) for p in path_a}
    b = {perturbation_to_drug_name(p) for p in path_b}
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def _select_diverse_nodes(
    candidates: List[Candidate],
    beam_size: int,
    cfg: Dict[str, Any],
) -> List[SearchNode]:
    """
    Greedily select beam nodes while penalizing redundancy with already selected nodes.
    """
    diversity = cfg.get("diversity", {}) or {}
    path_lambda = float(diversity.get("path_overlap_penalty", 0.0))
    state_lambda = float(diversity.get("state_similarity_penalty", 0.0))
    use_state_penalty = bool(diversity.get("use_state_similarity_penalty", False)) and state_lambda > 0

    remaining = list(candidates)
    selected: List[Candidate] = []
    selected_states: List[torch.Tensor] = []

    while remaining and len(selected) < beam_size:
        best_i = None
        best_adjusted = math.inf

        for i, cand in enumerate(remaining):
            base_score = cand.adjusted_score if np.isfinite(cand.adjusted_score) else cand.score_sinkhorn
            path_pen = 0.0
            if path_lambda > 0 and selected:
                path_pen = max(_path_overlap_fraction(cand.path, s.path) for s in selected)

            state_pen = 0.0
            if use_state_penalty and selected_states:
                state_pen = _state_similarity_penalty(cand.state, selected_states)

            adjusted = base_score + path_lambda * path_pen + state_lambda * state_pen

            if adjusted < best_adjusted:
                best_adjusted = adjusted
                best_i = i

        chosen = remaining.pop(best_i)
        chosen.adjusted_score = float(best_adjusted)
        selected.append(chosen)
        selected_states.append(chosen.state.detach())

    nodes = [
        SearchNode(
            path=c.path,
            drug_names=c.drug_names,
            state=c.state.detach(),
            score_sinkhorn=float(c.score_sinkhorn),
            score_energy_distance=float(c.score_energy_distance),
            score_prefilter=float(c.score_prefilter),
            prefilter_metric=str(c.prefilter_metric),
            adjusted_score=float(c.adjusted_score),
            parent_score_sinkhorn=c.parent_score_sinkhorn,
            robust_score_mean=float(c.robust_score_mean),
            robust_score_median=float(c.robust_score_median),
            robust_score_std=float(c.robust_score_std),
            robust_score_worst=float(c.robust_score_worst),
            robust_score_aggregate=float(c.robust_score_aggregate),
            robust_n_samples=int(c.robust_n_samples),
        )
        for c in selected
    ]
    return nodes


def _select_best_nodes(candidates: List[Candidate], beam_size: int) -> List[SearchNode]:
    candidates = sorted(
        candidates,
        key=lambda c: c.adjusted_score if np.isfinite(c.adjusted_score) else c.score_sinkhorn,
    )[:beam_size]
    nodes = [
        SearchNode(
            path=c.path,
            drug_names=c.drug_names,
            state=c.state.detach(),
            score_sinkhorn=float(c.score_sinkhorn),
            score_energy_distance=float(c.score_energy_distance),
            score_prefilter=float(c.score_prefilter),
            prefilter_metric=str(c.prefilter_metric),
            adjusted_score=float(c.adjusted_score),
            parent_score_sinkhorn=c.parent_score_sinkhorn,
            robust_score_mean=float(c.robust_score_mean),
            robust_score_median=float(c.robust_score_median),
            robust_score_std=float(c.robust_score_std),
            robust_score_worst=float(c.robust_score_worst),
            robust_score_aggregate=float(c.robust_score_aggregate),
            robust_n_samples=int(c.robust_n_samples),
        )
        for c in candidates
    ]
    return nodes


# -----------------------------
# Output helpers
# -----------------------------


def node_to_result_row(
    node: SearchNode,
    algorithm: str,
    depth: int,
    rank: int,
    start_cell: Optional[str] = None,
    target_cell: Optional[str] = None,
) -> Dict[str, Any]:
    delta = np.nan
    if node.parent_score_sinkhorn is not None and np.isfinite(node.parent_score_sinkhorn):
        delta = node.score_sinkhorn - float(node.parent_score_sinkhorn)

    return {
        "algorithm": algorithm,
        "depth": depth,
        "rank": rank,
        "num_drugs": len(node.path),
        "path_json": json.dumps(list(node.path), ensure_ascii=False),
        "drug_names_json": json.dumps(list(node.drug_names), ensure_ascii=False),
        "path_string": " -> ".join(node.path),
        "drug_name_string": " -> ".join(node.drug_names),
        "last_perturbation": node.last_drug,
        "last_drug_name": node.last_drug_name,
        "score_sinkhorn_ot": node.score_sinkhorn,
        "score_energy_distance": node.score_energy_distance,
        "score_prefilter": node.score_prefilter,
        "prefilter_metric": node.prefilter_metric,
        "adjusted_score": node.adjusted_score,
        "delta_sinkhorn_from_parent": delta,
        "robust_score_mean": node.robust_score_mean,
        "robust_score_median": node.robust_score_median,
        "robust_score_std": node.robust_score_std,
        "robust_score_worst": node.robust_score_worst,
        "robust_score_aggregate": node.robust_score_aggregate,
        "robust_n_samples": node.robust_n_samples,
        "start_cell": start_cell,
        "target_cell": target_cell,
    }


def write_results_tsv(rows: List[Dict[str, Any]], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "results.tsv"
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False)
    return path


def save_checkpoint(
    output_dir: str | Path,
    cfg: Dict[str, Any],
    depth_to_nodes: Dict[int, List[SearchNode]],
    rows: List[Dict[str, Any]],
    state_dtype: str = "float16",
    baseline_scores: Optional[Dict[str, float]] = None,
    projection_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save a single checkpoint.pt containing all depths.

    States are moved to CPU. By default they are stored as float16 to reduce disk use.

    Parameters
    ----------
    baseline_scores:
        Optional dictionary with baseline distances of the un-perturbed start state
        against the target state, e.g. {"sinkhorn": float, "energy_distance": float, "target_self_term": float}.
        Persisted so downstream reports can show start -> first-drug improvements
        without recomputing the baseline.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if state_dtype == "float16" else torch.float32

    depths = {}
    for depth, nodes in depth_to_nodes.items():
        if nodes:
            states = torch.stack([n.state.detach().cpu().to(dtype) for n in nodes], dim=0)
        else:
            states = torch.empty(0)

        depths[int(depth)] = {
            "paths": [n.path for n in nodes],
            "drug_names": [n.drug_names for n in nodes],
            "scores_sinkhorn": [float(n.score_sinkhorn) for n in nodes],
            "scores_energy_distance": [float(n.score_energy_distance) for n in nodes],
            "scores_prefilter": [float(n.score_prefilter) for n in nodes],
            "prefilter_metrics": [str(n.prefilter_metric) for n in nodes],
            "adjusted_scores": [float(n.adjusted_score) for n in nodes],
            "robust_score_mean": [float(n.robust_score_mean) for n in nodes],
            "robust_score_median": [float(n.robust_score_median) for n in nodes],
            "robust_score_std": [float(n.robust_score_std) for n in nodes],
            "robust_score_worst": [float(n.robust_score_worst) for n in nodes],
            "robust_score_aggregate": [float(n.robust_score_aggregate) for n in nodes],
            "robust_n_samples": [int(n.robust_n_samples) for n in nodes],
            "states": states,
        }

    ckpt = {
        "config": cfg,
        "baseline_scores": baseline_scores,
        "projection_metadata": projection_metadata,
        "depths": depths,
        "results_rows": rows,
        "saved_at_unix_time": time.time(),
    }

    path = output_dir / "checkpoint.pt"
    torch.save(ckpt, path)
    return path


# -----------------------------
# Core search
# -----------------------------


def _prefilter_metric(cfg: Dict[str, Any]) -> str:
    search_cfg = cfg.get("search", {}) or {}
    metric = str(search_cfg.get("prefilter_metric", "sinkhorn_low_iter"))
    allowed = {"energy", "sinkhorn_low_iter"}
    if metric not in allowed:
        raise ValueError(f"search.prefilter_metric must be one of {sorted(allowed)}, got {metric!r}")
    return metric


def _prefilter_sinkhorn_iters(cfg: Dict[str, Any]) -> int:
    search_cfg = cfg.get("search", {}) or {}
    iters = int(search_cfg.get("prefilter_sinkhorn_iters", 10))
    if iters < 1:
        raise ValueError(f"search.prefilter_sinkhorn_iters must be >= 1, got {iters}")
    return iters


def _score_prefilter_batch(pred_batch: torch.Tensor, scorer, cfg: Dict[str, Any], device: torch.device) -> torch.Tensor:
    metric = _prefilter_metric(cfg)
    if metric == "energy":
        return scorer.energy_distance(pred_batch)

    # Low-iteration Sinkhorn prefilter; uses scorer projection/normalize/epsilon when set.
    return scorer.sinkhorn(pred_batch, n_iters=_prefilter_sinkhorn_iters(cfg))


def _expand_and_prefilter(
    beam: List[SearchNode],
    perturbations: Sequence[str],
    converter,
    scorer,
    cfg: Dict[str, Any],
    top_m: int,
    device: torch.device,
) -> List[Candidate]:
    """
    Expand beam nodes over allowed perturbations and keep only top_m by prefilter score.

    This is memory-efficient because only top_m candidate states are retained.
    """
    search_cfg = cfg.get("search", {}) or {}
    chunk_size = int(search_cfg.get("converter_chunk_size", 16))
    prefilter_metric = _prefilter_metric(cfg)

    heap = []
    counter = 0

    for parent_idx, node in enumerate(beam):
        allowed = [p for p in perturbations if path_allows_perturbation(node, p, cfg)]
        if not allowed:
            continue

        for labels, pred_batch in converter.convert_many_iter(
            node.state,
            perturbations=allowed,
            chunk_size=chunk_size,
            return_cpu=False,
        ):
            # pred_batch: [chunk, n_cells, emb_dim]
            prefilter_scores = _score_prefilter_batch(pred_batch, scorer, cfg, device)

            prefilter_cpu = prefilter_scores.detach().cpu().numpy()
            for j, p in enumerate(labels):
                prefilter_score = float(prefilter_cpu[j])
                drug_name = perturbation_to_drug_name(p)

                cand = Candidate(
                    path=node.path + (str(p),),
                    drug_names=node.drug_names + (drug_name,),
                    state=pred_batch[j].detach().clone(),
                    parent_score_sinkhorn=node.score_sinkhorn,
                    score_energy_distance=prefilter_score if prefilter_metric == "energy" else math.inf,
                    score_prefilter=prefilter_score,
                    prefilter_metric=prefilter_metric,
                )

                # Keep a max-heap by using negative score. heap[0] is current worst among kept.
                item = (-prefilter_score, counter, cand)
                counter += 1

                if len(heap) < top_m:
                    heapq.heappush(heap, item)
                else:
                    # If candidate is better than current worst, replace.
                    if prefilter_score < -heap[0][0]:
                        heapq.heapreplace(heap, item)

            # Release references promptly.
            del pred_batch, prefilter_scores

    # Convert heap to candidates sorted by prefilter score ascending.
    candidates = [item[2] for item in heap]
    candidates.sort(key=lambda c: c.score_prefilter)
    return candidates


def _rerank_candidates_with_sinkhorn(candidates: List[Candidate], scorer, device: torch.device) -> List[Candidate]:
    if not candidates:
        return []

    states = torch.stack([c.state.to(device=device, dtype=torch.float32) for c in candidates], dim=0)
    sink_scores = scorer.sinkhorn(states).detach().cpu().numpy()
    energy_scores = scorer.energy_distance(states).detach().cpu().numpy()

    for c, s, e in zip(candidates, sink_scores, energy_scores):
        c.score_sinkhorn = float(s)
        c.score_energy_distance = float(e)
        c.adjusted_score = float(s)

    candidates.sort(key=lambda c: c.score_sinkhorn)
    return candidates


def _robust_rerank_enabled(cfg: Dict[str, Any]) -> bool:
    robust_cfg = cfg.get("robustness", {}) or {}
    return bool(robust_cfg.get("enabled", False))


def _sample_start_target(sample: Any) -> Tuple[Any, Any]:
    """Extract start/target arrays from a LoadedCellStates-like object or dict."""
    if isinstance(sample, dict):
        return sample["start_embeddings"], sample["target_embeddings"]
    return getattr(sample, "start_embeddings"), getattr(sample, "target_embeddings")


def _make_scorer_like(reference_scorer, target_state):
    """Build a DistributionScorer with the same scoring parameters as the primary scorer."""
    from .scoring import DistributionScorer

    return DistributionScorer(
        target_state=target_state,
        device=getattr(reference_scorer, "device", None),
        normalize=bool(getattr(reference_scorer, "normalize", True)),
        sinkhorn_metric=getattr(reference_scorer, "sinkhorn_metric", "cosine"),
        sinkhorn_epsilon=float(getattr(reference_scorer, "sinkhorn_epsilon", 0.05)),
        sinkhorn_iters=int(getattr(reference_scorer, "sinkhorn_iters", 100)),
        projection=getattr(reference_scorer, "projection", None),
        projection_auto_metric=bool(getattr(reference_scorer, "projection_auto_metric", True)),
        projection_auto_epsilon=False,
    )


@torch.inference_mode()
def _apply_path(converter, start_embeddings, path: Sequence[str], device: torch.device) -> torch.Tensor:
    """Replay one ordered perturbation path from a fresh sampled start state."""
    state = _to_device_state(start_embeddings, device=device)
    for perturbation in path:
        state = converter.convert_one(state, str(perturbation), return_cpu=False)
    return state


def _aggregate_robust_scores(values: Sequence[float], cfg: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    robust_cfg = cfg.get("robustness", {}) or {}
    aggregation = str(robust_cfg.get("aggregation", "mean_plus_std"))
    std_penalty = float(robust_cfg.get("std_penalty", 0.5))

    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    std = float(np.std(arr, ddof=0))
    worst = float(np.max(arr))

    if aggregation == "mean":
        aggregate = mean
    elif aggregation == "median":
        aggregate = median
    elif aggregation in {"worst", "max"}:
        aggregate = worst
    elif aggregation == "mean_plus_std":
        aggregate = mean + std_penalty * std
    else:
        raise ValueError(
            "robustness.aggregation must be one of: mean, median, worst, mean_plus_std"
        )

    return float(aggregate), {
        "mean": mean,
        "median": median,
        "std": std,
        "worst": worst,
    }


def _robust_rerank_candidates(
    candidates: List[Candidate],
    robust_samples: Sequence[Any],
    converter,
    scorer,
    cfg: Dict[str, Any],
    device: torch.device,
) -> List[Candidate]:
    """
    Rerank the post-Sinkhorn candidate pool by replaying each path on fresh
    start/target samples and aggregating the validation scores.
    """
    if not candidates:
        return []

    robust_cfg = cfg.get("robustness", {}) or {}
    metric = str(robust_cfg.get("metric", "sinkhorn"))
    if metric not in {"sinkhorn", "energy_distance"}:
        raise ValueError("robustness.metric must be 'sinkhorn' or 'energy_distance'")

    prepared_samples = []
    for sample in robust_samples:
        sample_start, sample_target = _sample_start_target(sample)
        prepared_samples.append((sample_start, _make_scorer_like(scorer, sample_target)))

    if not prepared_samples:
        raise ValueError("Robust reranking was enabled, but no robust_samples were provided.")

    print(
        f"robust rerank: {len(candidates)} candidates x {len(prepared_samples)} "
        f"samples using {metric}"
    )

    for cand in candidates:
        scores: List[float] = []
        for sample_start, sample_scorer in prepared_samples:
            state = _apply_path(converter, sample_start, cand.path, device=device)
            if metric == "energy_distance":
                score = sample_scorer.energy_distance(state)
            else:
                score = sample_scorer.sinkhorn(state)
            scores.append(float(score.item()))

        aggregate, stats = _aggregate_robust_scores(scores, cfg)
        cand.robust_score_mean = stats["mean"]
        cand.robust_score_median = stats["median"]
        cand.robust_score_std = stats["std"]
        cand.robust_score_worst = stats["worst"]
        cand.robust_score_aggregate = aggregate
        cand.robust_n_samples = len(scores)
        cand.adjusted_score = aggregate

    candidates.sort(key=lambda c: c.adjusted_score)
    return candidates


def run_search(
    converter,
    scorer,
    start_embeddings,
    cfg: Dict[str, Any],
    perturbations: Optional[Sequence[str]] = None,
    output_dir: Optional[str | Path] = None,
    start_cell: Optional[str] = None,
    target_cell: Optional[str] = None,
    robust_samples: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Main dispatch function.

    cfg["search"]["algorithm"] may be:
        - deterministic_beam
        - diverse_beam
    """
    search_cfg = cfg.get("search", {}) or {}
    algorithm = str(search_cfg.get("algorithm", "deterministic_beam"))

    if algorithm == "deterministic_beam":
        return deterministic_beam_search(
            converter=converter,
            scorer=scorer,
            start_embeddings=start_embeddings,
            cfg=cfg,
            perturbations=perturbations,
            output_dir=output_dir,
            start_cell=start_cell,
            target_cell=target_cell,
            robust_samples=robust_samples,
        )

    if algorithm == "diverse_beam":
        return diverse_beam_search(
            converter=converter,
            scorer=scorer,
            start_embeddings=start_embeddings,
            cfg=cfg,
            perturbations=perturbations,
            output_dir=output_dir,
            start_cell=start_cell,
            target_cell=target_cell,
            robust_samples=robust_samples,
        )

    raise ValueError(f"Unknown search algorithm: {algorithm}")


def deterministic_beam_search(
    converter,
    scorer,
    start_embeddings,
    cfg: Dict[str, Any],
    perturbations: Optional[Sequence[str]] = None,
    output_dir: Optional[str | Path] = None,
    start_cell: Optional[str] = None,
    target_cell: Optional[str] = None,
    robust_samples: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Deterministic beam search.

    At each depth:
        1. Expand each beam node over all allowed perturbations.
        2. Score all candidates with the configured prefilter metric.
        3. Keep top prefilter_multiplier * beam_size candidates.
        4. Rerank those candidates with Sinkhorn OT.
        5. Keep top beam_size candidates.
        6. Save results.tsv and checkpoint.pt.
    """
    return _beam_search_impl(
        algorithm="deterministic_beam",
        converter=converter,
        scorer=scorer,
        start_embeddings=start_embeddings,
        cfg=cfg,
        perturbations=perturbations,
        output_dir=output_dir,
        start_cell=start_cell,
        target_cell=target_cell,
        robust_samples=robust_samples,
        diverse=False,
    )


def diverse_beam_search(
    converter,
    scorer,
    start_embeddings,
    cfg: Dict[str, Any],
    perturbations: Optional[Sequence[str]] = None,
    output_dir: Optional[str | Path] = None,
    start_cell: Optional[str] = None,
    target_cell: Optional[str] = None,
    robust_samples: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """
    Diverse beam search.

    Same expansion/prefilter/rerank as deterministic beam, but final beam
    selection adds:
        - path overlap penalty
        - optional state centroid similarity penalty
    """
    return _beam_search_impl(
        algorithm="diverse_beam",
        converter=converter,
        scorer=scorer,
        start_embeddings=start_embeddings,
        cfg=cfg,
        perturbations=perturbations,
        output_dir=output_dir,
        start_cell=start_cell,
        target_cell=target_cell,
        robust_samples=robust_samples,
        diverse=True,
    )


def _beam_search_impl(
    algorithm: str,
    converter,
    scorer,
    start_embeddings,
    cfg: Dict[str, Any],
    perturbations: Optional[Sequence[str]],
    output_dir: Optional[str | Path],
    start_cell: Optional[str],
    target_cell: Optional[str],
    robust_samples: Optional[Sequence[Any]],
    diverse: bool,
) -> Dict[str, Any]:
    search_cfg = cfg.get("search", {}) or {}
    output_cfg = cfg.get("output", {}) or {}

    max_depth = int(search_cfg.get("max_depth", 5))
    beam_size = int(search_cfg.get("beam_size", 32))
    prefilter_multiplier = int(search_cfg.get("prefilter_multiplier", 10))
    prefilter_metric = _prefilter_metric(cfg)
    prefilter_sinkhorn_iters = _prefilter_sinkhorn_iters(cfg)
    state_checkpoint_dtype = str(output_cfg.get("state_checkpoint_dtype", "float16"))
    use_robust_rerank = _robust_rerank_enabled(cfg)

    if output_dir is None:
        output_dir = output_cfg.get("output_dir", "search_outputs")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config copy for reproducibility.
    save_search_config(cfg, output_dir / "search_config.used.yaml")

    device = getattr(converter, "device", None)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if perturbations is None:
        perturbations = converter.list_perturbations(include_control=False)
    perturbations = filter_allowed_perturbations(perturbations, cfg)

    if len(perturbations) == 0:
        raise ValueError("No perturbations available after filtering.")

    if use_robust_rerank and not robust_samples:
        raise ValueError("Robust reranking is enabled, but robust_samples is empty or None.")

    start_state = _to_device_state(start_embeddings, device=device)

    print("\n=== Search start ===")
    print(f"algorithm: {algorithm}")
    print(f"max_depth: {max_depth}")
    print(f"beam_size: {beam_size}")
    print(f"prefilter_metric: {prefilter_metric}")
    if prefilter_metric == "sinkhorn_low_iter":
        print(f"prefilter_sinkhorn_iters: {prefilter_sinkhorn_iters}")
    print(f"n perturbation labels: {len(perturbations)}")
    if use_robust_rerank:
        robust_cfg = cfg.get("robustness", {}) or {}
        print(
            "robust rerank: enabled "
            f"(n_samples={len(robust_samples)}, "
            f"metric={robust_cfg.get('metric', 'sinkhorn')}, "
            f"aggregation={robust_cfg.get('aggregation', 'mean_plus_std')})"
        )
    print(f"output_dir: {output_dir}")

    # Score the un-perturbed starting state against the target. This baseline is
    # the parent score for depth-1 children, so that delta_sinkhorn_from_parent
    # is finite (and biologically meaningful) for the first drug step.
    with torch.inference_mode():
        baseline_energy_distance = float(scorer.energy_distance(start_state).item())
        baseline_sinkhorn = float(scorer.sinkhorn(start_state).item())

    baseline_scores: Dict[str, float] = {
        "sinkhorn": baseline_sinkhorn,
        "energy_distance": baseline_energy_distance,
        "target_self_term": scorer.target_self_term,
    }
    projection_metadata = scorer.projection_metadata() if hasattr(scorer, "projection_metadata") else None

    print("\n--- Baseline (start state vs target, no drugs) ---")
    print(f"baseline sinkhorn OT:   {baseline_sinkhorn:.6g}")
    print(f"baseline energy dist:   {baseline_energy_distance:.6g}")
    print(f"target self-term E||Y-Y'||: {scorer.target_self_term:.6g}")

    # Root node at depth 0 carries the baseline distances. Its parent score stays
    # None (no parent), so its own delta is NaN, but every depth-1 child inherits
    # the baseline as its parent score and produces a finite delta.
    root_node = SearchNode(
        path=tuple(),
        drug_names=tuple(),
        state=start_state,
        score_sinkhorn=baseline_sinkhorn,
        score_energy_distance=baseline_energy_distance,
        score_prefilter=baseline_energy_distance,
        prefilter_metric="baseline",
        adjusted_score=baseline_sinkhorn,
        parent_score_sinkhorn=None,
    )

    beam = [root_node]

    depth_to_nodes: Dict[int, List[SearchNode]] = {0: [root_node]}
    rows: List[Dict[str, Any]] = [
        node_to_result_row(
            node=root_node,
            algorithm=algorithm,
            depth=0,
            rank=1,
            start_cell=start_cell,
            target_cell=target_cell,
        )
    ]

    for depth in range(1, max_depth + 1):
        t0 = time.perf_counter()
        top_m = max(beam_size, prefilter_multiplier * beam_size)

        print(f"\n--- Depth {depth}/{max_depth} ---")
        print(f"current beam size: {len(beam)}")
        print(f"{prefilter_metric} prefilter top_m: {top_m}")

        candidates = _expand_and_prefilter(
            beam=beam,
            perturbations=perturbations,
            converter=converter,
            scorer=scorer,
            cfg=cfg,
            top_m=top_m,
            device=device,
        )

        if not candidates:
            print("No candidates generated; stopping early.")
            break

        print(f"candidates retained after {prefilter_metric} prefilter: {len(candidates)}")

        candidates = _rerank_candidates_with_sinkhorn(
            candidates=candidates,
            scorer=scorer,
            device=device,
        )

        if use_robust_rerank:
            candidates = _robust_rerank_candidates(
                candidates=candidates,
                robust_samples=robust_samples,
                converter=converter,
                scorer=scorer,
                cfg=cfg,
                device=device,
            )

        if diverse:
            new_beam = _select_diverse_nodes(candidates, beam_size=beam_size, cfg=cfg)
        else:
            new_beam = _select_best_nodes(candidates, beam_size=beam_size)

        beam = new_beam
        depth_to_nodes[depth] = beam

        # Append rows for this depth.
        for rank, node in enumerate(beam, start=1):
            rows.append(
                node_to_result_row(
                    node=node,
                    algorithm=algorithm,
                    depth=depth,
                    rank=rank,
                    start_cell=start_cell,
                    target_cell=target_cell,
                )
            )

        results_path = write_results_tsv(rows, output_dir)
        ckpt_path = save_checkpoint(
            output_dir=output_dir,
            cfg=cfg,
            depth_to_nodes=depth_to_nodes,
            rows=rows,
            state_dtype=state_checkpoint_dtype,
            baseline_scores=baseline_scores,
            projection_metadata=projection_metadata,
        )

        elapsed = time.perf_counter() - t0
        best = beam[0]
        print(f"depth {depth} complete in {elapsed:.2f}s")
        print(f"best sinkhorn: {best.score_sinkhorn:.6g}")
        print(f"best energy:   {best.score_energy_distance:.6g}")
        print(f"best path:     {' -> '.join(best.drug_names)}")
        print(f"wrote: {results_path}")
        print(f"wrote: {ckpt_path}")

    print("\n=== Search complete ===")

    return {
        "algorithm": algorithm,
        "beam": beam,
        "depth_to_nodes": depth_to_nodes,
        "results_rows": rows,
        "baseline_scores": baseline_scores,
        "output_dir": str(output_dir),
        "results_tsv": str(output_dir / "results.tsv"),
        "checkpoint": str(output_dir / "checkpoint.pt"),
    }
