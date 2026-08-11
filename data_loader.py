#!/usr/bin/env python
"""
data_loader.py

Utilities for loading start and target cell-state embeddings from an AnnData .h5ad
that already contains SE embeddings in adata.obsm[embed_key], usually "X_state".

This module is intentionally independent of the ST-SE converter and scoring modules.

Typical use:
    from data_loader import load_start_target_embeddings

    pair = load_start_target_embeddings(
        h5ad_path="WT_256_per_cell_name.SE600M.h5ad",
        start_cell="J82",
        target_cell="A-172",
        cell_col="cell_name",
        embed_key="X_state",
        start_sample=256,
        target_sample=256,
        seed=42,
    )

    start_embeddings = pair.start_embeddings  # np.ndarray [256, 2058]
    target_embeddings = pair.target_embeddings  # np.ndarray [256, 2058]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional, Sequence, Union, Dict, Any, Tuple

import json
import numpy as np
import scanpy as sc


SampleSpec = Union[int, Literal["all"]]


@dataclass
class LoadedCellStates:
    """Container returned by load_start_target_embeddings()."""

    start_embeddings: np.ndarray
    target_embeddings: np.ndarray
    start_cell: str
    target_cell: str
    cell_col: str
    embed_key: str
    start_obs_names: list[str]
    target_obs_names: list[str]
    start_n_available: int
    target_n_available: int
    start_n_sampled: int
    target_n_sampled: int
    seed: int
    replace_start: bool
    replace_target: bool
    batch_selection: str = "standard"
    batch_selection_candidate_index: Optional[int] = None
    batch_selection_rank: Optional[int] = None
    batch_selection_score_sinkhorn_ot: Optional[float] = None
    batch_selection_score_energy_distance: Optional[float] = None
    batch_selection_adjusted_score: Optional[float] = None
    batch_selection_overlap_fraction: Optional[float] = None
    start_seed_obs_name: Optional[str] = None
    target_seed_obs_name: Optional[str] = None

    def metadata(self) -> Dict[str, Any]:
        """Return JSON-serializable metadata, excluding the embedding arrays."""
        d = asdict(self)
        d.pop("start_embeddings", None)
        d.pop("target_embeddings", None)
        return d

    def save_npz(self, output_npz: str | Path) -> None:
        """
        Save start/target embeddings and metadata in a compact NumPy archive.

        This is useful for:
          - caching the extracted start/target state
          - feeding the same target state into multiple scoring/search runs
          - making the search workflow reproducible

        Note:
          For the beam search inner loop, keep embeddings in memory. Use this file
          for checkpointing / reproducibility, not per-candidate I/O.
        """
        output_npz = Path(output_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_npz,
            start_embeddings=self.start_embeddings.astype(np.float32, copy=False),
            target_embeddings=self.target_embeddings.astype(np.float32, copy=False),
            metadata=json.dumps(self.metadata()),
            start_obs_names=np.asarray(self.start_obs_names, dtype=object),
            target_obs_names=np.asarray(self.target_obs_names, dtype=object),
        )

    @staticmethod
    def load_npz(input_npz: str | Path) -> "LoadedCellStates":
        """Load a cache created by save_npz()."""
        z = np.load(input_npz, allow_pickle=True)
        metadata = json.loads(str(z["metadata"].item()))

        return LoadedCellStates(
            start_embeddings=z["start_embeddings"].astype(np.float32, copy=False),
            target_embeddings=z["target_embeddings"].astype(np.float32, copy=False),
            start_obs_names=list(z["start_obs_names"].astype(str)),
            target_obs_names=list(z["target_obs_names"].astype(str)),
            **{
                k: v
                for k, v in metadata.items()
                if k not in {"start_obs_names", "target_obs_names"}
            },
        )


def _parse_sample_spec(value: SampleSpec) -> SampleSpec:
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "all":
            return "all"
        try:
            ivalue = int(value)
        except ValueError as exc:
            raise ValueError(f"Sample spec must be an integer or 'all', got {value!r}") from exc
        return ivalue
    return int(value)


def _sample_indices(
    available_indices: np.ndarray,
    sample: SampleSpec,
    rng: np.random.Generator,
    replace_if_needed: bool,
    label: str,
) -> tuple[np.ndarray, bool]:
    """
    Sample indices reproducibly.

    If sample == "all", all available indices are returned.
    If sample is an integer larger than available cells:
      - if replace_if_needed=True, sample with replacement
      - otherwise raise ValueError
    """
    sample = _parse_sample_spec(sample)

    if len(available_indices) == 0:
        raise ValueError(f"No cells available for {label}")

    if sample == "all":
        return available_indices.copy(), False

    if sample <= 0:
        raise ValueError(f"Sample size for {label} must be positive or 'all', got {sample}")

    if len(available_indices) >= sample:
        chosen = rng.choice(available_indices, size=sample, replace=False)
        return chosen, False

    if not replace_if_needed:
        raise ValueError(
            f"Requested {sample} cells for {label}, but only {len(available_indices)} are available. "
            "Set replace_if_needed=True to sample with replacement."
        )

    chosen = rng.choice(available_indices, size=sample, replace=True)
    return chosen, True


def load_start_target_embeddings(
    h5ad_path: str | Path,
    start_cell: str,
    target_cell: str,
    cell_col: str = "cell_name",
    embed_key: str = "X_state",
    start_sample: SampleSpec = 256,
    target_sample: SampleSpec = 256,
    seed: int = 42,
    replace_if_needed: bool = True,
    dtype: np.dtype = np.float32,
    start_index_pool: Optional[np.ndarray] = None,
    target_index_pool: Optional[np.ndarray] = None,
) -> LoadedCellStates:
    """
    Load start and target cell-state embeddings from an h5ad file.

    Parameters
    ----------
    h5ad_path:
        Path to AnnData object with SE embeddings in adata.obsm[embed_key].
    start_cell:
        Starting cell type / cell line label, for example "J82".
    target_cell:
        Target cell type / cell line label, for example "A-172".
    cell_col:
        adata.obs column containing cell-type / cell-line labels.
    embed_key:
        adata.obsm key containing SE embeddings, usually "X_state".
    start_sample:
        Number of starting cells to sample, or "all".
    target_sample:
        Number of target cells to sample, or "all".
        For scoring, target_sample can be much larger than start_sample if the
        h5ad contains more target cells.
    seed:
        Random seed for reproducible sampling.
    replace_if_needed:
        If True, sample with replacement when requested sample size exceeds the
        number of available cells.
    dtype:
        Output dtype for embeddings.
    start_index_pool / target_index_pool:
        Optional arrays of allowed absolute row indices. When provided, sampling
        is restricted to the intersection of the label mask and the pool. This
        enables held-out projection fit/eval splits without leaking cells.

    Returns
    -------
    LoadedCellStates
        start_embeddings: np.ndarray [start_sample, emb_dim]
        target_embeddings: np.ndarray [target_sample, emb_dim]
    """
    h5ad_path = Path(h5ad_path)
    rng = np.random.default_rng(seed)

    ad = sc.read_h5ad(h5ad_path)

    return _load_start_target_embeddings_from_adata(
        ad=ad,
        h5ad_path=h5ad_path,
        start_cell=start_cell,
        target_cell=target_cell,
        cell_col=cell_col,
        embed_key=embed_key,
        start_sample=start_sample,
        target_sample=target_sample,
        seed=seed,
        rng=rng,
        replace_if_needed=replace_if_needed,
        dtype=dtype,
        start_index_pool=start_index_pool,
        target_index_pool=target_index_pool,
    )


def _load_start_target_embeddings_from_adata(
    *,
    ad,
    h5ad_path: str | Path,
    start_cell: str,
    target_cell: str,
    cell_col: str,
    embed_key: str,
    start_sample: SampleSpec,
    target_sample: SampleSpec,
    seed: int,
    rng: np.random.Generator,
    replace_if_needed: bool,
    dtype: np.dtype,
    start_index_pool: Optional[np.ndarray] = None,
    target_index_pool: Optional[np.ndarray] = None,
) -> LoadedCellStates:
    """
    Shared implementation for one or many reproducible start/target samples.

    start_index_pool / target_index_pool:
        Optional arrays of allowed absolute row indices. When provided, sampling
        is restricted to the intersection of the label mask and the pool. This
        enables held-out fit/eval splits without leaking cells between sets.
    """

    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available columns: {list(ad.obs.columns)}")

    if embed_key not in ad.obsm:
        raise KeyError(f"embed_key={embed_key!r} not found in adata.obsm. Available keys: {list(ad.obsm.keys())}")

    labels = ad.obs[cell_col].astype(str).values
    start_mask = labels == str(start_cell)
    target_mask = labels == str(target_cell)

    start_available = np.where(start_mask)[0]
    target_available = np.where(target_mask)[0]

    if start_index_pool is not None:
        start_available = np.intersect1d(start_available, np.asarray(start_index_pool))
    if target_index_pool is not None:
        target_available = np.intersect1d(target_available, np.asarray(target_index_pool))

    if len(start_available) == 0:
        examples = sorted(set(labels))[:20]
        raise ValueError(f"No cells found for start_cell={start_cell!r} in {cell_col!r}. Example labels: {examples}")

    if len(target_available) == 0:
        examples = sorted(set(labels))[:20]
        raise ValueError(f"No cells found for target_cell={target_cell!r} in {cell_col!r}. Example labels: {examples}")

    start_idx, replace_start = _sample_indices(
        start_available,
        sample=start_sample,
        rng=rng,
        replace_if_needed=replace_if_needed,
        label=f"start_cell={start_cell}",
    )

    target_idx, replace_target = _sample_indices(
        target_available,
        sample=target_sample,
        rng=rng,
        replace_if_needed=replace_if_needed,
        label=f"target_cell={target_cell}",
    )

    X_state = np.asarray(ad.obsm[embed_key])

    start_embeddings = X_state[start_idx].astype(dtype, copy=True)
    target_embeddings = X_state[target_idx].astype(dtype, copy=True)

    return LoadedCellStates(
        start_embeddings=start_embeddings,
        target_embeddings=target_embeddings,
        start_cell=str(start_cell),
        target_cell=str(target_cell),
        cell_col=str(cell_col),
        embed_key=str(embed_key),
        start_obs_names=list(ad.obs_names[start_idx].astype(str)),
        target_obs_names=list(ad.obs_names[target_idx].astype(str)),
        start_n_available=int(len(start_available)),
        target_n_available=int(len(target_available)),
        start_n_sampled=int(start_embeddings.shape[0]),
        target_n_sampled=int(target_embeddings.shape[0]),
        seed=int(seed),
        replace_start=bool(replace_start),
        replace_target=bool(replace_target),
    )


def load_start_target_embedding_batches(
    h5ad_path: str | Path,
    start_cell: str,
    target_cell: str,
    *,
    n_batches: int,
    cell_col: str = "cell_name",
    embed_key: str = "X_state",
    start_sample: SampleSpec = 256,
    target_sample: SampleSpec = 256,
    seed: int = 42,
    seed_offset: int = 1000,
    replace_if_needed: bool = True,
    dtype: np.dtype = np.float32,
    start_index_pool: Optional[np.ndarray] = None,
    target_index_pool: Optional[np.ndarray] = None,
) -> list[LoadedCellStates]:
    """
    Load multiple independent start/target samples from one h5ad read.

    This is intended for robust path reranking: each batch is a fresh sampled
    start distribution and target distribution, with deterministic seeds
    seed + seed_offset + batch_index.

    start_index_pool / target_index_pool:
        Optional arrays of allowed absolute row indices to restrict sampling
        (e.g. an evaluation half for held-out projection tests).
    """
    if n_batches <= 0:
        raise ValueError(f"n_batches must be positive, got {n_batches}")

    h5ad_path = Path(h5ad_path)
    ad = sc.read_h5ad(h5ad_path)

    batches: list[LoadedCellStates] = []
    for i in range(int(n_batches)):
        batch_seed = int(seed) + int(seed_offset) + i
        rng = np.random.default_rng(batch_seed)
        batches.append(
            _load_start_target_embeddings_from_adata(
                ad=ad,
                h5ad_path=h5ad_path,
                start_cell=start_cell,
                target_cell=target_cell,
                cell_col=cell_col,
                embed_key=embed_key,
                start_sample=start_sample,
                target_sample=target_sample,
                seed=batch_seed,
                rng=rng,
                replace_if_needed=replace_if_needed,
                dtype=dtype,
                start_index_pool=start_index_pool,
                target_index_pool=target_index_pool,
            )
        )
    return batches


def _label_available_indices(
    *,
    ad,
    start_cell: str,
    target_cell: str,
    cell_col: str,
    embed_key: str,
    start_index_pool: Optional[np.ndarray] = None,
    target_index_pool: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available columns: {list(ad.obs.columns)}")

    if embed_key not in ad.obsm:
        raise KeyError(f"embed_key={embed_key!r} not found in adata.obsm. Available keys: {list(ad.obsm.keys())}")

    labels = ad.obs[cell_col].astype(str).values
    start_available = np.where(labels == str(start_cell))[0]
    target_available = np.where(labels == str(target_cell))[0]

    if start_index_pool is not None:
        start_available = np.intersect1d(start_available, np.asarray(start_index_pool))
    if target_index_pool is not None:
        target_available = np.intersect1d(target_available, np.asarray(target_index_pool))

    if len(start_available) == 0:
        examples = sorted(set(labels))[:20]
        raise ValueError(f"No cells found for start_cell={start_cell!r} in {cell_col!r}. Example labels: {examples}")

    if len(target_available) == 0:
        examples = sorted(set(labels))[:20]
        raise ValueError(f"No cells found for target_cell={target_cell!r} in {cell_col!r}. Example labels: {examples}")

    return start_available, target_available


def _nearest_neighbor_indices(
    *,
    X: np.ndarray,
    available_indices: np.ndarray,
    seed_index: int,
    sample: SampleSpec,
    replace_if_needed: bool,
    label: str,
) -> Tuple[np.ndarray, bool]:
    sample = _parse_sample_spec(sample)
    if sample == "all":
        return available_indices.copy(), False
    if sample <= 0:
        raise ValueError(f"Sample size for {label} must be positive or 'all', got {sample}")

    seed_vec = X[int(seed_index)]
    candidates = X[available_indices]
    distances = np.sum((candidates - seed_vec) ** 2, axis=1)
    ordered = available_indices[np.argsort(distances, kind="stable")]

    if len(ordered) >= int(sample):
        return ordered[: int(sample)].copy(), False

    if not replace_if_needed:
        raise ValueError(
            f"Requested {sample} nearest-neighbor cells for {label}, but only {len(ordered)} are available. "
            "Set replace_if_needed=True to allow repeated nearest cells."
        )

    repeats = int(np.ceil(int(sample) / max(len(ordered), 1)))
    chosen = np.tile(ordered, repeats)[: int(sample)]
    return chosen.copy(), True


def _score_candidate_batches(
    *,
    X_state: np.ndarray,
    candidates: Sequence[Dict[str, Any]],
    normalize: bool,
    sinkhorn_metric: str,
    sinkhorn_epsilon: float,
    sinkhorn_iters: int,
    device: Optional[str],
    projection=None,
    projection_auto_epsilon: bool = False,
    chunk_size: int = 8,
) -> None:
    from scoring import energy_distance, sinkhorn_ot_distance

    chunk_size = max(1, int(chunk_size))
    metric = str(sinkhorn_metric)
    should_normalize = bool(normalize)
    if projection is not None:
        should_normalize = False
        if metric == "cosine":
            metric = "sqeuclidean"

    for start in range(0, len(candidates), chunk_size):
        chunk = candidates[start : start + chunk_size]
        start_batch = np.stack([X_state[row["start_indices"]] for row in chunk], axis=0).astype(np.float32, copy=False)
        target_batch = np.stack([X_state[row["target_indices"]] for row in chunk], axis=0).astype(np.float32, copy=False)

        if projection is not None:
            start_batch = projection.transform(start_batch)
            target_batch = projection.transform(target_batch)

        if projection is not None and projection_auto_epsilon:
            from projections import estimate_sinkhorn_epsilon

            sinkhorn_values = []
            for candidate_i in range(start_batch.shape[0]):
                auto_epsilon = estimate_sinkhorn_epsilon(
                    target_batch[candidate_i],
                    metric=metric,
                    scale=0.1,
                    device=device,
                )
                score = sinkhorn_ot_distance(
                    predicted_states=start_batch[candidate_i],
                    target_state=target_batch[candidate_i],
                    metric=metric,
                    normalize=should_normalize,
                    epsilon=auto_epsilon,
                    n_iters=int(sinkhorn_iters),
                    device=device,
                )
                sinkhorn_values.append(float(score.detach().cpu().item()))
            sinkhorn = np.asarray(sinkhorn_values, dtype=np.float32)
        else:
            sinkhorn = sinkhorn_ot_distance(
                predicted_states=start_batch,
                target_state=target_batch,
                metric=metric,
                normalize=should_normalize,
                epsilon=float(sinkhorn_epsilon),
                n_iters=int(sinkhorn_iters),
                device=device,
            ).detach().cpu().numpy()
        energy = energy_distance(
            predicted_states=start_batch,
            target_state=target_batch,
            normalize=should_normalize,
            device=device,
        ).detach().cpu().numpy()

        for row, sinkhorn_score, energy_score in zip(chunk, sinkhorn, energy):
            row["score_sinkhorn_ot"] = float(sinkhorn_score)
            row["score_energy_distance"] = float(energy_score)


def _overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    a_unique = set(np.asarray(a).astype(int).tolist())
    b_unique = set(np.asarray(b).astype(int).tolist())
    denom = max(1, min(len(a_unique), len(b_unique)))
    return float(len(a_unique & b_unique) / denom)


def _select_high_sensitivity_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    n_batches: int,
    overlap_penalty: float,
) -> list[Dict[str, Any]]:
    remaining = [dict(row) for row in candidates]
    selected: list[Dict[str, Any]] = []
    finite_scores = [float(row["score_sinkhorn_ot"]) for row in remaining if np.isfinite(float(row["score_sinkhorn_ot"]))]
    score_scale = float(np.nanmedian(finite_scores)) if finite_scores else 1.0
    penalty_scale = max(0.0, float(overlap_penalty)) * max(abs(score_scale), 1e-12)

    while remaining and len(selected) < int(n_batches):
        best_i = None
        best_key = None
        fallback_i = None
        fallback_key = None
        fallback_adjusted = None
        for i, row in enumerate(remaining):
            max_overlap = 0.0
            exact_duplicate = False
            for chosen in selected:
                start_same = np.array_equal(row["start_indices"], chosen["start_indices"])
                target_same = np.array_equal(row["target_indices"], chosen["target_indices"])
                if start_same and target_same:
                    exact_duplicate = True
                    break
                start_overlap = _overlap_fraction(row["start_indices"], chosen["start_indices"])
                target_overlap = _overlap_fraction(row["target_indices"], chosen["target_indices"])
                max_overlap = max(max_overlap, 0.5 * (start_overlap + target_overlap))

            fallback_overlap = 1.0 if exact_duplicate else max_overlap
            fallback_score = float(row["score_sinkhorn_ot"]) - penalty_scale * fallback_overlap
            fallback_candidate_key = (
                fallback_score,
                float(row["score_sinkhorn_ot"]),
                -float(row["score_energy_distance"]),
                -int(row["candidate_index"]),
            )
            if fallback_key is None or fallback_candidate_key > fallback_key:
                fallback_i = i
                fallback_key = fallback_candidate_key
                fallback_adjusted = fallback_score

            if exact_duplicate:
                continue

            adjusted = float(row["score_sinkhorn_ot"]) - penalty_scale * max_overlap
            key = (adjusted, float(row["score_sinkhorn_ot"]), -float(row["score_energy_distance"]), -int(row["candidate_index"]))
            if best_key is None or key > best_key:
                best_i = i
                best_key = key
                row["_adjusted_score"] = adjusted
                row["_overlap_fraction"] = max_overlap

        if best_i is None:
            if fallback_i is None:
                break
            remaining[int(fallback_i)]["_adjusted_score"] = float(fallback_adjusted)
            remaining[int(fallback_i)]["_overlap_fraction"] = 1.0
            selected.append(remaining.pop(int(fallback_i)))
            continue
        selected.append(remaining.pop(int(best_i)))

    return selected


def load_high_sensitivity_start_target_embedding_batches(
    h5ad_path: str | Path,
    start_cell: str,
    target_cell: str,
    *,
    n_batches: int,
    n_candidates: int = 300,
    overlap_penalty: float = 0.02,
    score_chunk_size: int = 8,
    cell_col: str = "cell_name",
    embed_key: str = "X_state",
    start_sample: SampleSpec = 256,
    target_sample: SampleSpec = 256,
    seed: int = 42,
    seed_offset: int = 1000,
    replace_if_needed: bool = True,
    dtype: np.dtype = np.float32,
    start_index_pool: Optional[np.ndarray] = None,
    target_index_pool: Optional[np.ndarray] = None,
    normalize: bool = True,
    sinkhorn_metric: str = "cosine",
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 100,
    device: Optional[str] = None,
    projection=None,
    projection_auto_epsilon: bool = False,
) -> tuple[list[LoadedCellStates], "Any"]:
    """
    Build local nearest-neighbor start/target batches, score them by baseline OT,
    and return the highest-separation non-identical candidates.

    Diversity is intentionally soft: overlap only subtracts a small penalty from
    the OT score during greedy selection, so small cell populations can reuse
    cells when necessary.
    """
    if n_batches <= 0:
        raise ValueError(f"n_batches must be positive, got {n_batches}")
    if n_candidates <= 0:
        raise ValueError(f"n_candidates must be positive, got {n_candidates}")

    h5ad_path = Path(h5ad_path)
    ad = sc.read_h5ad(h5ad_path)
    start_available, target_available = _label_available_indices(
        ad=ad,
        start_cell=start_cell,
        target_cell=target_cell,
        cell_col=cell_col,
        embed_key=embed_key,
        start_index_pool=start_index_pool,
        target_index_pool=target_index_pool,
    )
    X_state = np.asarray(ad.obsm[embed_key], dtype=np.float32)
    rng = np.random.default_rng(int(seed) + int(seed_offset))

    candidates: list[Dict[str, Any]] = []
    for candidate_index in range(int(n_candidates)):
        start_seed = int(rng.choice(start_available))
        target_seed = int(rng.choice(target_available))
        start_idx, replace_start = _nearest_neighbor_indices(
            X=X_state,
            available_indices=start_available,
            seed_index=start_seed,
            sample=start_sample,
            replace_if_needed=replace_if_needed,
            label=f"start_cell={start_cell}",
        )
        target_idx, replace_target = _nearest_neighbor_indices(
            X=X_state,
            available_indices=target_available,
            seed_index=target_seed,
            sample=target_sample,
            replace_if_needed=replace_if_needed,
            label=f"target_cell={target_cell}",
        )
        candidates.append(
            {
                "candidate_index": int(candidate_index),
                "seed": int(seed) + int(seed_offset) + int(candidate_index),
                "start_seed_index": int(start_seed),
                "target_seed_index": int(target_seed),
                "start_seed_obs_name": str(ad.obs_names[start_seed]),
                "target_seed_obs_name": str(ad.obs_names[target_seed]),
                "start_indices": start_idx,
                "target_indices": target_idx,
                "replace_start": bool(replace_start),
                "replace_target": bool(replace_target),
            }
        )

    _score_candidate_batches(
        X_state=X_state,
        candidates=candidates,
        normalize=normalize,
        sinkhorn_metric=sinkhorn_metric,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        device=device,
        projection=projection,
        projection_auto_epsilon=bool(projection_auto_epsilon),
        chunk_size=score_chunk_size,
    )
    selected = _select_high_sensitivity_candidates(
        candidates,
        n_batches=int(n_batches),
        overlap_penalty=float(overlap_penalty),
    )

    selected_candidate_ids = {int(row["candidate_index"]): rank for rank, row in enumerate(selected, start=1)}
    selected_by_id = {int(row["candidate_index"]): row for row in selected}
    candidate_rows = []
    for row in candidates:
        selected_rank = selected_candidate_ids.get(int(row["candidate_index"]))
        selected_row = selected_by_id.get(int(row["candidate_index"]), {})
        candidate_rows.append(
            {
                "candidate_index": int(row["candidate_index"]),
                "selected_rank": selected_rank,
                "selected": selected_rank is not None,
                "score_sinkhorn_ot": float(row["score_sinkhorn_ot"]),
                "score_energy_distance": float(row["score_energy_distance"]),
                "selection_adjusted_score": selected_row.get("_adjusted_score"),
                "selection_overlap_fraction": selected_row.get("_overlap_fraction"),
                "start_seed_obs_name": str(row["start_seed_obs_name"]),
                "target_seed_obs_name": str(row["target_seed_obs_name"]),
                "start_n_sampled": int(len(row["start_indices"])),
                "target_n_sampled": int(len(row["target_indices"])),
                "replace_start": bool(row["replace_start"]),
                "replace_target": bool(row["replace_target"]),
            }
        )

    batches: list[LoadedCellStates] = []
    for rank, row in enumerate(selected, start=1):
        start_idx = row["start_indices"]
        target_idx = row["target_indices"]
        batches.append(
            LoadedCellStates(
                start_embeddings=X_state[start_idx].astype(dtype, copy=True),
                target_embeddings=X_state[target_idx].astype(dtype, copy=True),
                start_cell=str(start_cell),
                target_cell=str(target_cell),
                cell_col=str(cell_col),
                embed_key=str(embed_key),
                start_obs_names=list(ad.obs_names[start_idx].astype(str)),
                target_obs_names=list(ad.obs_names[target_idx].astype(str)),
                start_n_available=int(len(start_available)),
                target_n_available=int(len(target_available)),
                start_n_sampled=int(len(start_idx)),
                target_n_sampled=int(len(target_idx)),
                seed=int(row["seed"]),
                replace_start=bool(row["replace_start"]),
                replace_target=bool(row["replace_target"]),
                batch_selection="high-sensitivity",
                batch_selection_candidate_index=int(row["candidate_index"]),
                batch_selection_rank=int(rank),
                batch_selection_score_sinkhorn_ot=float(row["score_sinkhorn_ot"]),
                batch_selection_score_energy_distance=float(row["score_energy_distance"]),
                batch_selection_adjusted_score=float(row.get("_adjusted_score", row["score_sinkhorn_ot"])),
                batch_selection_overlap_fraction=float(row.get("_overlap_fraction", 0.0)),
                start_seed_obs_name=str(row["start_seed_obs_name"]),
                target_seed_obs_name=str(row["target_seed_obs_name"]),
            )
        )

    try:
        import pandas as pd

        candidate_df = pd.DataFrame(candidate_rows)
        if not candidate_df.empty:
            candidate_df = candidate_df.sort_values(
                ["selected", "selected_rank", "score_sinkhorn_ot"],
                ascending=[False, True, False],
            ).reset_index(drop=True)
    except Exception:
        candidate_df = candidate_rows

    return batches, candidate_df


def list_cell_counts(
    h5ad_path: str | Path,
    cell_col: str = "cell_name",
) -> "np.ndarray":
    """
    Convenience utility to print and return counts per cell label.
    """
    ad = sc.read_h5ad(h5ad_path)
    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs. Available columns: {list(ad.obs.columns)}")
    counts = ad.obs[cell_col].astype(str).value_counts()
    return counts
