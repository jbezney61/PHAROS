#!/usr/bin/env python
"""
embedding_manifold_qc.py

Reusable FAISS-based manifold support diagnostics for ST-SE embeddings.

This module intentionally does not depend on the ST-SE converter. It treats
adata.obsm[embed_key] as the embedding manifold and asks whether query cell
states have reference-neighborhood support.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


LOGGER_NAME = "embedding_manifold_qc"
PRIMARY_SCORE = "local_density_ratio"


@dataclass
class ReferenceBuildParams:
    reference_h5ad: str
    output_dir: str
    reference_cell_col: str = "cell_name"
    reference_perturbation_col: str = "drugname_drugconc"
    reference_state_col: Optional[str] = None
    embed_key: str = "X_state"
    metric: str = "l2"
    k: int = 50
    seed: int = 42
    add_batch_size: int = 100_000
    search_batch_size: int = 16_384
    calibration_cells_per_state: int = 100
    calibration_splits: int = 3
    calibration_state_fraction: float = 0.10
    calibration_max_states_per_split: int = 250
    calibration_cell_names_per_split: int = 5
    calibration_max_cell_name_states_per_split: int = 250
    skip_heldout_state_calibration: bool = False
    skip_heldout_cell_name_calibration: bool = False
    cell_line_metadata: Optional[str] = None
    gpu_id: int = 0
    require_gpu: bool = True
    save_faiss_index: bool = False
    max_reference_cells: Optional[int] = None


@dataclass
class QueryScoreParams:
    reference_dir: str
    query_h5ad: str
    output_dir: str
    query_state_col: str = "cell_type"
    embed_key: str = "X_state"
    k: Optional[int] = None
    query_cells_per_state: int = 100
    seed: int = 42
    add_batch_size: int = 100_000
    search_batch_size: int = 16_384
    gpu_id: int = 0
    require_gpu: bool = True
    allow_k_mismatch: bool = False
    use_saved_index: bool = False
    save_query_neighbors: bool = False


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def setup_logger(output_dir: str | Path, log_name: str = "manifold_qc.log") -> logging.Logger:
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


def import_faiss():
    try:
        import faiss  # type: ignore
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError(
            "FAISS is required for embedding manifold QC. Install a GPU-enabled FAISS build, "
            "for example: conda install -c pytorch -c nvidia -c conda-forge faiss-gpu=1.14.2"
        ) from exc
    return faiss


def log_gpu_status(faiss: Any, *, gpu_id: int, require_gpu: bool, logger: logging.Logger) -> None:
    n_gpu = int(faiss.get_num_gpus())
    compile_options = getattr(faiss, "get_compile_options", lambda: "unknown")()
    logger.info("FAISS compile options: %s", compile_options)
    logger.info("FAISS visible GPUs: %d", n_gpu)
    if n_gpu <= int(gpu_id):
        msg = f"Requested FAISS GPU id {gpu_id}, but FAISS reports {n_gpu} visible GPU(s)."
        if require_gpu:
            raise RuntimeError(msg)
        logger.warning("%s Falling back to CPU FAISS.", msg)
    else:
        logger.info("Using FAISS GPU id %d.", int(gpu_id))

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("PyTorch CUDA available: true")
            logger.info("PyTorch CUDA device %d: %s", int(gpu_id), torch.cuda.get_device_name(int(gpu_id)))
            logger.info("PyTorch CUDA capability: %s", torch.cuda.get_device_capability(int(gpu_id)))
        else:
            logger.warning("PyTorch CUDA available: false")
    except Exception as exc:
        logger.info("PyTorch CUDA device check skipped: %s", exc)


def safe_prepare_dir(path: str | Path, *, overwrite: bool = False) -> Path:
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


def write_table(df: pd.DataFrame, preferred_path: str | Path, *, logger: Optional[logging.Logger] = None) -> Path:
    preferred_path = Path(preferred_path)
    preferred_path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_path.suffix == ".parquet":
        try:
            df.to_parquet(preferred_path, index=False)
            return preferred_path
        except Exception as exc:
            fallback = preferred_path.with_suffix(".tsv")
            if logger:
                logger.warning("Could not write parquet %s (%s); writing %s instead.", preferred_path, exc, fallback)
            df.to_csv(fallback, sep="\t", index=False)
            return fallback
    df.to_csv(preferred_path, sep="\t", index=False)
    return preferred_path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t")


def finite_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def empirical_percentile(scores: np.ndarray | Sequence[float], calibration_sorted: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    calibration_sorted = finite_array(calibration_sorted)
    if len(calibration_sorted) == 0:
        return np.full(scores.shape, np.nan, dtype=float)
    calibration_sorted = np.sort(calibration_sorted)
    return 100.0 * np.searchsorted(calibration_sorted, scores, side="right") / float(len(calibration_sorted))


def quantile(values: Sequence[float], q: float) -> float:
    arr = finite_array(values)
    if len(arr) == 0:
        return math.nan
    return float(np.quantile(arr, q))


def load_h5ad(path: str | Path, *, logger: Optional[logging.Logger] = None):
    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover - environment specific
        raise ImportError("anndata is required to read h5ad files.") from exc
    if logger:
        logger.info("Opening h5ad in backed read mode: %s", path)
    return ad.read_h5ad(path, backed="r")


def obsm_shape(adata: Any, embed_key: str) -> Tuple[int, int]:
    if embed_key not in adata.obsm:
        raise KeyError(f"Embedding key {embed_key!r} not found in adata.obsm. Available keys: {list(adata.obsm.keys())}")
    x = adata.obsm[embed_key]
    shape = tuple(x.shape)
    if len(shape) != 2:
        raise ValueError(f"adata.obsm[{embed_key!r}] must be 2D, got shape {shape}")
    return int(shape[0]), int(shape[1])


def to_dense_float32(x: Any) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D embedding batch, got shape {arr.shape}")
    return np.ascontiguousarray(arr)


def transform_embeddings(x: np.ndarray, metric: str) -> np.ndarray:
    x = to_dense_float32(x)
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x = x / norms
    elif metric != "l2":
        raise ValueError("metric must be 'l2' or 'cosine'")
    return np.ascontiguousarray(x, dtype=np.float32)


def faiss_distances_to_scores(distances: np.ndarray, metric: str) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float32)
    if metric == "l2":
        return np.sqrt(np.maximum(distances, 0.0)).astype(np.float32, copy=False)
    if metric == "cosine":
        return (1.0 - distances).astype(np.float32, copy=False)
    raise ValueError("metric must be 'l2' or 'cosine'")


def make_flat_index(faiss: Any, dim: int, metric: str):
    if metric == "l2":
        return faiss.IndexFlatL2(int(dim))
    if metric == "cosine":
        return faiss.IndexFlatIP(int(dim))
    raise ValueError("metric must be 'l2' or 'cosine'")


def create_faiss_index(
    *,
    embeddings: np.ndarray,
    metric: str,
    add_batch_size: int,
    gpu_id: int,
    require_gpu: bool,
    logger: logging.Logger,
    subset_indices: Optional[np.ndarray] = None,
):
    faiss = import_faiss()
    dim = int(embeddings.shape[1])
    cpu_index = make_flat_index(faiss, dim, metric)
    resources = None
    index = cpu_index

    if int(faiss.get_num_gpus()) > int(gpu_id):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, int(gpu_id), cpu_index)
        logger.info("Created GPU FAISS flat index: metric=%s dim=%d gpu=%d", metric, dim, int(gpu_id))
    elif require_gpu:
        raise RuntimeError("FAISS GPU is required but no suitable GPU is visible.")
    else:
        logger.warning("Using CPU FAISS flat index: metric=%s dim=%d", metric, dim)

    n_add = int(len(subset_indices)) if subset_indices is not None else int(embeddings.shape[0])
    add_batch_size = max(1, int(add_batch_size))
    logger.info("Adding %d reference vectors to FAISS index in batches of %d.", n_add, add_batch_size)
    for start in range(0, n_add, add_batch_size):
        end = min(start + add_batch_size, n_add)
        if subset_indices is None:
            xb = np.ascontiguousarray(embeddings[start:end], dtype=np.float32)
        else:
            xb = np.ascontiguousarray(embeddings[subset_indices[start:end]], dtype=np.float32)
        index.add(xb)
        if start == 0 or end == n_add or end % max(add_batch_size * 5, 1) == 0:
            logger.info("  FAISS add progress: %d/%d", end, n_add)
    return index, resources


def search_index(
    index: Any,
    queries: np.ndarray,
    *,
    k: int,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal <= 0:
        raise ValueError("Cannot search an empty FAISS index.")
    k_eff = min(int(k), ntotal)
    raw_dist, neigh = index.search(np.ascontiguousarray(queries, dtype=np.float32), k_eff)
    return faiss_distances_to_scores(raw_dist, metric), neigh.astype(np.int64, copy=False)


def read_obsm_rows(adata: Any, embed_key: str, indices: Sequence[int], *, metric: str) -> np.ndarray:
    indices_arr = np.asarray(indices, dtype=np.int64)
    if len(indices_arr) == 0:
        _, dim = obsm_shape(adata, embed_key)
        return np.empty((0, dim), dtype=np.float32)

    unique_indices, inverse = np.unique(indices_arr, return_inverse=True)
    x = adata.obsm[embed_key]
    try:
        batch = x[unique_indices]
    except Exception:
        pieces = [x[int(i) : int(i) + 1] for i in unique_indices]
        batch = np.vstack([to_dense_float32(p) for p in pieces])
    batch = transform_embeddings(batch, metric)
    return batch[inverse]


def iter_obsm_contiguous(
    adata: Any,
    embed_key: str,
    *,
    metric: str,
    batch_size: int,
    n_rows: Optional[int] = None,
) -> Iterable[Tuple[int, int, np.ndarray]]:
    n_total, _ = obsm_shape(adata, embed_key)
    n_rows = n_total if n_rows is None else min(int(n_rows), n_total)
    batch_size = max(1, int(batch_size))
    x = adata.obsm[embed_key]
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        yield start, end, transform_embeddings(x[start:end], metric)


def load_cell_line_tissue_map(path: Optional[str | Path], *, logger: logging.Logger) -> Dict[str, str]:
    if not path:
        return {}
    path = Path(path)
    if not path.exists():
        logger.warning("Cell-line metadata not found: %s", path)
        return {}
    df = pd.read_csv(path)
    required = {"cell_name", "Organ"}
    missing = required - set(df.columns)
    if missing:
        logger.warning("Cell-line metadata %s missing columns %s; tissue annotations disabled.", path, sorted(missing))
        return {}

    out: Dict[str, str] = {}
    for cell_name, sub in df.groupby("cell_name", sort=False):
        organs = [
            str(x)
            for x in sub["Organ"].dropna().astype(str).str.strip().unique().tolist()
            if str(x).strip() and str(x).strip().lower() not in {"nan", "none", "null"}
        ]
        out[str(cell_name)] = organs[0] if organs else "unknown"
        if len(set(organs)) > 1:
            logger.warning("Multiple Organ values for cell_name=%s; using %s.", cell_name, out[str(cell_name)])
    logger.info("Loaded tissue annotations for %d reference cell lines from %s.", len(out), path)
    return out


def make_reference_metadata(
    obs: pd.DataFrame,
    *,
    cell_col: str,
    perturbation_col: str,
    state_col: Optional[str],
    tissue_map: Dict[str, str],
    max_reference_cells: Optional[int],
) -> pd.DataFrame:
    n = len(obs) if max_reference_cells is None else min(int(max_reference_cells), len(obs))
    obs = obs.iloc[:n].copy()
    if cell_col not in obs.columns:
        raise KeyError(f"Reference cell column {cell_col!r} not found in adata.obs.")

    out = pd.DataFrame({"reference_index": np.arange(n, dtype=np.int64)})
    out["cell_name"] = obs[cell_col].astype(str).to_numpy()

    if state_col:
        if state_col not in obs.columns:
            raise KeyError(f"Reference state column {state_col!r} not found in adata.obs.")
        out["reference_state"] = obs[state_col].astype(str).to_numpy()
        if perturbation_col in obs.columns:
            out["drugname_drugconc"] = obs[perturbation_col].astype(str).to_numpy()
        else:
            out["drugname_drugconc"] = "unknown"
    else:
        if perturbation_col not in obs.columns:
            raise KeyError(
                f"Reference perturbation column {perturbation_col!r} not found in adata.obs. "
                "Pass --reference-state-col if the reference state is already encoded in one column."
            )
        out["drugname_drugconc"] = obs[perturbation_col].astype(str).to_numpy()
        out["reference_state"] = out["cell_name"].astype(str) + " | " + out["drugname_drugconc"].astype(str)

    out["reference_tissue"] = out["cell_name"].map(tissue_map).fillna("unknown").astype(str)
    return out


def summarize_reference_states(reference_cells: pd.DataFrame) -> pd.DataFrame:
    return (
        reference_cells.groupby(["reference_state", "cell_name", "drugname_drugconc", "reference_tissue"], as_index=False)
        .agg(n_reference_cells=("reference_index", "size"))
        .sort_values(["cell_name", "drugname_drugconc", "reference_state"])
        .reset_index(drop=True)
    )


def normalized_entropy(labels: Sequence[Any]) -> float:
    if len(labels) == 0:
        return math.nan
    values, counts = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    if len(values) <= 1:
        return 0.0
    probs = counts.astype(float) / float(counts.sum())
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(float(len(values)))


def _neighbor_entropy_per_row(labels_2d: np.ndarray) -> np.ndarray:
    return np.asarray([normalized_entropy(row) for row in labels_2d], dtype=np.float32)


def compute_reference_self_scores(
    *,
    embeddings: np.ndarray,
    index: Any,
    reference_cells: pd.DataFrame,
    metric: str,
    k: int,
    search_batch_size: int,
    logger: logging.Logger,
) -> Dict[str, np.ndarray]:
    n = int(embeddings.shape[0])
    k = int(k)
    search_k = min(k + 1, n)
    local_mean = np.full(n, np.nan, dtype=np.float32)
    local_kth = np.full(n, np.nan, dtype=np.float32)

    logger.info("Computing reference self kNN distances with self-neighbor removal.")
    for start in range(0, n, int(search_batch_size)):
        end = min(start + int(search_batch_size), n)
        dist, neigh = search_index(index, embeddings[start:end], k=search_k, metric=metric)
        for row_i, ref_i in enumerate(range(start, end)):
            mask = neigh[row_i] != ref_i
            d = dist[row_i][mask][:k]
            if len(d) == 0:
                continue
            local_mean[ref_i] = float(np.mean(d))
            local_kth[ref_i] = float(d[-1])
        if start == 0 or end == n or end % max(int(search_batch_size) * 10, 1) == 0:
            logger.info("  reference self-distance progress: %d/%d", end, n)

    local_mean = np.where(np.isfinite(local_mean), local_mean, np.nanmedian(local_mean)).astype(np.float32)
    local_kth = np.where(np.isfinite(local_kth), local_kth, np.nanmedian(local_kth)).astype(np.float32)

    local_ratio = np.full(n, np.nan, dtype=np.float32)
    state_entropy = np.full(n, np.nan, dtype=np.float32)
    tissue_entropy = np.full(n, np.nan, dtype=np.float32)
    state_values = reference_cells["reference_state"].astype(str).to_numpy()
    tissue_values = reference_cells["reference_tissue"].astype(str).to_numpy()

    logger.info("Computing reference local density ratios and neighbor entropies.")
    for start in range(0, n, int(search_batch_size)):
        end = min(start + int(search_batch_size), n)
        dist, neigh = search_index(index, embeddings[start:end], k=search_k, metric=metric)
        clean_neigh = np.full((end - start, k), -1, dtype=np.int64)
        clean_dist = np.full((end - start, k), np.nan, dtype=np.float32)
        for row_i, ref_i in enumerate(range(start, end)):
            mask = neigh[row_i] != ref_i
            ids = neigh[row_i][mask][:k]
            ds = dist[row_i][mask][:k]
            if len(ids) == 0:
                continue
            clean_neigh[row_i, : len(ids)] = ids
            clean_dist[row_i, : len(ds)] = ds
        valid = clean_neigh >= 0
        row_mean = np.nanmean(clean_dist, axis=1)
        denom = np.asarray(
            [
                float(np.mean(local_mean[row[mask]])) if np.any(mask) else math.nan
                for row, mask in zip(clean_neigh, valid)
            ],
            dtype=np.float32,
        )
        local_ratio[start:end] = row_mean / np.maximum(denom, 1e-12)
        state_entropy[start:end] = _neighbor_entropy_per_row(
            np.asarray(
                [state_values[row[mask]] if np.any(mask) else np.asarray([], dtype=object) for row, mask in zip(clean_neigh, valid)],
                dtype=object,
            )
        )
        tissue_entropy[start:end] = _neighbor_entropy_per_row(
            np.asarray(
                [tissue_values[row[mask]] if np.any(mask) else np.asarray([], dtype=object) for row, mask in zip(clean_neigh, valid)],
                dtype=object,
            )
        )
        if start == 0 or end == n or end % max(int(search_batch_size) * 10, 1) == 0:
            logger.info("  reference support-score progress: %d/%d", end, n)

    return {
        "knn_mean_distance": local_mean,
        "knn_kth_distance": local_kth,
        "local_density_ratio": local_ratio,
        "neighbor_state_entropy_norm": state_entropy,
        "neighbor_tissue_entropy_norm": tissue_entropy,
    }


def sample_indices_per_group(
    labels: Sequence[Any],
    *,
    n_per_group: int,
    rng: np.random.Generator,
    groups: Optional[Sequence[Any]] = None,
) -> np.ndarray:
    labels_arr = np.asarray(labels, dtype=object)
    group_values = np.asarray(list(groups), dtype=object) if groups is not None else pd.unique(labels_arr)
    chosen: List[np.ndarray] = []
    n_per_group = max(1, int(n_per_group))
    for group in group_values:
        idx = np.flatnonzero(labels_arr == group)
        if len(idx) == 0:
            continue
        if len(idx) > n_per_group:
            idx = rng.choice(idx, size=n_per_group, replace=False)
        chosen.append(np.asarray(idx, dtype=np.int64))
    if not chosen:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(chosen).astype(np.int64))


def summarize_state_scores(
    cell_scores: pd.DataFrame,
    *,
    state_col: str,
    calibration_name: str,
) -> pd.DataFrame:
    if cell_scores.empty:
        return pd.DataFrame()
    d = cell_scores.copy()
    required = [
        "knn_mean_distance",
        "knn_kth_distance",
        "local_density_ratio",
        "neighbor_state_entropy_norm",
        "neighbor_tissue_entropy_norm",
    ]
    agg_spec = {
        "n_cells_scored": (PRIMARY_SCORE, "size"),
        "median_knn_mean_distance": ("knn_mean_distance", "median"),
        "p95_knn_mean_distance": ("knn_mean_distance", lambda x: quantile(x, 0.95)),
        "median_knn_kth_distance": ("knn_kth_distance", "median"),
        "median_local_density_ratio": ("local_density_ratio", "median"),
        "p95_local_density_ratio": ("local_density_ratio", lambda x: quantile(x, 0.95)),
        "median_neighbor_state_entropy_norm": ("neighbor_state_entropy_norm", "median"),
        "median_neighbor_tissue_entropy_norm": ("neighbor_tissue_entropy_norm", "median"),
    }
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Cell score table missing required columns: {missing}")
    out = d.groupby(state_col, as_index=False).agg(**agg_spec)
    out = out.rename(columns={state_col: "state"})
    out["calibration"] = calibration_name
    return out


def build_seen_calibration(
    *,
    reference_cells: pd.DataFrame,
    self_scores: Dict[str, np.ndarray],
    cells_per_state: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    sampled = sample_indices_per_group(
        reference_cells["reference_state"].astype(str).to_numpy(),
        n_per_group=int(cells_per_state),
        rng=rng,
    )
    cell_df = pd.DataFrame(
        {
            "reference_index": sampled,
            "state": reference_cells["reference_state"].astype(str).to_numpy()[sampled],
            "knn_mean_distance": self_scores["knn_mean_distance"][sampled],
            "knn_kth_distance": self_scores["knn_kth_distance"][sampled],
            "local_density_ratio": self_scores["local_density_ratio"][sampled],
            "neighbor_state_entropy_norm": self_scores["neighbor_state_entropy_norm"][sampled],
            "neighbor_tissue_entropy_norm": self_scores["neighbor_tissue_entropy_norm"][sampled],
        }
    )
    state_df = summarize_state_scores(cell_df, state_col="state", calibration_name="seen_reference_state")
    cell_distributions = {
        "knn_mean_distance": finite_array(self_scores["knn_mean_distance"]),
        "knn_kth_distance": finite_array(self_scores["knn_kth_distance"]),
        "local_density_ratio": finite_array(self_scores["local_density_ratio"]),
    }
    return state_df, cell_distributions


def score_embedding_batch(
    *,
    query_embeddings: np.ndarray,
    index: Any,
    metric: str,
    k: int,
    reference_cells: pd.DataFrame,
    reference_local_mean: np.ndarray,
    index_to_reference: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    dist, neigh = search_index(index, query_embeddings, k=int(k), metric=metric)
    if index_to_reference is not None:
        neigh_global = index_to_reference[neigh]
    else:
        neigh_global = neigh

    ref_states = reference_cells["reference_state"].astype(str).to_numpy()
    ref_cell_names = reference_cells["cell_name"].astype(str).to_numpy()
    ref_tissues = reference_cells["reference_tissue"].astype(str).to_numpy()

    neighbor_local = reference_local_mean[neigh_global]
    knn_mean = np.mean(dist, axis=1).astype(np.float32)
    knn_kth = dist[:, -1].astype(np.float32)
    local_ratio = knn_mean / np.maximum(np.mean(neighbor_local, axis=1), 1e-12)
    neighbor_states = ref_states[neigh_global]
    neighbor_cell_names = ref_cell_names[neigh_global]
    neighbor_tissues = ref_tissues[neigh_global]

    return {
        "distances": dist,
        "neighbors": neigh_global,
        "knn_mean_distance": knn_mean,
        "knn_kth_distance": knn_kth,
        "local_density_ratio": local_ratio.astype(np.float32),
        "neighbor_state_entropy_norm": _neighbor_entropy_per_row(neighbor_states),
        "neighbor_tissue_entropy_norm": _neighbor_entropy_per_row(neighbor_tissues),
        "nearest_reference_index": neigh_global[:, 0].astype(np.int64),
        "nearest_reference_state": neighbor_states[:, 0],
        "nearest_reference_cell_name": neighbor_cell_names[:, 0],
        "nearest_reference_tissue": neighbor_tissues[:, 0],
        "neighbor_states": neighbor_states,
        "neighbor_tissues": neighbor_tissues,
    }


def score_reference_holdout_cells(
    *,
    embeddings: np.ndarray,
    heldout_indices: np.ndarray,
    train_indices: np.ndarray,
    reference_cells: pd.DataFrame,
    reference_local_mean: np.ndarray,
    metric: str,
    k: int,
    add_batch_size: int,
    search_batch_size: int,
    gpu_id: int,
    require_gpu: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if len(heldout_indices) == 0 or len(train_indices) == 0:
        return pd.DataFrame()
    index, resources = create_faiss_index(
        embeddings=embeddings,
        metric=metric,
        add_batch_size=add_batch_size,
        gpu_id=gpu_id,
        require_gpu=require_gpu,
        logger=logger,
        subset_indices=train_indices,
    )

    rows: List[pd.DataFrame] = []
    for start in range(0, len(heldout_indices), int(search_batch_size)):
        ids = heldout_indices[start : start + int(search_batch_size)]
        out = score_embedding_batch(
            query_embeddings=np.ascontiguousarray(embeddings[ids], dtype=np.float32),
            index=index,
            metric=metric,
            k=k,
            reference_cells=reference_cells,
            reference_local_mean=reference_local_mean,
            index_to_reference=train_indices,
        )
        rows.append(
            pd.DataFrame(
                {
                    "reference_index": ids,
                    "state": reference_cells["reference_state"].astype(str).to_numpy()[ids],
                    "cell_name": reference_cells["cell_name"].astype(str).to_numpy()[ids],
                    "knn_mean_distance": out["knn_mean_distance"],
                    "knn_kth_distance": out["knn_kth_distance"],
                    "local_density_ratio": out["local_density_ratio"],
                    "neighbor_state_entropy_norm": out["neighbor_state_entropy_norm"],
                    "neighbor_tissue_entropy_norm": out["neighbor_tissue_entropy_norm"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def heldout_state_calibration(
    *,
    embeddings: np.ndarray,
    reference_cells: pd.DataFrame,
    reference_local_mean: np.ndarray,
    metric: str,
    k: int,
    cells_per_state: int,
    n_splits: int,
    state_fraction: float,
    max_states_per_split: int,
    add_batch_size: int,
    search_batch_size: int,
    gpu_id: int,
    require_gpu: bool,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> pd.DataFrame:
    state_values = reference_cells["reference_state"].astype(str).to_numpy()
    unique_states = np.asarray(pd.unique(state_values), dtype=object)
    n_holdout = max(1, int(round(len(unique_states) * float(state_fraction))))
    n_holdout = min(n_holdout, int(max_states_per_split), len(unique_states))
    all_indices = np.arange(len(reference_cells), dtype=np.int64)
    rows: List[pd.DataFrame] = []

    for split in range(int(n_splits)):
        holdout_states = rng.choice(unique_states, size=n_holdout, replace=False)
        heldout_mask = np.isin(state_values, holdout_states)
        heldout_candidate = np.flatnonzero(heldout_mask)
        sampled = sample_indices_per_group(
            state_values[heldout_candidate],
            n_per_group=int(cells_per_state),
            rng=rng,
        )
        heldout_indices = heldout_candidate[sampled]
        train_indices = all_indices[~heldout_mask]
        logger.info(
            "Heldout-state calibration split %d/%d: %d heldout states, %d heldout cells, %d train cells.",
            split + 1,
            int(n_splits),
            len(holdout_states),
            len(heldout_indices),
            len(train_indices),
        )
        cell_df = score_reference_holdout_cells(
            embeddings=embeddings,
            heldout_indices=heldout_indices,
            train_indices=train_indices,
            reference_cells=reference_cells,
            reference_local_mean=reference_local_mean,
            metric=metric,
            k=k,
            add_batch_size=add_batch_size,
            search_batch_size=search_batch_size,
            gpu_id=gpu_id,
            require_gpu=require_gpu,
            logger=logger,
        )
        state_df = summarize_state_scores(cell_df, state_col="state", calibration_name="heldout_reference_state")
        state_df["split"] = split
        rows.append(state_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def heldout_cell_name_calibration(
    *,
    embeddings: np.ndarray,
    reference_cells: pd.DataFrame,
    reference_local_mean: np.ndarray,
    metric: str,
    k: int,
    cells_per_state: int,
    n_splits: int,
    cell_names_per_split: int,
    max_states_per_split: int,
    add_batch_size: int,
    search_batch_size: int,
    gpu_id: int,
    require_gpu: bool,
    rng: np.random.Generator,
    logger: logging.Logger,
) -> pd.DataFrame:
    state_values = reference_cells["reference_state"].astype(str).to_numpy()
    cell_values = reference_cells["cell_name"].astype(str).to_numpy()
    unique_cells = np.asarray(pd.unique(cell_values), dtype=object)
    n_holdout_cells = min(max(1, int(cell_names_per_split)), len(unique_cells))
    all_indices = np.arange(len(reference_cells), dtype=np.int64)
    rows: List[pd.DataFrame] = []

    for split in range(int(n_splits)):
        holdout_cells = rng.choice(unique_cells, size=n_holdout_cells, replace=False)
        heldout_cell_mask = np.isin(cell_values, holdout_cells)
        candidate_states = np.asarray(pd.unique(state_values[heldout_cell_mask]), dtype=object)
        if len(candidate_states) > int(max_states_per_split):
            candidate_states = rng.choice(candidate_states, size=int(max_states_per_split), replace=False)
        heldout_mask = heldout_cell_mask & np.isin(state_values, candidate_states)
        exclude_train_mask = heldout_cell_mask
        heldout_candidate = np.flatnonzero(heldout_mask)
        sampled = sample_indices_per_group(
            state_values[heldout_candidate],
            n_per_group=int(cells_per_state),
            rng=rng,
        )
        heldout_indices = heldout_candidate[sampled]
        train_indices = all_indices[~exclude_train_mask]
        logger.info(
            "Heldout-cell-line calibration split %d/%d: %d heldout cell lines, %d states scored, %d heldout cells, %d train cells.",
            split + 1,
            int(n_splits),
            len(holdout_cells),
            len(candidate_states),
            len(heldout_indices),
            len(train_indices),
        )
        cell_df = score_reference_holdout_cells(
            embeddings=embeddings,
            heldout_indices=heldout_indices,
            train_indices=train_indices,
            reference_cells=reference_cells,
            reference_local_mean=reference_local_mean,
            metric=metric,
            k=k,
            add_batch_size=add_batch_size,
            search_batch_size=search_batch_size,
            gpu_id=gpu_id,
            require_gpu=require_gpu,
            logger=logger,
        )
        state_df = summarize_state_scores(cell_df, state_col="state", calibration_name="heldout_cell_name")
        state_df["split"] = split
        rows.append(state_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_calibration_thresholds(
    *,
    cell_distributions: Dict[str, np.ndarray],
    state_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cell_rows = []
    for metric_name, values in cell_distributions.items():
        values = finite_array(values)
        cell_rows.append(
            {
                "calibration": "seen_reference_cell",
                "score": metric_name,
                "n": int(len(values)),
                "p50": quantile(values, 0.50),
                "p90": quantile(values, 0.90),
                "p95": quantile(values, 0.95),
                "p99": quantile(values, 0.99),
            }
        )

    state_rows = []
    if not state_scores.empty:
        for cal_name, sub in state_scores.groupby("calibration", sort=False):
            for col in [
                "median_knn_mean_distance",
                "p95_knn_mean_distance",
                "median_local_density_ratio",
                "p95_local_density_ratio",
            ]:
                if col not in sub.columns:
                    continue
                values = finite_array(sub[col])
                state_rows.append(
                    {
                        "calibration": cal_name,
                        "score": col,
                        "n": int(len(values)),
                        "p50": quantile(values, 0.50),
                        "p90": quantile(values, 0.90),
                        "p95": quantile(values, 0.95),
                        "p99": quantile(values, 0.99),
                    }
                )
    return pd.DataFrame(cell_rows), pd.DataFrame(state_rows)


def build_reference_manifold(
    *,
    reference_h5ad: str | Path,
    output_dir: str | Path,
    reference_cell_col: str = "cell_name",
    reference_perturbation_col: str = "drugname_drugconc",
    reference_state_col: Optional[str] = None,
    embed_key: str = "X_state",
    metric: str = "l2",
    k: int = 50,
    seed: int = 42,
    add_batch_size: int = 100_000,
    search_batch_size: int = 16_384,
    calibration_cells_per_state: int = 100,
    calibration_splits: int = 3,
    calibration_state_fraction: float = 0.10,
    calibration_max_states_per_split: int = 250,
    calibration_cell_names_per_split: int = 5,
    calibration_max_cell_name_states_per_split: int = 250,
    skip_heldout_state_calibration: bool = False,
    skip_heldout_cell_name_calibration: bool = False,
    cell_line_metadata: Optional[str | Path] = None,
    gpu_id: int = 0,
    require_gpu: bool = True,
    save_faiss_index: bool = False,
    max_reference_cells: Optional[int] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    arrays_dir = output_dir / "arrays"
    calibration_dir = output_dir / "calibration"
    for path in [tables_dir, arrays_dir, calibration_dir]:
        path.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    rng = np.random.default_rng(int(seed))

    params = ReferenceBuildParams(
        reference_h5ad=str(reference_h5ad),
        output_dir=str(output_dir),
        reference_cell_col=reference_cell_col,
        reference_perturbation_col=reference_perturbation_col,
        reference_state_col=reference_state_col,
        embed_key=embed_key,
        metric=metric,
        k=int(k),
        seed=int(seed),
        add_batch_size=int(add_batch_size),
        search_batch_size=int(search_batch_size),
        calibration_cells_per_state=int(calibration_cells_per_state),
        calibration_splits=int(calibration_splits),
        calibration_state_fraction=float(calibration_state_fraction),
        calibration_max_states_per_split=int(calibration_max_states_per_split),
        calibration_cell_names_per_split=int(calibration_cell_names_per_split),
        calibration_max_cell_name_states_per_split=int(calibration_max_cell_name_states_per_split),
        skip_heldout_state_calibration=bool(skip_heldout_state_calibration),
        skip_heldout_cell_name_calibration=bool(skip_heldout_cell_name_calibration),
        cell_line_metadata=str(cell_line_metadata) if cell_line_metadata else None,
        gpu_id=int(gpu_id),
        require_gpu=bool(require_gpu),
        save_faiss_index=bool(save_faiss_index),
        max_reference_cells=max_reference_cells,
    )

    logger.info("=== Build ST-SE embedding manifold reference ===")
    logger.info("Reference h5ad: %s", reference_h5ad)
    logger.info("Output directory: %s", output_dir)
    logger.info("Embedding key: %s", embed_key)
    logger.info("Metric: %s", metric)
    logger.info("k: %d", int(k))

    faiss = import_faiss()
    log_gpu_status(faiss, gpu_id=int(gpu_id), require_gpu=bool(require_gpu), logger=logger)

    adata = load_h5ad(reference_h5ad, logger=logger)
    n_obs, dim = obsm_shape(adata, embed_key)
    n_reference = n_obs if max_reference_cells is None else min(int(max_reference_cells), n_obs)
    logger.info("Reference cells available: %d; using: %d; embedding dim: %d", n_obs, n_reference, dim)

    tissue_map = load_cell_line_tissue_map(cell_line_metadata, logger=logger)
    reference_cells = make_reference_metadata(
        adata.obs,
        cell_col=reference_cell_col,
        perturbation_col=reference_perturbation_col,
        state_col=reference_state_col,
        tissue_map=tissue_map,
        max_reference_cells=n_reference,
    )
    reference_states = summarize_reference_states(reference_cells)
    logger.info("Reference states: %d", len(reference_states))
    logger.info("Reference cell lines: %d", reference_cells["cell_name"].nunique())
    logger.info("Reference tissues: %d", reference_cells["reference_tissue"].nunique())

    embeddings_path = arrays_dir / "reference_embeddings.float32.npy"
    embeddings = np.lib.format.open_memmap(embeddings_path, mode="w+", dtype=np.float32, shape=(n_reference, dim))
    logger.info("Writing transformed reference embeddings to %s", embeddings_path)
    for start, end, batch in iter_obsm_contiguous(
        adata,
        embed_key,
        metric=metric,
        batch_size=int(add_batch_size),
        n_rows=n_reference,
    ):
        embeddings[start:end] = batch
        if start == 0 or end == n_reference or end % max(int(add_batch_size) * 5, 1) == 0:
            logger.info("  embedding write progress: %d/%d", end, n_reference)
    embeddings.flush()

    reference_cells_path = write_table(reference_cells, tables_dir / "reference_cells.parquet", logger=logger)
    reference_states_path = write_table(reference_states, tables_dir / "reference_states.parquet", logger=logger)

    index, resources = create_faiss_index(
        embeddings=embeddings,
        metric=metric,
        add_batch_size=int(add_batch_size),
        gpu_id=int(gpu_id),
        require_gpu=bool(require_gpu),
        logger=logger,
    )

    saved_index_path = None
    if save_faiss_index:
        logger.info("Saving CPU copy of FAISS index. This can be large.")
        cpu_index = faiss.index_gpu_to_cpu(index) if int(faiss.get_num_gpus()) > int(gpu_id) else index
        saved_index_path = output_dir / "reference.faiss"
        faiss.write_index(cpu_index, str(saved_index_path))
        logger.info("Saved FAISS index: %s", saved_index_path)

    self_scores = compute_reference_self_scores(
        embeddings=embeddings,
        index=index,
        reference_cells=reference_cells,
        metric=metric,
        k=int(k),
        search_batch_size=int(search_batch_size),
        logger=logger,
    )
    score_paths: Dict[str, str] = {}
    for name, arr in self_scores.items():
        path = arrays_dir / f"reference_{name}.float32.npy"
        np.save(path, arr.astype(np.float32))
        score_paths[name] = str(path)

    seen_state_scores, seen_cell_distributions = build_seen_calibration(
        reference_cells=reference_cells,
        self_scores=self_scores,
        cells_per_state=int(calibration_cells_per_state),
        rng=rng,
    )
    calibration_state_parts = [seen_state_scores]

    if not skip_heldout_state_calibration and int(calibration_splits) > 0:
        calibration_state_parts.append(
            heldout_state_calibration(
                embeddings=embeddings,
                reference_cells=reference_cells,
                reference_local_mean=self_scores["knn_mean_distance"],
                metric=metric,
                k=int(k),
                cells_per_state=int(calibration_cells_per_state),
                n_splits=int(calibration_splits),
                state_fraction=float(calibration_state_fraction),
                max_states_per_split=int(calibration_max_states_per_split),
                add_batch_size=int(add_batch_size),
                search_batch_size=int(search_batch_size),
                gpu_id=int(gpu_id),
                require_gpu=bool(require_gpu),
                rng=rng,
                logger=logger,
            )
        )

    if not skip_heldout_cell_name_calibration and int(calibration_splits) > 0:
        calibration_state_parts.append(
            heldout_cell_name_calibration(
                embeddings=embeddings,
                reference_cells=reference_cells,
                reference_local_mean=self_scores["knn_mean_distance"],
                metric=metric,
                k=int(k),
                cells_per_state=int(calibration_cells_per_state),
                n_splits=int(calibration_splits),
                cell_names_per_split=int(calibration_cell_names_per_split),
                max_states_per_split=int(calibration_max_cell_name_states_per_split),
                add_batch_size=int(add_batch_size),
                search_batch_size=int(search_batch_size),
                gpu_id=int(gpu_id),
                require_gpu=bool(require_gpu),
                rng=rng,
                logger=logger,
            )
        )

    calibration_state_scores = pd.concat(
        [x for x in calibration_state_parts if x is not None and not x.empty],
        ignore_index=True,
        sort=False,
    )
    cell_thresholds, state_thresholds = build_calibration_thresholds(
        cell_distributions=seen_cell_distributions,
        state_scores=calibration_state_scores,
    )

    calibration_state_scores_path = write_table(
        calibration_state_scores,
        calibration_dir / "calibration_state_scores.parquet",
        logger=logger,
    )
    cell_thresholds_path = write_table(cell_thresholds, calibration_dir / "cell_thresholds.parquet", logger=logger)
    state_thresholds_path = write_table(state_thresholds, calibration_dir / "state_thresholds.parquet", logger=logger)

    seen_cell_score_paths: Dict[str, str] = {}
    for score_name, values in seen_cell_distributions.items():
        path = calibration_dir / f"seen_cell_{score_name}.float32.npy"
        np.save(path, values.astype(np.float32))
        seen_cell_score_paths[score_name] = str(path)

    metadata = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "n_reference_cells": int(n_reference),
        "n_reference_states": int(len(reference_states)),
        "n_reference_cell_names": int(reference_cells["cell_name"].nunique()),
        "n_reference_tissues": int(reference_cells["reference_tissue"].nunique()),
        "embedding_dim": int(dim),
        "primary_score": PRIMARY_SCORE,
        "calibration_state_rows": int(len(calibration_state_scores)),
        "calibrations": sorted(calibration_state_scores["calibration"].dropna().unique().tolist())
        if not calibration_state_scores.empty
        else [],
    }
    paths = {
        "reference_embeddings": str(embeddings_path),
        "reference_cells": str(reference_cells_path),
        "reference_states": str(reference_states_path),
        "reference_faiss_index": str(saved_index_path) if saved_index_path else None,
        "reference_self_scores": score_paths,
        "seen_cell_score_distributions": seen_cell_score_paths,
        "calibration_state_scores": str(calibration_state_scores_path),
        "cell_thresholds": str(cell_thresholds_path),
        "state_thresholds": str(state_thresholds_path),
        "log": str(output_dir / "manifold_qc.log"),
    }
    manifest = {
        "kind": "embedding_manifold_reference",
        "version": 1,
        "config": asdict(params),
        "metadata": metadata,
        "paths": paths,
    }
    manifest_path = write_json(output_dir / "reference_manifest.json", manifest)
    logger.info("Reference manifest: %s", manifest_path)
    logger.info("=== Reference build complete ===")
    return {"output_dir": str(output_dir), "manifest": str(manifest_path), "paths": paths, "metadata": metadata}


def load_reference_bundle(reference_dir: str | Path) -> Dict[str, Any]:
    reference_dir = Path(reference_dir)
    manifest_path = reference_dir / "reference_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Reference manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = manifest.get("paths", {})
    embeddings = np.load(paths["reference_embeddings"], mmap_mode="r")
    reference_cells = read_table(paths["reference_cells"])
    calibration_state_scores = read_table(paths["calibration_state_scores"])
    cell_thresholds = read_table(paths["cell_thresholds"])
    state_thresholds = read_table(paths["state_thresholds"])
    reference_local_mean = np.load(paths["reference_self_scores"]["knn_mean_distance"], mmap_mode="r")
    seen_cell_distributions = {
        name: np.load(path, mmap_mode="r")
        for name, path in paths.get("seen_cell_score_distributions", {}).items()
    }
    return {
        "manifest": manifest,
        "embeddings": embeddings,
        "reference_cells": reference_cells,
        "reference_local_mean": reference_local_mean,
        "calibration_state_scores": calibration_state_scores,
        "cell_thresholds": cell_thresholds,
        "state_thresholds": state_thresholds,
        "seen_cell_distributions": seen_cell_distributions,
    }


def sample_query_cells(
    obs: pd.DataFrame,
    *,
    state_col: str,
    cells_per_state: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if state_col not in obs.columns:
        raise KeyError(f"Query state column {state_col!r} not found in adata.obs.")
    states = obs[state_col].astype(str)
    rows = []
    chosen_indices: List[np.ndarray] = []
    for state, sub in states.groupby(states, sort=True):
        idx = sub.index
        positions = obs.index.get_indexer(idx)
        positions = positions[positions >= 0]
        n_total = len(positions)
        if n_total == 0:
            continue
        if n_total > int(cells_per_state):
            selected = rng.choice(positions, size=int(cells_per_state), replace=False)
        else:
            selected = positions
        chosen_indices.append(np.asarray(selected, dtype=np.int64))
        rows.append(
            {
                "query_cell_type": str(state),
                "n_cells_total": int(n_total),
                "n_cells_scored": int(len(selected)),
                "sampled_all_available": bool(len(selected) == n_total),
            }
        )
    if not chosen_indices:
        raise ValueError("No query cells were selected for scoring.")
    selected_indices = np.concatenate(chosen_indices).astype(np.int64)
    selected_indices.sort()
    return selected_indices, pd.DataFrame(rows)


def aggregate_neighbor_votes(
    *,
    query_states: np.ndarray,
    neighbor_labels: np.ndarray,
    distances: np.ndarray,
    vote_kind: str,
) -> pd.DataFrame:
    rows = []
    eps = 1e-6
    for state in pd.unique(query_states):
        mask = query_states == state
        labels = neighbor_labels[mask].reshape(-1)
        d = distances[mask].reshape(-1)
        weights = 1.0 / np.maximum(d, eps)
        tmp = pd.DataFrame({"label": labels.astype(str), "weight": weights.astype(float)})
        summary = tmp.groupby("label", as_index=False)["weight"].sum()
        total = float(summary["weight"].sum())
        summary = summary.sort_values("weight", ascending=False).reset_index(drop=True)
        summary["rank"] = np.arange(1, len(summary) + 1)
        summary["weight_fraction"] = summary["weight"] / total if total > 0 else np.nan
        summary["query_cell_type"] = str(state)
        summary["vote_kind"] = vote_kind
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out[["query_cell_type", "vote_kind", "rank", "label", "weight", "weight_fraction"]]


def summarize_query_states(
    *,
    cell_scores: pd.DataFrame,
    query_state_counts: pd.DataFrame,
    state_votes: pd.DataFrame,
    tissue_votes: pd.DataFrame,
    reference_states: pd.DataFrame,
    calibration_state_scores: pd.DataFrame,
    seen_cell_primary_distribution: np.ndarray,
) -> pd.DataFrame:
    summary = summarize_state_scores(cell_scores, state_col="query_cell_type", calibration_name="query")
    summary = summary.rename(columns={"state": "query_cell_type"}).drop(columns=["calibration"], errors="ignore")
    summary = summary.merge(query_state_counts, on="query_cell_type", how="left", suffixes=("", "_count"))
    if "n_cells_scored_count" in summary.columns:
        summary["n_cells_scored"] = summary["n_cells_scored_count"].fillna(summary["n_cells_scored"]).astype(int)
        summary = summary.drop(columns=["n_cells_scored_count"], errors="ignore")

    seen_95 = quantile(seen_cell_primary_distribution, 0.95)
    seen_99 = quantile(seen_cell_primary_distribution, 0.99)
    frac = (
        cell_scores.assign(
            above_seen_95=cell_scores[PRIMARY_SCORE].astype(float) > seen_95,
            above_seen_99=cell_scores[PRIMARY_SCORE].astype(float) > seen_99,
        )
        .groupby("query_cell_type", as_index=False)
        .agg(
            frac_cells_above_seen_95=("above_seen_95", "mean"),
            frac_cells_above_seen_99=("above_seen_99", "mean"),
            median_cell_ood_percentile_vs_seen=("ood_percentile_vs_seen_cell", "median"),
            p95_cell_ood_percentile_vs_seen=("ood_percentile_vs_seen_cell", lambda x: quantile(x, 0.95)),
        )
    )
    summary = summary.merge(frac, on="query_cell_type", how="left")

    primary_state_col = "median_local_density_ratio"
    if not calibration_state_scores.empty:
        for cal_name, sub in calibration_state_scores.groupby("calibration", sort=False):
            values = finite_array(sub[primary_state_col]) if primary_state_col in sub.columns else np.array([])
            col = f"ood_percentile_vs_{cal_name}"
            summary[col] = empirical_percentile(summary[primary_state_col].to_numpy(), values)

    top_state_votes = state_votes[state_votes["rank"] == 1].copy()
    top_state_votes = top_state_votes.rename(
        columns={
            "label": "nearest_reference_state",
            "weight_fraction": "nearest_reference_state_weight_fraction",
        }
    )[["query_cell_type", "nearest_reference_state", "nearest_reference_state_weight_fraction"]]
    summary = summary.merge(top_state_votes, on="query_cell_type", how="left")

    top_tissue_votes = tissue_votes[tissue_votes["rank"] == 1].copy()
    top_tissue_votes = top_tissue_votes.rename(
        columns={
            "label": "top_reference_tissue_vote",
            "weight_fraction": "top_reference_tissue_vote_weight_fraction",
        }
    )[["query_cell_type", "top_reference_tissue_vote", "top_reference_tissue_vote_weight_fraction"]]
    summary = summary.merge(top_tissue_votes, on="query_cell_type", how="left")

    reference_state_lookup = reference_states[
        ["reference_state", "cell_name", "drugname_drugconc", "reference_tissue"]
    ].drop_duplicates()
    summary = summary.merge(
        reference_state_lookup.rename(
            columns={
                "reference_state": "nearest_reference_state",
                "cell_name": "nearest_reference_cell_name",
                "drugname_drugconc": "nearest_reference_drugname_drugconc",
                "reference_tissue": "nearest_reference_tissue",
            }
        ),
        on="nearest_reference_state",
        how="left",
    )

    summary["reference_support_call"] = summary.apply(call_reference_support, axis=1)
    summary["support_call_reason"] = summary.apply(support_call_reason, axis=1)
    sort_col = "ood_percentile_vs_heldout_cell_name"
    if sort_col not in summary.columns:
        sort_col = "ood_percentile_vs_heldout_reference_state"
    if sort_col not in summary.columns:
        sort_col = "median_local_density_ratio"
    return summary.sort_values([sort_col, "frac_cells_above_seen_99"], ascending=[False, False]).reset_index(drop=True)


def call_reference_support(row: pd.Series) -> str:
    frac99 = float(row.get("frac_cells_above_seen_99", 0.0) or 0.0)
    pct_cell = row.get("ood_percentile_vs_heldout_cell_name", np.nan)
    pct_state = row.get("ood_percentile_vs_heldout_reference_state", np.nan)
    pct_seen = row.get("ood_percentile_vs_seen_reference_state", np.nan)
    hard_pct = pct_cell if np.isfinite(pct_cell) else pct_state

    if np.isfinite(hard_pct) and hard_pct >= 99.0:
        return "OOD"
    if np.isfinite(pct_state) and pct_state >= 99.0 and frac99 >= 0.25:
        return "OOD"
    if frac99 >= 0.50:
        return "OOD"
    if (np.isfinite(pct_state) and pct_state >= 95.0) or (np.isfinite(pct_seen) and pct_seen >= 99.0) or frac99 >= 0.10:
        return "boundary"
    return "core"


def support_call_reason(row: pd.Series) -> str:
    parts = []
    for col in [
        "ood_percentile_vs_heldout_cell_name",
        "ood_percentile_vs_heldout_reference_state",
        "ood_percentile_vs_seen_reference_state",
    ]:
        if col in row and pd.notna(row[col]):
            parts.append(f"{col}={float(row[col]):.1f}")
    if "frac_cells_above_seen_99" in row and pd.notna(row["frac_cells_above_seen_99"]):
        parts.append(f"frac_cells_above_seen_99={float(row['frac_cells_above_seen_99']):.3f}")
    return "; ".join(parts)


def run_query_manifold_qc(
    *,
    reference_dir: str | Path,
    query_h5ad: str | Path,
    output_dir: str | Path,
    query_state_col: str = "cell_type",
    embed_key: str = "X_state",
    k: Optional[int] = None,
    query_cells_per_state: int = 100,
    seed: int = 42,
    add_batch_size: int = 100_000,
    search_batch_size: int = 16_384,
    gpu_id: int = 0,
    require_gpu: bool = True,
    allow_k_mismatch: bool = False,
    use_saved_index: bool = False,
    save_query_neighbors: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    rng = np.random.default_rng(int(seed))

    logger.info("=== Score query states against ST-SE embedding manifold ===")
    logger.info("Reference directory: %s", reference_dir)
    logger.info("Query h5ad: %s", query_h5ad)
    logger.info("Query state column: %s", query_state_col)
    logger.info("Embedding key: %s", embed_key)

    faiss = import_faiss()
    log_gpu_status(faiss, gpu_id=int(gpu_id), require_gpu=bool(require_gpu), logger=logger)

    bundle = load_reference_bundle(reference_dir)
    manifest = bundle["manifest"]
    ref_cfg = manifest["config"]
    metric = str(ref_cfg["metric"])
    ref_k = int(ref_cfg["k"])
    score_k = ref_k if k is None else int(k)
    if score_k != ref_k and not allow_k_mismatch:
        raise ValueError(
            f"Requested k={score_k}, but reference calibration was built with k={ref_k}. "
            "Use the same k, rebuild the reference, or pass --allow-k-mismatch."
        )
    logger.info("Reference metric: %s", metric)
    logger.info("Reference calibration k: %d; scoring k: %d", ref_k, score_k)

    embeddings = bundle["embeddings"]
    reference_cells = bundle["reference_cells"]
    reference_states = summarize_reference_states(reference_cells)
    reference_local_mean = np.asarray(bundle["reference_local_mean"], dtype=np.float32)
    calibration_state_scores = bundle["calibration_state_scores"]
    seen_cell_distributions = bundle["seen_cell_distributions"]
    seen_primary = finite_array(seen_cell_distributions.get(PRIMARY_SCORE, np.array([])))
    if len(seen_primary) == 0:
        raise ValueError("Reference bundle does not contain seen-cell primary score calibration.")

    index = None
    resources = None
    saved_index = manifest.get("paths", {}).get("reference_faiss_index")
    if use_saved_index and saved_index:
        logger.info("Loading saved FAISS index: %s", saved_index)
        cpu_index = faiss.read_index(str(saved_index))
        if int(faiss.get_num_gpus()) > int(gpu_id):
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, int(gpu_id), cpu_index)
        elif require_gpu:
            raise RuntimeError("Saved index loaded, but FAISS GPU is required and unavailable.")
        else:
            index = cpu_index
    else:
        index, resources = create_faiss_index(
            embeddings=embeddings,
            metric=metric,
            add_batch_size=int(add_batch_size),
            gpu_id=int(gpu_id),
            require_gpu=bool(require_gpu),
            logger=logger,
        )

    adata = load_h5ad(query_h5ad, logger=logger)
    n_query_obs, query_dim = obsm_shape(adata, embed_key)
    if int(query_dim) != int(embeddings.shape[1]):
        raise ValueError(f"Query embedding dim {query_dim} does not match reference dim {embeddings.shape[1]}.")
    selected_indices, query_state_counts = sample_query_cells(
        adata.obs,
        state_col=query_state_col,
        cells_per_state=int(query_cells_per_state),
        rng=rng,
    )
    logger.info("Query cells available: %d", n_query_obs)
    logger.info("Query states: %d", len(query_state_counts))
    logger.info("Query cells selected for scoring: %d", len(selected_indices))

    obs_names = adata.obs_names.to_numpy()
    query_states_all = adata.obs[query_state_col].astype(str).to_numpy()
    cell_rows: List[pd.DataFrame] = []
    state_vote_parts: List[pd.DataFrame] = []
    tissue_vote_parts: List[pd.DataFrame] = []
    neighbor_index_parts: List[np.ndarray] = []
    neighbor_distance_parts: List[np.ndarray] = []
    neighbor_query_position_parts: List[np.ndarray] = []

    for start in range(0, len(selected_indices), int(search_batch_size)):
        ids = selected_indices[start : start + int(search_batch_size)]
        xq = read_obsm_rows(adata, embed_key, ids, metric=metric)
        out = score_embedding_batch(
            query_embeddings=xq,
            index=index,
            metric=metric,
            k=score_k,
            reference_cells=reference_cells,
            reference_local_mean=reference_local_mean,
        )
        cell_df = pd.DataFrame(
            {
                "query_position": ids.astype(np.int64),
                "query_obs_name": obs_names[ids].astype(str),
                "query_cell_type": query_states_all[ids].astype(str),
                "knn_mean_distance": out["knn_mean_distance"],
                "knn_kth_distance": out["knn_kth_distance"],
                "local_density_ratio": out["local_density_ratio"],
                "ood_percentile_vs_seen_cell": empirical_percentile(out[PRIMARY_SCORE], seen_primary),
                "neighbor_state_entropy_norm": out["neighbor_state_entropy_norm"],
                "neighbor_tissue_entropy_norm": out["neighbor_tissue_entropy_norm"],
                "nearest_reference_index": out["nearest_reference_index"],
                "nearest_reference_state": out["nearest_reference_state"],
                "nearest_reference_cell_name": out["nearest_reference_cell_name"],
                "nearest_reference_tissue": out["nearest_reference_tissue"],
            }
        )
        cell_rows.append(cell_df)
        state_vote_parts.append(
            aggregate_neighbor_votes(
                query_states=query_states_all[ids].astype(str),
                neighbor_labels=out["neighbor_states"],
                distances=out["distances"],
                vote_kind="reference_state",
            )
        )
        tissue_vote_parts.append(
            aggregate_neighbor_votes(
                query_states=query_states_all[ids].astype(str),
                neighbor_labels=out["neighbor_tissues"],
                distances=out["distances"],
                vote_kind="reference_tissue",
            )
        )
        if save_query_neighbors:
            neighbor_query_position_parts.append(ids.astype(np.int64, copy=True))
            neighbor_index_parts.append(np.asarray(out["neighbors"], dtype=np.int64).copy())
            neighbor_distance_parts.append(np.asarray(out["distances"], dtype=np.float32).copy())
        logger.info("  query scoring progress: %d/%d", min(start + int(search_batch_size), len(selected_indices)), len(selected_indices))

    cell_scores = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    state_votes_raw = pd.concat(state_vote_parts, ignore_index=True) if state_vote_parts else pd.DataFrame()
    tissue_votes_raw = pd.concat(tissue_vote_parts, ignore_index=True) if tissue_vote_parts else pd.DataFrame()

    # Re-aggregate votes across batches because each batch produced its own ranks.
    def combine_votes(votes: pd.DataFrame) -> pd.DataFrame:
        if votes.empty:
            return votes
        d = votes.groupby(["query_cell_type", "vote_kind", "label"], as_index=False)["weight"].sum()
        d["total_weight"] = d.groupby(["query_cell_type", "vote_kind"])["weight"].transform("sum")
        d["weight_fraction"] = d["weight"] / d["total_weight"]
        d = d.sort_values(["query_cell_type", "vote_kind", "weight"], ascending=[True, True, False])
        d["rank"] = d.groupby(["query_cell_type", "vote_kind"]).cumcount() + 1
        return d[["query_cell_type", "vote_kind", "rank", "label", "weight", "weight_fraction"]]

    state_votes = combine_votes(state_votes_raw)
    tissue_votes = combine_votes(tissue_votes_raw)
    state_scores = summarize_query_states(
        cell_scores=cell_scores,
        query_state_counts=query_state_counts,
        state_votes=state_votes,
        tissue_votes=tissue_votes,
        reference_states=reference_states,
        calibration_state_scores=calibration_state_scores,
        seen_cell_primary_distribution=seen_primary,
    )

    cell_scores_path = write_table(cell_scores, tables_dir / "query_cell_scores.parquet", logger=logger)
    state_scores_path = write_table(state_scores, tables_dir / "query_state_scores.parquet", logger=logger)
    state_votes_path = write_table(state_votes, tables_dir / "nearest_reference_state_votes.parquet", logger=logger)
    tissue_votes_path = write_table(tissue_votes, tables_dir / "nearest_reference_tissue_votes.parquet", logger=logger)
    calibration_copy_path = write_table(calibration_state_scores, tables_dir / "calibration_state_scores.parquet", logger=logger)
    query_counts_path = write_table(query_state_counts, tables_dir / "query_state_counts.parquet", logger=logger)
    query_neighbors_path = None
    if save_query_neighbors:
        query_neighbors_path = tables_dir / "query_reference_neighbors.npz"
        if neighbor_index_parts:
            np.savez_compressed(
                query_neighbors_path,
                query_position=np.concatenate(neighbor_query_position_parts).astype(np.int64, copy=False),
                neighbor_indices=np.vstack(neighbor_index_parts).astype(np.int64, copy=False),
                neighbor_distances=np.vstack(neighbor_distance_parts).astype(np.float32, copy=False),
                k=np.asarray([score_k], dtype=np.int64),
            )
        else:
            np.savez_compressed(
                query_neighbors_path,
                query_position=np.empty(0, dtype=np.int64),
                neighbor_indices=np.empty((0, 0), dtype=np.int64),
                neighbor_distances=np.empty((0, 0), dtype=np.float32),
                k=np.asarray([score_k], dtype=np.int64),
            )
        logger.info("Saved query neighbor cache: %s", query_neighbors_path)

    call_counts = state_scores["reference_support_call"].value_counts().to_dict() if not state_scores.empty else {}
    metadata = {
        "elapsed_seconds": float(time.perf_counter() - t0),
        "n_query_cells_available": int(n_query_obs),
        "n_query_cells_scored": int(len(cell_scores)),
        "n_query_states": int(len(state_scores)),
        "reference_metric": metric,
        "reference_k": int(ref_k),
        "score_k": int(score_k),
        "primary_score": PRIMARY_SCORE,
        "support_call_counts": {str(k): int(v) for k, v in call_counts.items()},
    }
    paths = {
        "query_cell_scores": str(cell_scores_path),
        "query_state_scores": str(state_scores_path),
        "nearest_reference_state_votes": str(state_votes_path),
        "nearest_reference_tissue_votes": str(tissue_votes_path),
        "calibration_state_scores": str(calibration_copy_path),
        "query_state_counts": str(query_counts_path),
        "query_reference_neighbors": str(query_neighbors_path) if query_neighbors_path else None,
        "log": str(output_dir / "manifold_qc.log"),
    }
    config = asdict(
        QueryScoreParams(
            reference_dir=str(reference_dir),
            query_h5ad=str(query_h5ad),
            output_dir=str(output_dir),
            query_state_col=query_state_col,
            embed_key=embed_key,
            k=k,
            query_cells_per_state=int(query_cells_per_state),
            seed=int(seed),
            add_batch_size=int(add_batch_size),
            search_batch_size=int(search_batch_size),
            gpu_id=int(gpu_id),
            require_gpu=bool(require_gpu),
            allow_k_mismatch=bool(allow_k_mismatch),
            use_saved_index=bool(use_saved_index),
            save_query_neighbors=bool(save_query_neighbors),
        )
    )
    manifest_out = {
        "kind": "embedding_manifold_query_qc",
        "version": 1,
        "config": config,
        "reference_manifest": manifest,
        "metadata": metadata,
        "paths": paths,
    }
    manifest_path = write_json(output_dir / "manifold_qc_config.used.json", manifest_out)
    logger.info("Query state table: %s", state_scores_path)
    logger.info("QC manifest: %s", manifest_path)
    logger.info("Support call counts: %s", metadata["support_call_counts"])
    logger.info("=== Query manifold QC complete ===")
    return {"output_dir": str(output_dir), "manifest": str(manifest_path), "paths": paths, "metadata": metadata}
