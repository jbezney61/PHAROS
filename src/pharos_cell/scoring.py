#!/usr/bin/env python
"""
scoring.py

Distribution-aware scoring functions for STATE/ST-SE sequential drug search.

Inputs are cell-state embeddings:
    predicted_states: [B, N, D] or [N, D]
    target_state:     [M, D]

where:
    B = number of candidate perturbations / beam nodes
    N = number of predicted cells, usually 256
    M = number of target cells, often 256 now but can be larger later
    D = SE embedding dimension, e.g. 2058

Implemented scores
------------------
1. Energy distance (euclidean on L2-normalized embeddings)
   - Fast, GPU-friendly, batch-friendly; same distributional family as the
     geomloss energy loss used in ST-SE training.
   - Full distance: 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||  (lower is better).
   - For a fixed target Y, E||Y-Y'|| is constant and cached in DistributionScorer.

2. Sinkhorn optimal transport distance
   - More biologically faithful distributional matching.
   - Builds a full pairwise cell-cell cost matrix and solves soft optimal transport.
   - Good for reranking top candidates and final evaluation.

Recommended use in beam search
------------------------------
Use energy distance to score all candidates quickly, keep top M, then rerank
those top M using Sinkhorn OT.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import numpy as np
import torch

if TYPE_CHECKING:
    from .projections import LinearProjection


TensorLike = Union[np.ndarray, torch.Tensor]


def as_3d_tensor(x: TensorLike, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert [N, D] or [B, N, D] input to [B, N, D]."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype, non_blocking=True)
    else:
        t = torch.as_tensor(x, device=device, dtype=dtype)

    if t.ndim == 2:
        t = t.unsqueeze(0)
    if t.ndim != 3:
        raise ValueError(f"Expected [N, D] or [B, N, D], got shape {tuple(t.shape)}")
    return t


def l2_normalize_embeddings(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2-normalize embeddings along the embedding dimension."""
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)


def _mean_pairwise_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean cdist(x, y) over the last two dimensions. x: [B,N,D], y: [B,M,D] -> [B]."""
    return torch.cdist(x, y, p=2).mean(dim=(1, 2))


def _mean_self_distance(x: torch.Tensor) -> torch.Tensor:
    """Mean cdist(x, x) over cells. x: [B,N,D] -> [B]. Diagonal terms are zero."""
    return torch.cdist(x, x, p=2).mean(dim=(1, 2))


def compute_target_self_term(
    target_state: TensorLike,
    *,
    normalize: bool = True,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """
    Compute E||Y - Y'|| for a target cloud Y.

    Returns a scalar tensor [1] on device. Constant for a fixed target during search.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    Y = as_3d_tensor(target_state, device=device)
    if Y.shape[0] != 1:
        raise ValueError("compute_target_self_term expects one target: [M, D] or [1, M, D]")

    if normalize:
        Y = l2_normalize_embeddings(Y)

    return _mean_self_distance(Y)  # [1]


def energy_distance(
    predicted_states: TensorLike,
    target_state: TensorLike,
    *,
    normalize: bool = True,
    target_self_term: Optional[torch.Tensor] = None,
    device: Optional[str | torch.device] = None,
) -> torch.Tensor:
    """
    Batch energy distance between predicted and target cell-state distributions.

    Lower is better. Returns ~0 when predicted and target empirical measures match.

    Formula (euclidean metric on optionally L2-normalized embeddings):
        E(X, Y) = 2 * E||X - Y|| - E||X - X'|| - E||Y - Y'||

    Parameters
    ----------
    predicted_states:
        [N, D] or [B, N, D]
    target_state:
        [M, D], [1, M, D], or [B, M, D] matching batch size (pair screening).
    target_self_term:
        Optional precomputed E||Y - Y'|| as scalar [1]. When provided, skips
        recomputing cdist on the target (search with fixed target).
    """
    if device is None:
        if torch.is_tensor(predicted_states):
            device = predicted_states.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    X = as_3d_tensor(predicted_states, device=device)
    Y = as_3d_tensor(target_state, device=device)

    if Y.shape[0] == 1 and X.shape[0] > 1:
        Y = Y.expand(X.shape[0], -1, -1)
    elif Y.shape[0] != X.shape[0]:
        raise ValueError("target_state must be [M, D], [1, M, D], or [B, M, D] matching predicted batch size")

    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"Embedding dimensions differ: predicted D={X.shape[-1]}, target D={Y.shape[-1]}")

    if normalize:
        X = l2_normalize_embeddings(X)
        Y = l2_normalize_embeddings(Y)

    cross = _mean_pairwise_distance(X, Y)
    self_pred = _mean_self_distance(X)
    score = 2.0 * cross - self_pred

    if target_self_term is not None:
        score = score - target_self_term.to(device=device, dtype=score.dtype).reshape(-1)
    else:
        self_target = _mean_self_distance(Y)
        score = score - self_target

    return score


def pairwise_cost_matrix(
    X: torch.Tensor,
    Y: torch.Tensor,
    metric: Literal["cosine", "sqeuclidean", "euclidean"] = "cosine",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute pairwise cell-cell cost matrix C.

    X: [B, N, D]
    Y: [1, M, D] or [B, M, D]

    returns:
        C: [B, N, M]
    """
    if normalize:
        X = l2_normalize_embeddings(X)
        Y = l2_normalize_embeddings(Y)

    if metric == "cosine":
        sim = torch.bmm(X, Y.transpose(1, 2))
        return 1.0 - sim.clamp(-1.0, 1.0)

    if metric == "sqeuclidean":
        return torch.cdist(X, Y, p=2) ** 2

    if metric == "euclidean":
        return torch.cdist(X, Y, p=2)

    raise ValueError(f"Unsupported metric: {metric}")


def sinkhorn_ot_distance(
    predicted_states: TensorLike,
    target_state: TensorLike,
    metric: Literal["cosine", "sqeuclidean", "euclidean"] = "cosine",
    normalize: bool = True,
    epsilon: float = 0.05,
    n_iters: int = 100,
    device: Optional[str | torch.device] = None,
    return_transport: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch entropic Sinkhorn optimal transport distance.

    Lower is better.
    """
    if device is None:
        if torch.is_tensor(predicted_states):
            device = predicted_states.device
        else:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    X = as_3d_tensor(predicted_states, device=device)
    Y = as_3d_tensor(target_state, device=device)

    if Y.shape[0] == 1 and X.shape[0] > 1:
        Y = Y.expand(X.shape[0], -1, -1)
    elif Y.shape[0] != X.shape[0]:
        raise ValueError("target_state must be [M, D], [1, M, D], or [B, M, D] matching predicted batch size")

    if X.shape[-1] != Y.shape[-1]:
        raise ValueError(f"Embedding dimensions differ: predicted D={X.shape[-1]}, target D={Y.shape[-1]}")

    B, N, _ = X.shape
    M = Y.shape[1]

    C = pairwise_cost_matrix(X, Y, metric=metric, normalize=normalize)  # [B, N, M]

    log_a = torch.full((B, N), -np.log(N), dtype=X.dtype, device=device)
    log_b = torch.full((B, M), -np.log(M), dtype=X.dtype, device=device)

    log_K = -C / epsilon
    u = torch.zeros_like(log_a)
    v = torch.zeros_like(log_b)

    for _ in range(n_iters):
        u = log_a - torch.logsumexp(log_K + v[:, None, :], dim=2)
        v = log_b - torch.logsumexp(log_K + u[:, :, None], dim=1)

    log_P = log_K + u[:, :, None] + v[:, None, :]
    P = torch.exp(log_P)
    score = torch.sum(P * C, dim=(1, 2))

    if return_transport:
        return score, P
    return score


@dataclass
class DistributionScorer:
    """
    Convenience scorer for beam search.

    Stores:
      - target state on device
      - cached target self-term E||Y-Y'|| (constant for fixed target)
      - Sinkhorn hyperparameters
      - optional linear projection (PLS-DA / PCA) applied before scoring

    Use:
        scorer = DistributionScorer(target_embeddings, device="cuda:0")
        fast_scores = scorer.energy_distance(candidate_batch)
        final_scores = scorer.sinkhorn(top_candidate_batch)
    """

    target_state: TensorLike
    device: Optional[str | torch.device] = None
    normalize: bool = True
    sinkhorn_metric: Literal["cosine", "sqeuclidean", "euclidean"] = "cosine"
    sinkhorn_epsilon: float = 0.05
    sinkhorn_iters: int = 100
    projection: Optional["LinearProjection"] = None
    projection_auto_metric: bool = True
    projection_auto_epsilon: bool = False
    sinkhorn_epsilon_scale: float = 0.1

    _target_self_term: torch.Tensor = field(init=False, repr=False)
    _sinkhorn_epsilon_auto: Optional[float] = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device)

        if self.projection is not None:
            if self.normalize:
                warnings.warn(
                    "L2 normalization is disabled when a linear projection is active.",
                    stacklevel=2,
                )
            self.normalize = False
            if self.projection_auto_metric and self.sinkhorn_metric == "cosine":
                self.sinkhorn_metric = "sqeuclidean"

            self.target = self._transform_states(self.target_state)
            if self.projection_auto_epsilon:
                from .projections import estimate_sinkhorn_epsilon

                auto_eps = estimate_sinkhorn_epsilon(
                    self.target,
                    metric=self.sinkhorn_metric if self.sinkhorn_metric != "cosine" else "sqeuclidean",
                    scale=self.sinkhorn_epsilon_scale,
                    device=self.device,
                )
                self._sinkhorn_epsilon_auto = auto_eps
                self.sinkhorn_epsilon = auto_eps
                print(
                    f"Sinkhorn auto-epsilon (projection): {auto_eps:.6g} "
                    f"= {self.sinkhorn_epsilon_scale} × median target pairwise cost"
                )
        else:
            self.target = as_3d_tensor(self.target_state, device=self.device)

        if self.target.shape[0] != 1:
            raise ValueError("DistributionScorer expects one target distribution: [M, D] or [1, M, D]")

        self._target_self_term = compute_target_self_term(
            self.target,
            normalize=self.normalize,
            device=self.device,
        )

    def _transform_states(self, states: TensorLike) -> torch.Tensor:
        if self.projection is None:
            return as_3d_tensor(states, device=self.device)
        projected = self.projection.transform(states, device=self.device)
        return as_3d_tensor(projected, device=self.device)

    @property
    def sinkhorn_epsilon_auto(self) -> Optional[float]:
        return self._sinkhorn_epsilon_auto

    def projection_metadata(self) -> Optional[dict]:
        if self.projection is None:
            return None
        meta = self.projection.metadata_dict()
        meta["sinkhorn_epsilon"] = float(self.sinkhorn_epsilon)
        meta["sinkhorn_epsilon_auto"] = self._sinkhorn_epsilon_auto
        meta["sinkhorn_metric"] = self.sinkhorn_metric
        meta["normalize"] = self.normalize
        return meta

    @property
    def target_self_term(self) -> float:
        """Cached E||Y - Y'|| for the fixed target distribution."""
        return float(self._target_self_term.item())

    def energy_distance(self, predicted_states: TensorLike) -> torch.Tensor:
        """Full energy distance vs the fixed target (lower is better)."""
        X = self._transform_states(predicted_states)
        return energy_distance(
            predicted_states=X,
            target_state=self.target,
            normalize=self.normalize,
            target_self_term=self._target_self_term,
            device=self.device,
        )

    def sinkhorn(self, predicted_states: TensorLike, return_transport: bool = False, *, n_iters: Optional[int] = None):
        X = self._transform_states(predicted_states)
        return sinkhorn_ot_distance(
            predicted_states=X,
            target_state=self.target,
            metric=self.sinkhorn_metric,
            normalize=self.normalize,
            epsilon=self.sinkhorn_epsilon,
            n_iters=self.sinkhorn_iters if n_iters is None else n_iters,
            device=self.device,
            return_transport=return_transport,
        )

    def two_stage_rank(self, predicted_states: TensorLike, top_m: int = 50) -> dict:
        """
        Fast two-stage ranking:
          1. Score all candidates with energy distance.
          2. Rerank the best top_m candidates with Sinkhorn OT.
        """
        X = as_3d_tensor(predicted_states, device=self.device)
        B = X.shape[0]
        top_m = min(top_m, B)

        energy_scores = self.energy_distance(X)
        prelim_idx = torch.topk(-energy_scores, k=top_m).indices  # lower is better

        sink_scores = self.sinkhorn(X[prelim_idx])
        final_order = torch.argsort(sink_scores)

        final_idx = prelim_idx[final_order]
        final_sinkhorn = sink_scores[final_order]
        final_energy = energy_scores[final_idx]

        return {
            "final_indices": final_idx,
            "final_sinkhorn_scores": final_sinkhorn,
            "final_energy_scores": final_energy,
            "all_energy_scores": energy_scores,
            "prelim_indices": prelim_idx,
        }
