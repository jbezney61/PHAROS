#!/usr/bin/env python
"""
projections.py

Supervised linear dimensionality reduction for distribution scoring.

Fits a projection on start vs target cell-state embeddings (PLS-DA, PCA+PLS, or PCA)
and applies it only at scoring time. ST-SE conversion stays in the full embedding space.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

ProjectionMethod = Literal["pls_da", "pca_pls_da", "pca"]
TensorLike = Union[np.ndarray, torch.Tensor]


def _as_float32_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}")
    return X


def _subsample_rows(X: np.ndarray, cap: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if cap is None or X.shape[0] <= cap:
        return X
    idx = rng.choice(X.shape[0], size=int(cap), replace=False)
    return X[idx]


def _parse_component_grid(value: Union[str, Sequence[int]], *, name: str) -> List[int]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.replace(",", " ").split()]
    else:
        raw = [str(part).strip() for part in value]
    out: List[int] = []
    for part in raw:
        if not part:
            continue
        try:
            ivalue = int(part)
        except ValueError as exc:
            raise ValueError(f"{name} must contain integer component counts, got {part!r}") from exc
        if ivalue <= 0:
            raise ValueError(f"{name} component counts must be positive, got {ivalue}")
        if ivalue not in out:
            out.append(ivalue)
    if not out:
        raise ValueError(f"{name} must contain at least one component count")
    return sorted(out)


def _rotation_hash(rotation: np.ndarray) -> str:
    digest = hashlib.sha256(rotation.astype(np.float32, copy=False).tobytes()).hexdigest()
    return digest[:16]


@dataclass
class LinearProjection:
    """
    Linear map: Z = ((X - x_mean) * x_scale_inv) @ rotation, optionally / component_std.
    """

    method: ProjectionMethod
    n_components: int
    input_dim: int
    x_mean: np.ndarray
    rotation: np.ndarray
    component_std: np.ndarray
    x_scale: Optional[np.ndarray] = None
    fit_metadata: Dict[str, Any] = field(default_factory=dict)
    whiten_components: bool = False

    _rotation_t: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _x_mean_t: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _x_scale_inv_t: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _component_std_t: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.x_mean = np.asarray(self.x_mean, dtype=np.float32)
        self.rotation = np.asarray(self.rotation, dtype=np.float32)
        self.component_std = np.asarray(self.component_std, dtype=np.float32)
        if self.x_scale is not None:
            self.x_scale = np.asarray(self.x_scale, dtype=np.float32)

        if self.rotation.shape[0] != self.input_dim:
            raise ValueError(f"rotation rows {self.rotation.shape[0]} != input_dim {self.input_dim}")
        if self.rotation.shape[1] != self.n_components:
            raise ValueError(f"rotation cols {self.rotation.shape[1]} != n_components {self.n_components}")

    @property
    def output_dim(self) -> int:
        return int(self.n_components)

    @property
    def rotation_hash(self) -> str:
        return _rotation_hash(self.rotation)

    def summary(self) -> str:
        meta = self.fit_metadata or {}
        lines = [
            f"method={self.method}",
            f"n_components={self.n_components}",
            f"input_dim={self.input_dim}",
            f"whiten_components={self.whiten_components}",
            f"rotation_hash={self.rotation_hash}",
        ]
        if "var_recovered" in meta:
            lines.append(f"var_recovered={meta['var_recovered']:.4f}")
        if "var_recovered_bio" in meta:
            lines.append(f"var_recovered_bio={meta['var_recovered_bio']:.4f}")
        if "var_recovered_tail" in meta:
            lines.append(f"var_recovered_tail={meta['var_recovered_tail']:.4f}")
        return "\n".join(lines)

    def metadata_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_components": self.n_components,
            "input_dim": self.input_dim,
            "whiten_components": self.whiten_components,
            "rotation_hash": self.rotation_hash,
            "fit_metadata": self.fit_metadata,
        }

    @classmethod
    def fit(
        cls,
        start_X: np.ndarray,
        target_X: np.ndarray,
        *,
        method: ProjectionMethod = "pls_da",
        n_components: int = 32,
        fit_cap_per_class: Optional[int] = 4000,
        seed: int = 42,
        whiten_components: bool = False,
        pca_prefilter_components: Optional[int] = None,
    ) -> "LinearProjection":
        """Fit a linear projection from start and target embedding clouds."""
        start_X = _as_float32_2d(start_X)
        target_X = _as_float32_2d(target_X)
        if start_X.shape[1] != target_X.shape[1]:
            raise ValueError("start and target must share embedding dimension")

        rng = np.random.default_rng(seed)
        start_fit = _subsample_rows(start_X, fit_cap_per_class, rng)
        target_fit = _subsample_rows(target_X, fit_cap_per_class, rng)

        X = np.vstack([start_fit, target_fit])
        y = np.concatenate(
            [np.zeros(start_fit.shape[0], dtype=np.float32), np.ones(target_fit.shape[0], dtype=np.float32)]
        ).reshape(-1, 1)

        input_dim = X.shape[1]
        n_components = int(min(n_components, input_dim, X.shape[0] - 1))
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        x_mean = X.mean(axis=0).astype(np.float32)
        x_scale = X.std(axis=0, ddof=0).astype(np.float32)
        x_scale = np.where(x_scale < 1e-8, 1.0, x_scale).astype(np.float32)
        X_scaled = ((X - x_mean) / x_scale).astype(np.float32)

        pca_prefilter = int(pca_prefilter_components or min(256, input_dim))
        pca_prefilter = min(pca_prefilter, input_dim, X.shape[0] - 1)

        if method == "pca":
            model = PCA(n_components=n_components, random_state=seed)
            model.fit(X_scaled)
            rotation = model.components_.T.astype(np.float32)
            latent = model.transform(X_scaled)
        elif method == "pls_da":
            pls = PLSRegression(n_components=n_components, scale=False)
            pls.fit(X_scaled, y)
            rotation = pls.x_rotations_.astype(np.float32)
            latent = pls.transform(X_scaled)
        elif method == "pca_pls_da":
            pca = PCA(n_components=pca_prefilter, random_state=seed)
            X_pca = pca.fit_transform(X_scaled)
            pls = PLSRegression(n_components=min(n_components, X_pca.shape[1]), scale=False)
            pls.fit(X_pca, y)
            rotation = (pca.components_.T @ pls.x_rotations_).astype(np.float32)
            latent = pls.transform(X_pca)
            n_components = rotation.shape[1]
        else:
            raise ValueError(f"Unsupported projection method: {method}")

        component_std = np.std(latent, axis=0, ddof=0).astype(np.float32)
        if whiten_components:
            component_std = np.where(component_std < 1e-8, 1.0, component_std).astype(np.float32)
        else:
            component_std = np.ones_like(component_std, dtype=np.float32)

        fit_metadata: Dict[str, Any] = {
            "seed": seed,
            "fit_cap_per_class": fit_cap_per_class,
            "n_start_fit": int(start_fit.shape[0]),
            "n_target_fit": int(target_fit.shape[0]),
            "pca_prefilter_components": pca_prefilter if method == "pca_pls_da" else None,
        }

        proj = cls(
            method=method,
            n_components=int(n_components),
            input_dim=int(input_dim),
            x_mean=x_mean,
            x_scale=x_scale,
            rotation=rotation,
            component_std=component_std,
            fit_metadata=fit_metadata,
            whiten_components=whiten_components,
        )

        diag = diagnose_projection(start_X, target_X, proj, verbose=False)
        proj.fit_metadata.update(diag)
        return proj

    def _ensure_torch(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        if self._rotation_t is None or self._rotation_t.device != device:
            self._rotation_t = torch.as_tensor(self.rotation, device=device, dtype=dtype)
            self._x_mean_t = torch.as_tensor(self.x_mean, device=device, dtype=dtype)
            if self.x_scale is not None:
                inv = 1.0 / torch.as_tensor(self.x_scale, device=device, dtype=dtype)
                self._x_scale_inv_t = inv
            else:
                self._x_scale_inv_t = None
            if self.whiten_components:
                self._component_std_t = torch.as_tensor(self.component_std, device=device, dtype=dtype)
            else:
                self._component_std_t = None

    def transform_numpy(self, X: np.ndarray) -> np.ndarray:
        X = _as_float32_2d(X)
        centered = X - self.x_mean
        if self.x_scale is not None:
            centered = centered / self.x_scale
        Z = centered @ self.rotation
        if self.whiten_components:
            Z = Z / self.component_std
        return Z.astype(np.float32, copy=False)

    def transform(self, X: TensorLike, device: Optional[torch.device] = None) -> Union[np.ndarray, torch.Tensor]:
        """Transform [N, D] or [B, N, D] embeddings to projected space."""
        if isinstance(X, np.ndarray):
            if X.ndim == 2:
                return self.transform_numpy(X)
            if X.ndim == 3:
                return np.stack([self.transform_numpy(x) for x in X], axis=0)
            raise ValueError(f"Expected [N, D] or [B, N, D], got {X.shape}")

        if not torch.is_tensor(X):
            X = torch.as_tensor(X)

        orig_device = X.device
        if device is None:
            device = orig_device if X.is_cuda else torch.device("cpu")
        device = torch.device(device)

        squeeze_batch = False
        if X.ndim == 2:
            X = X.unsqueeze(0)
            squeeze_batch = True
        if X.ndim != 3:
            raise ValueError(f"Expected [N, D] or [B, N, D], got {tuple(X.shape)}")

        self._ensure_torch(device, dtype=torch.float32)
        flat = X.reshape(-1, X.shape[-1]).to(device=device, dtype=torch.float32)
        centered = flat - self._x_mean_t
        if self._x_scale_inv_t is not None:
            centered = centered * self._x_scale_inv_t
        Z = centered @ self._rotation_t
        if self._component_std_t is not None:
            Z = Z / self._component_std_t
        Z = Z.reshape(X.shape[0], X.shape[1], -1)
        if squeeze_batch:
            Z = Z.squeeze(0)
        if orig_device != device:
            Z = Z.to(orig_device)
        return Z

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "n_components": self.n_components,
            "input_dim": self.input_dim,
            "x_mean": self.x_mean,
            "rotation": self.rotation,
            "component_std": self.component_std,
            "whiten_components": np.array(self.whiten_components),
            "fit_metadata": json.dumps(self.fit_metadata),
        }
        if self.x_scale is not None:
            payload["x_scale"] = self.x_scale
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LinearProjection":
        z = np.load(path, allow_pickle=False)
        fit_metadata = json.loads(str(z["fit_metadata"].item())) if "fit_metadata" in z else {}
        x_scale = z["x_scale"].astype(np.float32) if "x_scale" in z else None
        whiten = bool(z["whiten_components"].item()) if "whiten_components" in z else False
        return cls(
            method=str(z["method"].item()),
            n_components=int(z["n_components"]),
            input_dim=int(z["input_dim"]),
            x_mean=z["x_mean"].astype(np.float32),
            rotation=z["rotation"].astype(np.float32),
            component_std=z["component_std"].astype(np.float32),
            x_scale=x_scale,
            fit_metadata=fit_metadata,
            whiten_components=whiten,
        )


def _var_recovered(delta: np.ndarray, rotation: np.ndarray, x_scale: Optional[np.ndarray]) -> float:
    if x_scale is not None:
        delta = delta / x_scale
    num = float(np.dot(delta @ rotation, delta @ rotation))
    den = float(np.dot(delta, delta))
    if den <= 0:
        return float("nan")
    return num / den


def diagnose_projection(
    start_X: np.ndarray,
    target_X: np.ndarray,
    projection: LinearProjection,
    *,
    bio_dim: int = 2048,
    verbose: bool = True,
) -> Dict[str, float]:
    """Diagnostics for how much start→target separation is preserved in projected space."""
    start_X = _as_float32_2d(start_X)
    target_X = _as_float32_2d(target_X)
    delta = target_X.mean(axis=0) - start_X.mean(axis=0)

    var_all = _var_recovered(delta, projection.rotation, projection.x_scale)
    out: Dict[str, float] = {
        "var_recovered": var_all,
        "delta_norm_full": float(np.linalg.norm(delta)),
    }

    if delta.shape[0] >= bio_dim:
        delta_bio = delta[:bio_dim]
        delta_tail = delta[bio_dim:]
        rot_bio = projection.rotation[:bio_dim, :]
        rot_tail = projection.rotation[bio_dim:, :]
        out["var_recovered_bio"] = _var_recovered(delta_bio, rot_bio, projection.x_scale[:bio_dim] if projection.x_scale is not None else None)
        out["var_recovered_tail"] = _var_recovered(delta_tail, rot_tail, projection.x_scale[bio_dim:] if projection.x_scale is not None else None)

    comp_std = projection.component_std
    out["component_std_min"] = float(comp_std.min())
    out["component_std_median"] = float(np.median(comp_std))
    out["component_std_max"] = float(comp_std.max())

    if verbose:
        print("\n=== Projection diagnostics ===")
        print(projection.summary())
        print(f"var_recovered (centroid start→target): {var_all:.4f}")
        if "var_recovered_bio" in out:
            print(f"  bio dims [:{bio_dim}]: {out['var_recovered_bio']:.4f}")
            print(f"  tail dims [{bio_dim}:]: {out['var_recovered_tail']:.4f}")
        print(
            f"component_std: min={out['component_std_min']:.4g}, "
            f"median={out['component_std_median']:.4g}, max={out['component_std_max']:.4g}"
        )

    return out


def _normalized_centroid_separation(start_Z: np.ndarray, target_Z: np.ndarray) -> float:
    start_Z = _as_float32_2d(start_Z)
    target_Z = _as_float32_2d(target_Z)
    start_mean = start_Z.mean(axis=0)
    target_mean = target_Z.mean(axis=0)
    between = float(np.linalg.norm(target_mean - start_mean))
    start_spread = float(np.mean(np.sum((start_Z - start_mean) ** 2, axis=1)))
    target_spread = float(np.mean(np.sum((target_Z - target_mean) ** 2, axis=1)))
    denom = float(np.sqrt(max(start_spread + target_spread, 1e-12)))
    return between / denom


def _component_complexity_key(row: Dict[str, Any]) -> Tuple[int, int]:
    pls = int(row.get("pls_components") or row.get("n_components") or 0)
    pca = int(row.get("pca_components") or 0)
    return pls, pca


def select_projection_components(
    start_X: np.ndarray,
    target_X: np.ndarray,
    *,
    method: str,
    pca_grid: Union[str, Sequence[int]],
    pls_grid: Union[str, Sequence[int]],
    fit_frac: float,
    repeats: int,
    small_cell_threshold: int,
    fallback_pca: int,
    fallback_pls: int,
    fit_cap: Optional[int],
    seed: int,
    whiten: bool = False,
    selection_rule: str = "one_se",
) -> Dict[str, Any]:
    """
    Select projection components from held-out start/target geometry only.

    The selected values are intended for a final refit on the full projection
    fit pool. Drug-pair outcomes are deliberately not used here.
    """
    start_X = _as_float32_2d(start_X)
    target_X = _as_float32_2d(target_X)
    if start_X.shape[1] != target_X.shape[1]:
        raise ValueError("start and target must share embedding dimension")
    if not (0.0 < float(fit_frac) < 1.0):
        raise ValueError("projection selection fit fraction must be between 0 and 1")
    if int(repeats) <= 0:
        raise ValueError("projection selection repeats must be positive")
    if int(small_cell_threshold) <= 1:
        raise ValueError("projection selection small-cell threshold must be > 1")
    selection_rule = str(selection_rule).replace("-", "_").casefold()
    if selection_rule not in {"one_se", "best"}:
        raise ValueError("projection selection rule must be 'one_se' or 'best'")

    method = str(method)
    pca_values = _parse_component_grid(pca_grid, name="pca_grid")
    pls_values = _parse_component_grid(pls_grid, name="pls_grid")
    n_start = int(start_X.shape[0])
    n_target = int(target_X.shape[0])
    n_min_class = min(n_start, n_target)
    input_dim = int(start_X.shape[1])

    fallback_pca = int(fallback_pca)
    fallback_pls = int(fallback_pls)
    if fallback_pca <= 0 or fallback_pls <= 0:
        raise ValueError("projection selection fallback PCA/PLS values must be positive")

    base_info: Dict[str, Any] = {
        "enabled": True,
        "method": method,
        "pca_grid": pca_values,
        "pls_grid": pls_values,
        "fit_frac": float(fit_frac),
        "repeats": int(repeats),
        "small_cell_threshold": int(small_cell_threshold),
        "fallback_pca": int(fallback_pca),
        "fallback_pls": int(fallback_pls),
        "selection_rule": selection_rule,
        "n_start_available": n_start,
        "n_target_available": n_target,
        "input_dim": input_dim,
        "metric": "heldout_normalized_centroid_separation",
    }

    def fallback(reason: str) -> Dict[str, Any]:
        pca_eff = int(min(fallback_pca, input_dim, max(n_start + n_target - 1, 1)))
        pls_eff = int(min(fallback_pls, pca_eff if method == "pca_pls_da" else input_dim, max(n_start + n_target - 1, 1)))
        if method == "pca":
            n_components = pca_eff
            pca_prefilter = pca_eff
            summary_row_pca = pca_eff
            summary_row_pls = None
        elif method == "pca_pls_da":
            n_components = pls_eff
            pca_prefilter = pca_eff
            summary_row_pca = pca_eff
            summary_row_pls = n_components
        else:
            n_components = pls_eff
            pca_prefilter = pca_eff
            summary_row_pca = None
            summary_row_pls = n_components
        summary_row = {
            "pca_components": summary_row_pca,
            "pls_components": summary_row_pls,
            "n_components": int(n_components),
            "mean_score": np.nan,
            "sem_score": np.nan,
            "n_repeats_scored": 0,
            "selected": True,
            "selection_skipped_reason": reason,
            "selected_by": "fallback",
        }
        return {
            **base_info,
            "selected_by": "fallback",
            "selection_skipped_reason": reason,
            "selected_pca_components": int(pca_prefilter),
            "selected_pls_components": int(n_components),
            "selected_projection_components": int(n_components),
            "results": [],
            "summary": [summary_row],
        }

    if n_min_class < int(small_cell_threshold):
        return fallback("small_dataset")

    rng = np.random.default_rng(int(seed))
    rows: List[Dict[str, Any]] = []
    summary_map: Dict[Tuple[int, int], List[float]] = {}
    for repeat in range(int(repeats)):
        s_perm = rng.permutation(n_start)
        t_perm = rng.permutation(n_target)
        s_cut = int(round(n_start * float(fit_frac)))
        t_cut = int(round(n_target * float(fit_frac)))
        s_cut = min(max(1, s_cut), n_start - 1)
        t_cut = min(max(1, t_cut), n_target - 1)
        s_fit_idx, s_eval_idx = s_perm[:s_cut], s_perm[s_cut:]
        t_fit_idx, t_eval_idx = t_perm[:t_cut], t_perm[t_cut:]
        start_fit = start_X[s_fit_idx]
        target_fit = target_X[t_fit_idx]
        start_eval = start_X[s_eval_idx]
        target_eval = target_X[t_eval_idx]
        max_components = int(min(input_dim, start_fit.shape[0] + target_fit.shape[0] - 1))
        if max_components < 1:
            continue

        if method == "pca_pls_da":
            candidates = [
                (pca, pls)
                for pca in pca_values
                for pls in pls_values
                if pls <= pca and pca <= max_components and pls <= max_components
            ]
        elif method == "pca":
            candidates = [(pca, pca) for pca in pca_values if pca <= max_components]
        elif method == "pls_da":
            candidates = [(pca_values[0], pls) for pls in pls_values if pls <= max_components]
        else:
            raise ValueError(f"Unsupported projection method for auto-selection: {method!r}")

        for pca, pls in candidates:
            projection = LinearProjection.fit(
                start_fit,
                target_fit,
                method=method,
                n_components=int(pls),
                fit_cap_per_class=fit_cap,
                seed=int(seed) + int(repeat),
                whiten_components=bool(whiten),
                pca_prefilter_components=int(pca),
            )
            start_Z = projection.transform_numpy(start_eval)
            target_Z = projection.transform_numpy(target_eval)
            score = float(_normalized_centroid_separation(start_Z, target_Z))
            key = (int(pca), int(pls))
            summary_map.setdefault(key, []).append(score)
            rows.append(
                {
                    "repeat": int(repeat),
                    "pca_components": int(pca) if method == "pca_pls_da" else (int(pca) if method == "pca" else None),
                    "pls_components": int(pls) if method != "pca" else None,
                    "n_components": int(pls),
                    "heldout_normalized_centroid_separation": score,
                    "fit_start_n": int(start_fit.shape[0]),
                    "fit_target_n": int(target_fit.shape[0]),
                    "eval_start_n": int(start_eval.shape[0]),
                    "eval_target_n": int(target_eval.shape[0]),
                }
            )

    if not rows:
        return fallback("no_valid_component_pairs")

    summary: List[Dict[str, Any]] = []
    for (pca, pls), scores in summary_map.items():
        arr = np.asarray(scores, dtype=np.float64)
        sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        summary.append(
            {
                "pca_components": int(pca) if method in {"pca", "pca_pls_da"} else None,
                "pls_components": int(pls) if method != "pca" else None,
                "n_components": int(pls),
                "mean_score": float(arr.mean()),
                "sem_score": sem,
                "n_repeats_scored": int(len(arr)),
            }
        )
    summary = sorted(summary, key=lambda row: (float(row["mean_score"]), -_component_complexity_key(row)[0], -_component_complexity_key(row)[1]), reverse=True)
    best = summary[0]
    selected = best
    selected_by = "best_mean_score"
    if selection_rule == "one_se":
        threshold = float(best["mean_score"]) - float(best["sem_score"])
        eligible = [row for row in summary if float(row["mean_score"]) >= threshold]
        selected = sorted(eligible, key=lambda row: (_component_complexity_key(row), -float(row["mean_score"])))[0]
        selected_by = "one_standard_error"

    selected_pca = int(selected["pca_components"]) if selected.get("pca_components") is not None else int(pca_values[0])
    selected_n = int(selected["n_components"])
    for row in summary:
        row["selected"] = bool(
            int(row["n_components"]) == selected_n
            and (
                row.get("pca_components") is None
                or int(row["pca_components"]) == selected_pca
            )
        )
    return {
        **base_info,
        "selected_by": selected_by,
        "selection_skipped_reason": None,
        "best_mean_score": float(best["mean_score"]),
        "best_sem_score": float(best["sem_score"]),
        "selected_mean_score": float(selected["mean_score"]),
        "selected_sem_score": float(selected["sem_score"]),
        "selected_pca_components": selected_pca,
        "selected_pls_components": selected_n if method != "pca" else None,
        "selected_projection_components": selected_n,
        "results": rows,
        "summary": summary,
    }


def estimate_sinkhorn_epsilon(
    target_state: TensorLike,
    *,
    metric: Literal["sqeuclidean", "euclidean", "cosine"] = "sqeuclidean",
    scale: float = 0.1,
    device: Optional[torch.device] = None,
) -> float:
    """Auto epsilon as scale × median pairwise cost in the target cloud."""
    from .scoring import as_3d_tensor, pairwise_cost_matrix

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Y = as_3d_tensor(target_state, device=device)
    if Y.shape[0] != 1:
        raise ValueError("estimate_sinkhorn_epsilon expects one target batch")
    C = pairwise_cost_matrix(Y, Y, metric=metric, normalize=False)
    n = C.shape[1]
    if n < 2:
        return float(scale)
    mask = ~torch.eye(n, dtype=torch.bool, device=C.device)
    costs = C[0][mask]
    median_cost = float(torch.median(costs).item())
    return float(scale * max(median_cost, 1e-12))


SMALL_DATASET_K_WARN_THRESHOLD = 64


def setup_projection_and_pools(
    *,
    adata: str | Path,
    start_cell: str,
    target_cell: str,
    cell_col: str,
    embed_key: str,
    method: str,
    n_components: int,
    whiten: bool,
    fit_cap: Optional[int],
    pca_prefilter: int,
    split_mode: str,
    split_frac: float,
    seed: int,
    small_dataset_threshold: int = 512,
    auto_select_components: bool = False,
    selection_pca_grid: Union[str, Sequence[int]] = (96, 128, 192, 256),
    selection_pls_grid: Union[str, Sequence[int]] = (32, 64, 96, 128, 192),
    selection_fit_frac: float = 0.5,
    selection_repeats: int = 10,
    selection_small_cell_threshold: int = 150,
    selection_fallback_pca: int = 128,
    selection_fallback_pls: int = 64,
    selection_rule: str = "one_se",
) -> Tuple[Any, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """
    Fit a LinearProjection on (start, target) cells and return evaluation pools.

    split_mode:
        "auto"    -> if min(n_start, n_target) <= small_dataset_threshold, behave
                     like "none" (fit on all cells, eval pools = None). Otherwise
                     behave like "holdout". A warning is emitted in the small-data
                     case when n_components is large (rotation may overfit).
        "none"    -> fit on all start/target cells; eval pools = None (use all).
        "holdout" -> split each class into FIT/EVAL by a seeded permutation;
                     fit the projection on FIT, restrict scoring to EVAL.

    The projection is only used at scoring time; ST-SE conversion always runs in
    the full embedding space. This helper is deliberately independent of the
    positive-control validation workflow so that the search can reuse it without
    importing that module.

    Returns (projection, start_eval_pool, target_eval_pool, split_info).
    """
    import scanpy as sc

    ad = sc.read_h5ad(adata)
    if cell_col not in ad.obs:
        raise KeyError(f"cell_col={cell_col!r} not found in adata.obs.")
    if embed_key not in ad.obsm:
        raise KeyError(f"embed_key={embed_key!r} not found in adata.obsm.")

    labels = ad.obs[cell_col].astype(str).values
    start_avail = np.where(labels == str(start_cell))[0]
    target_avail = np.where(labels == str(target_cell))[0]
    if len(start_avail) == 0 or len(target_avail) == 0:
        raise ValueError("start or target cells not found for projection fit.")

    X = np.asarray(ad.obsm[embed_key])
    rng = np.random.default_rng(int(seed))

    # Resolve the "auto" split based on how many cells are available per class.
    requested_split = str(split_mode)
    resolved_split = requested_split
    n_min_class = int(min(len(start_avail), len(target_avail)))
    small_dataset = False
    if requested_split == "auto":
        if n_min_class <= int(small_dataset_threshold):
            resolved_split = "none"
            small_dataset = True
            print(
                f"projection split=auto: small dataset "
                f"(start={len(start_avail)}, target={len(target_avail)}, "
                f"threshold={small_dataset_threshold}) -> fitting on ALL cells (no holdout)."
            )
        else:
            resolved_split = "holdout"
            print(
                f"projection split=auto: large dataset "
                f"(start={len(start_avail)}, target={len(target_avail)}) -> holdout."
            )

    if (
        small_dataset
        and str(method) != "none"
        and int(n_components) > SMALL_DATASET_K_WARN_THRESHOLD
    ):
        warnings.warn(
            f"Small dataset (min class size {n_min_class}) with n_components="
            f"{n_components}: the projection is fit on the same cells used for "
            f"scoring and a large K may overfit the rotation to noise. Consider "
            f"K<={SMALL_DATASET_K_WARN_THRESHOLD}.",
            stacklevel=2,
        )

    if resolved_split == "holdout":
        s_perm = rng.permutation(start_avail)
        t_perm = rng.permutation(target_avail)
        s_cut = max(1, int(round(len(s_perm) * float(split_frac))))
        t_cut = max(1, int(round(len(t_perm) * float(split_frac))))
        start_fit_idx, start_eval_idx = s_perm[:s_cut], s_perm[s_cut:]
        target_fit_idx, target_eval_idx = t_perm[:t_cut], t_perm[t_cut:]
        if len(start_eval_idx) == 0 or len(target_eval_idx) == 0:
            raise ValueError("Holdout split left an empty evaluation pool; adjust split_frac.")
        start_eval_pool, target_eval_pool = start_eval_idx, target_eval_idx
    elif resolved_split == "none":
        start_fit_idx, target_fit_idx = start_avail, target_avail
        start_eval_pool, target_eval_pool = None, None
    else:
        raise ValueError(f"Unknown split_mode: {requested_split!r}")

    print(f"projection split mode: {requested_split} (resolved: {resolved_split})")
    print(f"  fit start cells:  {len(start_fit_idx)}")
    print(f"  fit target cells: {len(target_fit_idx)}")
    if resolved_split == "holdout":
        print(f"  eval start pool:  {len(start_eval_pool)}")
        print(f"  eval target pool: {len(target_eval_pool)}")

    # method == "none" with split == "holdout" is the matched control arm:
    # restrict scoring to the same eval pool but score in full embedding space.
    projection = None
    component_selection_info: Optional[Dict[str, Any]] = None
    if str(method) != "none":
        start_fit_X = X[start_fit_idx].astype(np.float32)
        target_fit_X = X[target_fit_idx].astype(np.float32)
        effective_n_components = int(n_components)
        effective_pca_prefilter = int(pca_prefilter)
        if bool(auto_select_components):
            component_selection_info = select_projection_components(
                start_fit_X,
                target_fit_X,
                method=str(method),
                pca_grid=selection_pca_grid,
                pls_grid=selection_pls_grid,
                fit_frac=float(selection_fit_frac),
                repeats=int(selection_repeats),
                small_cell_threshold=int(selection_small_cell_threshold),
                fallback_pca=int(selection_fallback_pca),
                fallback_pls=int(selection_fallback_pls),
                fit_cap=fit_cap,
                seed=int(seed),
                whiten=bool(whiten),
                selection_rule=str(selection_rule),
            )
            effective_n_components = int(component_selection_info["selected_projection_components"])
            effective_pca_prefilter = int(component_selection_info["selected_pca_components"])
            print("\n=== Projection component auto-selection ===")
            if component_selection_info.get("selection_skipped_reason"):
                print(
                    "auto-selection skipped: "
                    f"{component_selection_info['selection_skipped_reason']}; "
                    f"using PCA={effective_pca_prefilter}, components={effective_n_components}"
                )
            else:
                print(
                    "selected projection components: "
                    f"PCA={effective_pca_prefilter}, components={effective_n_components}; "
                    f"rule={component_selection_info.get('selected_by')}"
                )
        projection = LinearProjection.fit(
            start_fit_X,
            target_fit_X,
            method=method,
            n_components=effective_n_components,
            fit_cap_per_class=fit_cap,
            seed=int(seed),
            whiten_components=bool(whiten),
            pca_prefilter_components=effective_pca_prefilter,
        )
        if component_selection_info is not None:
            projection.fit_metadata["component_selection"] = {
                key: value
                for key, value in component_selection_info.items()
                if key not in {"results", "summary"}
            }
        diagnose_projection(start_fit_X, target_fit_X, projection, verbose=True)

    split_info = {
        "split_mode": requested_split,
        "resolved_split": resolved_split,
        "small_dataset": bool(small_dataset),
        "small_dataset_threshold": int(small_dataset_threshold),
        "n_start_available": int(len(start_avail)),
        "n_target_available": int(len(target_avail)),
        "split_frac": float(split_frac) if resolved_split == "holdout" else None,
        "projection_method": method,
        "projection_auto_select_components": bool(auto_select_components),
        "n_start_fit": int(len(start_fit_idx)),
        "n_target_fit": int(len(target_fit_idx)),
        "n_start_eval_pool": int(len(start_eval_pool)) if start_eval_pool is not None else None,
        "n_target_eval_pool": int(len(target_eval_pool)) if target_eval_pool is not None else None,
        "projection": projection.metadata_dict() if projection is not None else None,
        "projection_component_selection": component_selection_info,
    }
    return projection, start_eval_pool, target_eval_pool, split_info
