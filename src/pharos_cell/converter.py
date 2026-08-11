"""
converter.py

In-memory converter for Arc STATE ST-SE models.

This module wraps a trained StateTransitionPerturbationModel so it can be used
inside a beam/tree search without repeatedly reading/writing AnnData files.

Core use:
    converter = StateSEConverter(model_dir=ST_RUN, checkpoint=ST_CKPT)
    y = converter.convert_one(x_state_256, perturbation_label)

Input/output embeddings are expected to be SE embeddings, e.g. adata.obsm['X_state']
with shape [n_cells, embedding_dim]. For the Tahoe ST-SE checkpoint used in the
examples, n_cells is typically 256 and embedding_dim is typically 2058.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml

ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class ConversionResult:
    """Container for one perturbation conversion result."""

    perturbation: str
    embeddings: torch.Tensor  # [n_cells, emb_dim]


class StateSEConverter:
    """
    In-memory ST-SE converter for sequential latent-space perturbation search.

    This avoids the CLI's AnnData padding/writing. It exposes the ST-SE model as:

        current SE embeddings + perturbation label -> predicted SE embeddings

    Notes
    -----
    - Perturbation labels must exactly match keys in model_dir/pert_onehot_map.pt.
    - The converter keeps the model and perturbation one-hot vectors on GPU when
      CUDA is available.
    - For beam search, prefer convert_many_iter() so you can score/prune chunks
      without materializing all perturbation outputs at once.
    """

    def __init__(
        self,
        model_dir: str,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        max_set_len: Optional[int] = 256,
        use_amp: bool = True,
        amp_dtype: Optional[torch.dtype] = None,
        strict_embedding_dim: bool = True,
        verbose: bool = True,
    ) -> None:
        from state.tx.models.state_transition import StateTransitionPerturbationModel

        self.model_dir = os.path.abspath(model_dir)
        self.checkpoint = checkpoint or os.path.join(self.model_dir, "checkpoints", "final.ckpt")
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_set_len = max_set_len
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = amp_dtype or self._default_amp_dtype()
        self.strict_embedding_dim = strict_embedding_dim
        self.verbose = verbose

        self.cfg = self._load_yaml(os.path.join(self.model_dir, "config.yaml"))
        self.var_dims = self._load_pickle(os.path.join(self.model_dir, "var_dims.pkl"))
        self.pert_dim = self.var_dims.get("pert_dim")
        self.batch_dim = self.var_dims.get("batch_dim")

        self.pert_onehot_map = torch.load(
            os.path.join(self.model_dir, "pert_onehot_map.pt"),
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(self.pert_onehot_map, dict):
            raise TypeError("Expected pert_onehot_map.pt to contain a dictionary")

        self.pert_name_lookup: Dict[str, object] = {str(k): k for k in self.pert_onehot_map.keys()}
        self.pert_names: List[str] = list(self.pert_name_lookup.keys())

        self.batch_onehot_map = self._try_load_onehot_map("batch_onehot_map")
        self.cell_type_onehot_map = self._try_load_onehot_map("cell_type_onehot_map")

        self.model = StateTransitionPerturbationModel.load_from_checkpoint(
            self.checkpoint,
            map_location="cpu",
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.max_set_len is None:
            self.max_set_len = int(getattr(self.model, "cell_sentence_len", 256))

        self.uses_batch_encoder = getattr(self.model, "batch_encoder", None) is not None
        self.output_space = getattr(
            self.model,
            "output_space",
            self.cfg.get("data", {}).get("kwargs", {}).get("output_space", "gene"),
        )

        # Cache perturbation vectors on device.
        self.pert_vecs: Dict[str, torch.Tensor] = {}
        for p_str, original_key in self.pert_name_lookup.items():
            vec = self.pert_onehot_map[original_key]
            if not torch.is_tensor(vec):
                vec = torch.as_tensor(vec)
            self.pert_vecs[p_str] = vec.float().to(self.device)

        if self.verbose:
            print("Loaded ST-SE converter")
            print(f"  model_dir:      {self.model_dir}")
            print(f"  checkpoint:     {self.checkpoint}")
            print(f"  device:         {self.device}")
            print(f"  n perturbations:{len(self.pert_names)}")
            print(f"  max_set_len:    {self.max_set_len}")
            print(f"  batch encoder:  {self.uses_batch_encoder}")
            print(f"  output_space:   {self.output_space}")
            print(f"  amp:            {self.use_amp} ({self.amp_dtype if self.use_amp else 'off'})")

    @staticmethod
    def _load_yaml(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing config file: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_pickle(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing pickle file: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _try_load_onehot_map(self, basename: str):
        for suffix in [".torch", ".pt", ".pkl"]:
            path = os.path.join(self.model_dir, basename + suffix)
            if not os.path.exists(path):
                continue
            if suffix == ".pkl":
                with open(path, "rb") as f:
                    return pickle.load(f)
            return torch.load(path, map_location="cpu", weights_only=False)
        return None

    def _default_amp_dtype(self) -> torch.dtype:
        # H100/H200 class GPUs are typically happy with bfloat16. Fall back to float16
        # if needed by passing amp_dtype=torch.float16 in the constructor.
        return torch.bfloat16

    def list_perturbations(self, include_control: bool = False) -> List[str]:
        """Return perturbation labels available in the checkpoint map."""
        if include_control:
            return list(self.pert_names)
        return [p for p in self.pert_names if "DMSO" not in p and "control" not in p.lower()]

    def find_perturbations(self, query: str) -> List[str]:
        """Case-insensitive substring search over perturbation labels."""
        q = query.lower()
        return [p for p in self.pert_names if q in p.lower()]

    def get_perturbation_vector(self, perturbation: str) -> torch.Tensor:
        """Return one-hot/vector for an exact perturbation label."""
        perturbation = str(perturbation)
        if perturbation not in self.pert_vecs:
            examples = "\n  ".join(self.pert_names[:10])
            raise KeyError(
                f"Perturbation not found in pert_onehot_map: {perturbation!r}\n"
                f"First available labels:\n  {examples}"
            )
        return self.pert_vecs[perturbation]

    def _ensure_tensor(self, embeddings: ArrayLike) -> torch.Tensor:
        if torch.is_tensor(embeddings):
            x = embeddings
        else:
            x = torch.as_tensor(embeddings)

        if x.ndim != 2:
            raise ValueError(f"Expected embeddings with shape [n_cells, emb_dim], got {tuple(x.shape)}")

        return x.to(self.device, dtype=torch.float32, non_blocking=True)

    def _prepare_batch(
        self,
        x_window: torch.Tensor,
        perturbation: str,
        batch_indices: Optional[torch.Tensor] = None,
    ) -> dict:
        vec = self.get_perturbation_vector(perturbation)
        pert_oh = vec.unsqueeze(0).repeat(x_window.shape[0], 1)

        batch = {
            "ctrl_cell_emb": x_window,
            "pert_emb": pert_oh,
            "pert_name": [perturbation] * x_window.shape[0],
        }
        if batch_indices is not None:
            batch["batch"] = batch_indices.to(self.device, dtype=torch.long, non_blocking=True)
        return batch

    def _extract_embedding_prediction(self, batch_out: dict, input_dim: int) -> torch.Tensor:
        if "preds" not in batch_out:
            raise KeyError(f"Model output missing 'preds'. Keys: {list(batch_out.keys())}")
        pred = batch_out["preds"]
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred, device=self.device)

        if self.strict_embedding_dim and pred.shape[1] != input_dim:
            available = {k: tuple(v.shape) for k, v in batch_out.items() if torch.is_tensor(v)}
            raise RuntimeError(
                f"Expected latent embedding output dim {input_dim}, got {pred.shape[1]}. "
                f"Available tensor outputs: {available}. "
                "This checkpoint may be returning gene-space predictions for 'preds'."
            )
        return pred

    @torch.inference_mode()
    def convert_one(
        self,
        embeddings: ArrayLike,
        perturbation: str,
        batch_indices: Optional[torch.Tensor] = None,
        return_cpu: bool = False,
    ) -> torch.Tensor:
        """
        Apply one perturbation to one set of cell embeddings.

        Parameters
        ----------
        embeddings:
            Array/tensor with shape [n_cells, emb_dim], usually [256, 2058].
        perturbation:
            Exact perturbation label from pert_onehot_map.pt.
        batch_indices:
            Optional integer batch covariates if using a batch-encoder checkpoint.
        return_cpu:
            If True, return a CPU tensor. Otherwise return a tensor on self.device.
        """
        x = self._ensure_tensor(embeddings)
        input_dim = x.shape[1]

        outs: List[torch.Tensor] = []
        for start in range(0, x.shape[0], int(self.max_set_len)):
            end = min(start + int(self.max_set_len), x.shape[0])
            x_window = x[start:end]
            bi_window = batch_indices[start:end] if batch_indices is not None else None
            batch = self._prepare_batch(x_window, perturbation, bi_window)

            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    batch_out = self.model.predict_step(batch, batch_idx=0, padded=False)
            else:
                batch_out = self.model.predict_step(batch, batch_idx=0, padded=False)

            pred = self._extract_embedding_prediction(batch_out, input_dim=input_dim)
            outs.append(pred.float())

        y = torch.cat(outs, dim=0)
        return y.detach().cpu() if return_cpu else y

    @torch.inference_mode()
    def convert_many_iter(
        self,
        embeddings: ArrayLike,
        perturbations: Optional[Sequence[str]] = None,
        chunk_size: int = 16,
        return_cpu: bool = False,
    ) -> Iterator[Tuple[List[str], torch.Tensor]]:
        """
        Stream predictions for many perturbations.

        Yields
        ------
        labels:
            Perturbation labels in this chunk.
        pred_batch:
            Tensor with shape [chunk, n_cells, emb_dim].
        """
        if perturbations is None:
            perturbations = self.list_perturbations(include_control=False)
        perturbations = list(map(str, perturbations))

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        for i in range(0, len(perturbations), chunk_size):
            labels = perturbations[i : i + chunk_size]
            preds = [self.convert_one(embeddings, p, return_cpu=False) for p in labels]
            pred_batch = torch.stack(preds, dim=0)  # [B, n_cells, emb_dim]
            if return_cpu:
                pred_batch = pred_batch.detach().cpu()
            yield labels, pred_batch

    @torch.inference_mode()
    def convert_many(
        self,
        embeddings: ArrayLike,
        perturbations: Optional[Sequence[str]] = None,
        chunk_size: int = 16,
        return_cpu: bool = False,
    ) -> Tuple[List[str], torch.Tensor]:
        """Materialize predictions for many perturbations. Prefer convert_many_iter for large screens."""
        all_labels: List[str] = []
        all_preds: List[torch.Tensor] = []
        for labels, preds in self.convert_many_iter(
            embeddings,
            perturbations=perturbations,
            chunk_size=chunk_size,
            return_cpu=return_cpu,
        ):
            all_labels.extend(labels)
            all_preds.append(preds)
        return all_labels, torch.cat(all_preds, dim=0)
