#!/usr/bin/env python
"""
cell_converter.py

End-to-end CLI for one ST-SE cell-state conversion search.

This wrapper runs the full workflow for a single conversion:

    1. Load starting and target cell-state embeddings from an SE-embedded h5ad.
    2. Load the ST-SE converter model onto GPU/CPU.
    3. Initialize the distributional scorer.
    4. Run deterministic or diverse beam search.
    5. Generate a generic search summary report with plots and tables.
    6. Optionally generate a sample/drug metadata report if metadata files are available.

It expects these companion modules to be importable from the same directory or PYTHONPATH:

    data_loader.py
    converter.py
    scoring.py
    search.py
    make_search_report.py
    make_sample_drug_report.py

Minimal example
---------------
    export CUDA_VISIBLE_DEVICES=0

    python cell_converter.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --cell-col cell_name \
      --embed-key X_state \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --output-dir runs/J82_to_A172 \
      --algorithm deterministic_beam \
      --max-depth 5 \
      --beam-size 32

Smoke test
----------------
    python cell_converter.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --start-cell J82 \
      --target-cell A-172 \
      --model-dir "$ST_RUN" \
      --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
      --output-dir runs/test_J82_to_A172 \
      --max-depth 2 \
      --beam-size 2 \
      --max-drugs-to-consider 6 \
      --prefilter-multiplier 3 \
      --converter-chunk-size 3 \
      --sinkhorn-iters 50

Outputs
-------
output_dir/
    search/
        results.tsv
        checkpoint.pt
        search_config.used.yaml
    cache/
        start_target_states.npz
    report/
        summary.md
        tables/
        figures/
    sample_drug_report/
        summary.md
        tables/
        figures/
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml


def parse_csv_or_none(x: Optional[str]) -> Optional[List[str]]:
    """Parse comma-separated string into a list, preserving None."""
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() in {"none", "null"}:
        return None
    return [v.strip() for v in x.split(",") if v.strip()]


def parse_int_grid(value: str, *, name: str) -> List[int]:
    raw = [part.strip() for part in str(value).replace(",", " ").split()]
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


def normalize_selection_rule(value: str) -> str:
    return str(value).replace("-", "_").casefold()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run an end-to-end ST-SE sequential drug search to convert one SE-embedded "
            "cell type/state into another."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = p.add_argument_group("Required inputs")
    required.add_argument("--adata", required=True, help="Input h5ad containing SE embeddings in adata.obsm[embed_key].")
    required.add_argument("--start-cell", required=True, help="Starting cell type/cell line label, e.g. J82.")
    required.add_argument("--target-cell", required=True, help="Target cell type/cell line label, e.g. A-172.")
    required.add_argument("--model-dir", required=True, help="ST-SE training run directory.")
    required.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")

    data = p.add_argument_group("Data loading")
    data.add_argument("--checkpoint", default=None, help="Path to ST-SE checkpoint. Defaults to model_dir/checkpoints/final.ckpt.")
    data.add_argument("--cell-col", default="cell_name", help="adata.obs column containing start/target labels.")
    data.add_argument("--embed-key", default="X_state", help="adata.obsm key containing SE embeddings.")
    data.add_argument("--start-sample", default="256", help="Number of start cells to sample, or 'all'.")
    data.add_argument("--target-sample", default="256", help="Number of target cells to sample, or 'all'.")
    data.add_argument("--seed", type=int, default=42, help="Random seed for sampling and projections.")
    data.add_argument("--no-replace-if-needed", action="store_true", help="Error if requested sample size exceeds available cells.")
    data.add_argument("--save-state-cache", action=argparse.BooleanOptionalAction, default=True, help="Save start/target embeddings as npz.")
    data.add_argument(
        "--batch-selection",
        choices=["standard", "high-sensitivity"],
        default="standard",
        help=(
            "How the initial start/target batch is selected. 'standard' keeps random sampling. "
            "'high-sensitivity' screens local nearest-neighbor batches and keeps the most separated one."
        ),
    )
    data.add_argument(
        "--batch-candidates",
        type=int,
        default=300,
        help="Number of local candidate batch pairs screened when --batch-selection high-sensitivity.",
    )
    data.add_argument(
        "--batch-overlap-penalty",
        type=float,
        default=0.02,
        help=(
            "Soft overlap penalty for high-sensitivity greedy batch selection. "
            "Small values keep OT separation as the dominant criterion."
        ),
    )
    data.add_argument(
        "--batch-selection-score-chunk-size",
        type=int,
        default=8,
        help="Candidate batch pairs scored per chunk during high-sensitivity batch selection.",
    )

    compute = p.add_argument_group("Compute")
    compute.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu. Defaults to cuda:0 if available.")
    compute.add_argument("--max-set-len", type=int, default=256, help="Maximum cells per ST-SE forward pass.")
    compute.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True, help="Use autocast mixed precision on CUDA.")
    compute.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16", help="Autocast dtype.")

    search = p.add_argument_group("Search")
    search.add_argument("--config", default=None, help="Optional YAML search config; CLI values override it.")
    search.add_argument("--algorithm", choices=["deterministic_beam", "diverse_beam"], default="deterministic_beam")
    search.add_argument("--max-depth", type=int, default=5, help="Maximum number of sequential drugs.")
    search.add_argument("--beam-size", type=int, default=32, help="Number of paths retained after each depth.")
    search.add_argument("--prefilter-multiplier", type=int, default=10, help="Sinkhorn rerank pool = beam_size * this value.")
    search.add_argument(
        "--prefilter-metric",
        choices=["energy", "sinkhorn_low_iter"],
        default="sinkhorn_low_iter",
        help="Metric used to prefilter expanded candidates before full Sinkhorn reranking.",
    )
    search.add_argument(
        "--prefilter-sinkhorn-iters",
        type=int,
        default=10,
        help="Sinkhorn iterations used by the sinkhorn_low_iter prefilter (ignored for energy prefilter).",
    )
    search.add_argument("--converter-chunk-size", type=int, default=16, help="Number of perturbations processed per converter chunk.")
    search.add_argument("--max-drugs-to-consider", type=int, default=None, help="Optional limiter for smoke tests only.")

    filt = p.add_argument_group("Perturbation filters and constraints")
    filt.add_argument("--allow-repeated-drug-names", action=argparse.BooleanOptionalAction, default=False,
                      help="If false, a path cannot use the same base drug more than once, even at another dose.")
    filt.add_argument("--allow-repeated-perturbation-labels", action=argparse.BooleanOptionalAction, default=False,
                      help="If false, exact perturbation labels cannot repeat.")
    filt.add_argument("--allow-control-like-drugs", action=argparse.BooleanOptionalAction, default=False,
                      help="If false, DMSO/control-like perturbations are excluded.")
    filt.add_argument("--banned-drug-names", default=None, help="Comma-separated base drug names to exclude.")
    filt.add_argument("--banned-perturbation-labels", default=None, help="Comma-separated exact perturbation labels to exclude.")
    filt.add_argument("--allowed-drug-names", default=None, help="Comma-separated base drug names to allow; excludes all others.")
    filt.add_argument("--allowed-drug-name-contains", default=None, help="Comma-separated substrings allowed in base drug names.")

    diversity = p.add_argument_group("Diverse beam settings")
    diversity.add_argument("--path-overlap-penalty", type=float, default=0.05, help="Diverse beam path-overlap penalty.")
    diversity.add_argument("--use-state-similarity-penalty", action=argparse.BooleanOptionalAction, default=False)
    diversity.add_argument("--state-similarity-penalty", type=float, default=0.02)

    scoring = p.add_argument_group("Scoring")
    scoring.add_argument("--sinkhorn-epsilon", type=float, default=0.05, help="Entropic regularization for Sinkhorn OT.")
    scoring.add_argument("--sinkhorn-iters", type=int, default=100, help="Sinkhorn iterations.")
    scoring.add_argument("--sinkhorn-metric", choices=["cosine", "sqeuclidean", "euclidean"], default="cosine")
    scoring.add_argument("--no-normalize-embeddings", action="store_true", help="Disable L2 normalization before scoring.")

    projection = p.add_argument_group("Projection (optional PLS-DA / PCA scoring subspace)")
    projection.add_argument(
        "--projection-method",
        choices=["none", "pls_da", "pca_pls_da", "pca"],
        default="none",
        help="Linear dimensionality reduction applied only at scoring time.",
    )
    projection.add_argument("--projection-components", type=int, default=128, help="Number of latent dimensions K.")
    projection.add_argument(
        "--projection-whiten",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whiten projected components by their training std. Default off (best for ST-SE outputs).",
    )
    projection.add_argument(
        "--projection-fit-cap",
        type=int,
        default=4000,
        help="Max cells per class used to fit the projection (subsample if exceeded).",
    )
    projection.add_argument(
        "--projection-target-split",
        choices=["auto", "none", "holdout"],
        default="auto",
        help="'auto' fits on all cells when min class size <= threshold, else holdout. "
        "'holdout' always splits fit/eval. 'none' fits on all cells.",
    )
    projection.add_argument(
        "--projection-split-frac",
        type=float,
        default=0.5,
        help="Fraction of cells per class used to fit the projection in holdout mode.",
    )
    projection.add_argument(
        "--projection-small-dataset-threshold",
        type=int,
        default=512,
        help="In 'auto' mode, fit on all cells when min(start, target) cell count <= this.",
    )
    projection.add_argument(
        "--projection-cache-path",
        default=None,
        help="Path to save/load projection .npz. Default: <output>/cache/projection.npz",
    )
    projection.add_argument(
        "--projection-auto-epsilon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When projection is active, set Sinkhorn epsilon to 0.1 × median target pairwise cost.",
    )
    projection.add_argument(
        "--projection-pca-prefilter",
        type=int,
        default=256,
        help="PCA prefilter dimensions for --projection-method pca_pls_da.",
    )
    projection.add_argument(
        "--projection-auto-select-components",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Select PCA/PLS component counts from held-out start/target geometry before fitting the final projection. "
            "When enabled, --projection-components and --projection-pca-prefilter are replaced by the selected values."
        ),
    )
    projection.add_argument(
        "--projection-selection-pca-grid",
        default="96,128,192,256",
        help="Comma- or space-separated PCA prefilter candidates for --projection-auto-select-components.",
    )
    projection.add_argument(
        "--projection-selection-pls-grid",
        default="32,64,96,128,192",
        help="Comma- or space-separated PLS/component candidates for --projection-auto-select-components.",
    )
    projection.add_argument(
        "--projection-selection-fit-frac",
        type=float,
        default=0.5,
        help="Fraction of start and target cells used for each projection-selection fit split.",
    )
    projection.add_argument(
        "--projection-selection-repeats",
        type=int,
        default=10,
        help="Repeated stratified fit/eval splits used for projection component selection.",
    )
    projection.add_argument(
        "--projection-selection-small-cell-threshold",
        type=int,
        default=150,
        help="If either class has fewer cells than this, skip selection and use fallback PCA/PLS values.",
    )
    projection.add_argument(
        "--projection-selection-fallback-pca",
        type=int,
        default=128,
        help="Fallback PCA prefilter components when auto-selection is skipped for a small dataset.",
    )
    projection.add_argument(
        "--projection-selection-fallback-pls",
        type=int,
        default=64,
        help="Fallback PLS/components when auto-selection is skipped for a small dataset.",
    )
    projection.add_argument(
        "--projection-selection-rule",
        choices=["one_se", "best"],
        default="one_se",
        help="Rule used to select from projection sweep results.",
    )

    robust = p.add_argument_group("Robust reranking")
    robust.add_argument(
        "--robust-rerank",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="At each depth, rerank the post-Sinkhorn candidate pool across additional sampled start/target batches.",
    )
    robust.add_argument("--robust-n-samples", type=int, default=3, help="Number of additional sampled batches used for robust reranking.")
    robust.add_argument(
        "--robust-start-sample",
        default=None,
        help="Start sample size for robust reranking batches. Defaults to --start-sample.",
    )
    robust.add_argument(
        "--robust-target-sample",
        default=None,
        help="Target sample size for robust reranking batches. Defaults to --target-sample.",
    )
    robust.add_argument(
        "--robust-seed-offset",
        type=int,
        default=1000,
        help="Robust batch seeds are seed + robust_seed_offset + batch_index.",
    )
    robust.add_argument(
        "--robust-metric",
        choices=["sinkhorn", "energy_distance"],
        default="sinkhorn",
        help="Metric used to score each robust validation batch.",
    )
    robust.add_argument(
        "--robust-aggregation",
        choices=["mean", "median", "worst", "mean_plus_std"],
        default="mean_plus_std",
        help="How per-batch robust scores are aggregated for candidate selection.",
    )
    robust.add_argument(
        "--robust-std-penalty",
        type=float,
        default=0.5,
        help="Penalty multiplier used when --robust-aggregation mean_plus_std.",
    )

    out = p.add_argument_group("Output and report")
    out.add_argument("--state-checkpoint-dtype", choices=["float16", "float32"], default="float16")
    out.add_argument("--skip-report", action="store_true", help="Run search only; skip the generic search report.")
    out.add_argument("--conversion-threshold", type=float, default=None, help="Report-only Sinkhorn threshold for counting conversions.")
    out.add_argument("--report-top-n-paths", type=int, default=25)
    out.add_argument("--report-top-n-drugs", type=int, default=20)

    out.add_argument(
        "--skip-sample-drug-report",
        action="store_true",
        help=(
            "Skip metadata-aware sample/drug report. By default, this report is generated "
            "if metadata files are available or explicitly provided."
        ),
    )
    out.add_argument(
        "--metadata-dir",
        default="metadata",
        help="Directory containing cell_line_metadata.csv and drug_metadata.csv for the sample/drug report.",
    )
    out.add_argument(
        "--cell-metadata",
        default=None,
        help="Optional explicit path to cell_line_metadata.csv.",
    )
    out.add_argument(
        "--drug-metadata",
        default=None,
        help="Optional explicit path to drug_metadata.csv.",
    )
    out.add_argument(
        "--sample-drug-top-n-paths",
        type=int,
        default=50,
        help="Number of top paths used for the sample/drug metadata report.",
    )
    out.add_argument(
        "--sample-drug-background",
        choices=["metadata", "search"],
        default="metadata",
        help=(
            "Background universe for MOA enrichment in sample/drug report. "
            "'metadata' uses all drugs in drug_metadata.csv; 'search' uses drugs observed in results.tsv."
        ),
    )
    out.add_argument(
        "--sample-drug-final-depth-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, sample/drug report analyzes top paths only from the deepest search depth. "
            "If false, it analyzes top paths across all depths."
        ),
    )

    out.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite non-empty output directory.")

    return p.parse_args()


def load_yaml_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update base with update."""
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def build_search_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a search.py-compatible config from YAML + CLI args."""
    cfg = load_yaml_config(args.config)

    cli_cfg = {
        "search": {
            "algorithm": args.algorithm,
            "max_depth": args.max_depth,
            "beam_size": args.beam_size,
            "prefilter_multiplier": args.prefilter_multiplier,
            "prefilter_metric": args.prefilter_metric,
            "prefilter_sinkhorn_iters": args.prefilter_sinkhorn_iters,
            "converter_chunk_size": args.converter_chunk_size,
            "max_drugs_to_consider": args.max_drugs_to_consider,
        },
        "constraints": {
            "allow_repeated_drug_names": args.allow_repeated_drug_names,
            "allow_repeated_perturbation_labels": args.allow_repeated_perturbation_labels,
            "allow_control_like_drugs": args.allow_control_like_drugs,
            "banned_drug_names": parse_csv_or_none(args.banned_drug_names) or [],
            "banned_perturbation_labels": parse_csv_or_none(args.banned_perturbation_labels) or [],
            "allowed_drug_names": parse_csv_or_none(args.allowed_drug_names),
            "allowed_drug_name_contains": parse_csv_or_none(args.allowed_drug_name_contains),
        },
        "diversity": {
            "path_overlap_penalty": args.path_overlap_penalty,
            "use_state_similarity_penalty": args.use_state_similarity_penalty,
            "state_similarity_penalty": args.state_similarity_penalty,
        },
        "robustness": {
            "enabled": args.robust_rerank,
            "n_samples": args.robust_n_samples,
            "start_sample": args.robust_start_sample or args.start_sample,
            "target_sample": args.robust_target_sample or args.target_sample,
            "seed_offset": args.robust_seed_offset,
            "metric": args.robust_metric,
            "aggregation": args.robust_aggregation,
            "std_penalty": args.robust_std_penalty,
        },
        "output": {
            "output_dir": "",
            "state_checkpoint_dtype": args.state_checkpoint_dtype,
        },
    }
    return deep_update(cfg, cli_cfg)


def prepare_output_dirs(args: argparse.Namespace) -> Dict[str, Path]:
    output_dir = Path(args.output_dir)
    search_dir = output_dir / "search"
    cache_dir = output_dir / "cache"
    report_dir = output_dir / "report"
    sample_drug_report_dir = output_dir / "sample_drug_report"

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output_dir}\n"
                "Use --overwrite or choose a new --output-dir."
            )

    search_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    sample_drug_report_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "search_dir": search_dir,
        "cache_dir": cache_dir,
        "report_dir": report_dir,
        "sample_drug_report_dir": sample_drug_report_dir,
    }


def _check_eval_pool_capacity(name: str, pool, sample) -> None:
    """Warn if a held-out eval pool is smaller than the requested sample size."""
    if pool is None:
        return
    try:
        requested = int(sample)
    except (TypeError, ValueError):
        return
    if len(pool) < requested:
        print(
            f"WARNING: {name} eval pool has {len(pool)} cells but --{name}-sample="
            f"{requested}. Cells will be sampled WITH replacement (duplicates). "
            f"Consider reducing --{name}-sample, lowering --projection-split-frac, "
            f"or using --projection-target-split none."
        )


def _validate_batch_selection_args(args: argparse.Namespace) -> None:
    if args.batch_selection != "high-sensitivity":
        return
    if int(args.batch_candidates) <= 0:
        raise ValueError("--batch-candidates must be positive when --batch-selection high-sensitivity")
    if int(args.batch_selection_score_chunk_size) <= 0:
        raise ValueError(
            "--batch-selection-score-chunk-size must be positive when --batch-selection high-sensitivity"
        )


def _validate_projection_args(args: argparse.Namespace) -> None:
    if not bool(args.projection_auto_select_components):
        return
    if args.projection_method == "none":
        raise ValueError("--projection-auto-select-components requires --projection-method != none")
    parse_int_grid(args.projection_selection_pca_grid, name="--projection-selection-pca-grid")
    parse_int_grid(args.projection_selection_pls_grid, name="--projection-selection-pls-grid")
    if not (0.0 < float(args.projection_selection_fit_frac) < 1.0):
        raise ValueError("--projection-selection-fit-frac must be between 0 and 1")
    if int(args.projection_selection_repeats) <= 0:
        raise ValueError("--projection-selection-repeats must be positive")
    if int(args.projection_selection_small_cell_threshold) <= 1:
        raise ValueError("--projection-selection-small-cell-threshold must be > 1")
    if int(args.projection_selection_fallback_pca) <= 0:
        raise ValueError("--projection-selection-fallback-pca must be positive")
    if int(args.projection_selection_fallback_pls) <= 0:
        raise ValueError("--projection-selection-fallback-pls must be positive")
    if normalize_selection_rule(args.projection_selection_rule) not in {"one_se", "best"}:
        raise ValueError("--projection-selection-rule must be 'one_se' or 'best'")


def _projection_selection_request(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "method": str(args.projection_method),
        "pca_grid": parse_int_grid(args.projection_selection_pca_grid, name="--projection-selection-pca-grid"),
        "pls_grid": parse_int_grid(args.projection_selection_pls_grid, name="--projection-selection-pls-grid"),
        "fit_frac": float(args.projection_selection_fit_frac),
        "repeats": int(args.projection_selection_repeats),
        "small_cell_threshold": int(args.projection_selection_small_cell_threshold),
        "fallback_pca": int(args.projection_selection_fallback_pca),
        "fallback_pls": int(args.projection_selection_fallback_pls),
        "selection_rule": normalize_selection_rule(args.projection_selection_rule),
    }


def _assert_projection_cache_matches_auto_selection(
    projection: Any,
    *,
    args: argparse.Namespace,
    cache_path: Path,
) -> None:
    if not bool(args.projection_auto_select_components):
        return
    component_selection = (getattr(projection, "fit_metadata", None) or {}).get("component_selection")
    if not component_selection:
        raise ValueError(
            f"Existing projection cache {cache_path} does not contain auto-selection metadata. "
            "Use --overwrite, delete the cache, or pass a fresh --projection-cache-path."
        )
    expected = _projection_selection_request(args)
    mismatches: List[str] = []
    for key, expected_value in expected.items():
        actual_value = component_selection.get(key)
        if isinstance(expected_value, float):
            try:
                actual_float = float(actual_value)
            except (TypeError, ValueError):
                mismatches.append(f"{key}: cache={actual_value!r}, requested={expected_value!r}")
                continue
            if abs(actual_float - expected_value) > 1e-12:
                mismatches.append(f"{key}: cache={actual_value!r}, requested={expected_value!r}")
        else:
            if actual_value != expected_value:
                mismatches.append(f"{key}: cache={actual_value!r}, requested={expected_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"Existing projection cache {cache_path} was auto-selected with different settings ({mismatch_text}). "
            "Use --overwrite, delete the cache, or pass a fresh --projection-cache-path."
        )


def _write_projection_component_selection_table(split_info: Dict[str, Any], dirs: Dict[str, Path]) -> Optional[Path]:
    component_selection = (split_info or {}).get("projection_component_selection") or {}
    rows = component_selection.get("summary") or []
    if not rows:
        return None

    path = dirs["cache_dir"] / "projection_component_selection.tsv"
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    split_info["projection_component_selection_path"] = str(path)
    return path


def setup_projection(
    args: argparse.Namespace,
    dirs: Dict[str, Path],
) -> Tuple[Optional[Any], Optional[Any], Optional[Any], Dict[str, Any]]:
    """
    Fit or load a scoring projection and return held-out eval pools.

    Returns (projection, start_eval_pool, target_eval_pool, split_info).
    When projection-method is none, all values are None/empty.
    """
    if args.projection_method == "none":
        return None, None, None, {}

    from projections import LinearProjection, setup_projection_and_pools

    cache_path = Path(args.projection_cache_path) if args.projection_cache_path else dirs["cache_dir"] / "projection.npz"
    pool_kwargs = dict(
        adata=args.adata,
        start_cell=args.start_cell,
        target_cell=args.target_cell,
        cell_col=args.cell_col,
        embed_key=args.embed_key,
        split_mode=args.projection_target_split,
        split_frac=args.projection_split_frac,
        seed=args.seed,
        small_dataset_threshold=args.projection_small_dataset_threshold,
    )

    if cache_path.exists() and not args.overwrite:
        print(f"loading projection cache: {cache_path}")
        projection = LinearProjection.load(cache_path)
        _assert_projection_cache_matches_auto_selection(projection, args=args, cache_path=cache_path)
        print(projection.summary())
        _, start_eval_pool, target_eval_pool, split_info = setup_projection_and_pools(
            **pool_kwargs,
            method="none",
            n_components=1,
            whiten=False,
            fit_cap=args.projection_fit_cap,
            pca_prefilter=args.projection_pca_prefilter,
        )
        if args.projection_auto_select_components:
            split_info["projection_component_selection"] = (projection.fit_metadata or {}).get("component_selection")
        return projection, start_eval_pool, target_eval_pool, split_info

    print("fitting projection from start/target embeddings")
    projection, start_eval_pool, target_eval_pool, split_info = setup_projection_and_pools(
        **pool_kwargs,
        method=args.projection_method,
        n_components=args.projection_components,
        whiten=args.projection_whiten,
        fit_cap=args.projection_fit_cap,
        pca_prefilter=args.projection_pca_prefilter,
        auto_select_components=bool(args.projection_auto_select_components),
        selection_pca_grid=str(args.projection_selection_pca_grid),
        selection_pls_grid=str(args.projection_selection_pls_grid),
        selection_fit_frac=float(args.projection_selection_fit_frac),
        selection_repeats=int(args.projection_selection_repeats),
        selection_small_cell_threshold=int(args.projection_selection_small_cell_threshold),
        selection_fallback_pca=int(args.projection_selection_fallback_pca),
        selection_fallback_pls=int(args.projection_selection_fallback_pls),
        selection_rule=str(args.projection_selection_rule),
    )
    selection_path = _write_projection_component_selection_table(split_info, dirs)
    if selection_path is not None:
        print(f"projection component selection: {selection_path}")
    projection.save(cache_path)
    print(f"saved projection cache: {cache_path}")
    return projection, start_eval_pool, target_eval_pool, split_info


def save_run_manifest(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    dirs: Dict[str, Path],
    *,
    projection_cache_path: Optional[str] = None,
    projection_split_info: Optional[Dict[str, Any]] = None,
) -> Path:
    manifest = {
        "args": vars(args),
        "search_config": cfg,
        "paths": {k: str(v) for k, v in dirs.items()},
    }
    if projection_cache_path is not None:
        manifest["projection_cache_path"] = projection_cache_path
    if projection_split_info:
        manifest["projection_split_info"] = projection_split_info
    path = dirs["output_dir"] / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def main() -> None:
    args = parse_args()
    _validate_batch_selection_args(args)
    _validate_projection_args(args)
    dirs = prepare_output_dirs(args)

    from data_loader import (
        load_high_sensitivity_start_target_embedding_batches,
        load_start_target_embedding_batches,
        load_start_target_embeddings,
    )
    from converter import StateSEConverter
    from scoring import DistributionScorer
    from search import run_search, save_search_config

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    print("\n=== Cell converter workflow ===")
    print(f"start cell:  {args.start_cell}")
    print(f"target cell: {args.target_cell}")
    print(f"device:      {device}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")

    cfg = build_search_config(args)
    cfg["output"]["output_dir"] = str(dirs["search_dir"])

    save_search_config(cfg, dirs["output_dir"] / "cell_converter.search_config.yaml")

    projection = None
    start_eval_pool = None
    target_eval_pool = None
    projection_split_info: Dict[str, Any] = {}
    projection_cache_path = None
    if args.projection_method != "none":
        print("\n[1/5] Setting up scoring projection")
        projection, start_eval_pool, target_eval_pool, projection_split_info = setup_projection(args, dirs)
        projection_cache_path = str(
            Path(args.projection_cache_path) if args.projection_cache_path else dirs["cache_dir"] / "projection.npz"
        )
        _check_eval_pool_capacity("start", start_eval_pool, args.start_sample)
        _check_eval_pool_capacity("target", target_eval_pool, args.target_sample)

    print("\n[1/5] Loading start and target SE embeddings")
    if args.batch_selection == "high-sensitivity":
        print(
            "high-sensitivity batch search: "
            f"{args.batch_candidates} local candidates, overlap penalty={args.batch_overlap_penalty}"
        )
        selected_batches, batch_selection_candidates = load_high_sensitivity_start_target_embedding_batches(
            h5ad_path=args.adata,
            start_cell=args.start_cell,
            target_cell=args.target_cell,
            cell_col=args.cell_col,
            embed_key=args.embed_key,
            start_sample=args.start_sample,
            target_sample=args.target_sample,
            n_batches=1,
            n_candidates=int(args.batch_candidates),
            overlap_penalty=float(args.batch_overlap_penalty),
            score_chunk_size=int(args.batch_selection_score_chunk_size),
            seed=int(args.seed),
            seed_offset=0,
            replace_if_needed=not args.no_replace_if_needed,
            start_index_pool=start_eval_pool,
            target_index_pool=target_eval_pool,
            normalize=not args.no_normalize_embeddings,
            sinkhorn_metric=args.sinkhorn_metric,
            sinkhorn_epsilon=float(args.sinkhorn_epsilon),
            sinkhorn_iters=int(args.sinkhorn_iters),
            device=device,
            projection=projection,
            projection_auto_epsilon=bool(args.projection_auto_epsilon),
        )
        pair = selected_batches[0]
        batch_selection_path = dirs["cache_dir"] / "batch_selection_candidates.tsv"
        if hasattr(batch_selection_candidates, "to_csv"):
            batch_selection_candidates.to_csv(batch_selection_path, sep="\t", index=False)
            print(f"batch selection candidates: {batch_selection_path}")
        else:
            batch_selection_path = dirs["cache_dir"] / "batch_selection_candidates.json"
            batch_selection_path.write_text(json.dumps(batch_selection_candidates, indent=2, default=str))
            print(f"batch selection candidates: {batch_selection_path}")
        print(
            "selected initial batch: "
            f"candidate={pair.batch_selection_candidate_index}, "
            f"baseline_OT={pair.batch_selection_score_sinkhorn_ot:.6g}"
        )
    else:
        pair = load_start_target_embeddings(
            h5ad_path=args.adata,
            start_cell=args.start_cell,
            target_cell=args.target_cell,
            cell_col=args.cell_col,
            embed_key=args.embed_key,
            start_sample=args.start_sample,
            target_sample=args.target_sample,
            seed=args.seed,
            replace_if_needed=not args.no_replace_if_needed,
            start_index_pool=start_eval_pool,
            target_index_pool=target_eval_pool,
        )

    print(f"start embeddings:  {pair.start_embeddings.shape}")
    print(f"target embeddings: {pair.target_embeddings.shape}")
    print(f"start available/sample:  {pair.start_n_available}/{pair.start_n_sampled}")
    print(f"target available/sample: {pair.target_n_available}/{pair.target_n_sampled}")

    state_cache_path = dirs["cache_dir"] / "start_target_states.npz"
    if args.save_state_cache:
        pair.save_npz(state_cache_path)
        print(f"cached start/target embeddings: {state_cache_path}")

    robust_samples = None
    if args.robust_rerank:
        if args.robust_n_samples <= 0:
            raise ValueError("--robust-n-samples must be positive when --robust-rerank is enabled")

        robust_start_sample = args.robust_start_sample or args.start_sample
        robust_target_sample = args.robust_target_sample or args.target_sample

        print("\n[1b/5] Loading robust reranking start/target batches")
        print(f"robust batches:       {args.robust_n_samples}")
        print(f"robust start sample:  {robust_start_sample}")
        print(f"robust target sample: {robust_target_sample}")
        robust_samples = load_start_target_embedding_batches(
            h5ad_path=args.adata,
            start_cell=args.start_cell,
            target_cell=args.target_cell,
            cell_col=args.cell_col,
            embed_key=args.embed_key,
            start_sample=robust_start_sample,
            target_sample=robust_target_sample,
            n_batches=args.robust_n_samples,
            seed=args.seed,
            seed_offset=args.robust_seed_offset,
            replace_if_needed=not args.no_replace_if_needed,
            start_index_pool=start_eval_pool,
            target_index_pool=target_eval_pool,
        )
        for i, sample in enumerate(robust_samples, start=1):
            print(
                f"  batch {i}: start {sample.start_embeddings.shape}, "
                f"target {sample.target_embeddings.shape}, seed={sample.seed}"
            )

    manifest_path = save_run_manifest(
        args,
        cfg,
        dirs,
        projection_cache_path=projection_cache_path,
        projection_split_info=projection_split_info or None,
    )
    print(f"manifest:    {manifest_path}")

    print("\n[2/5] Loading ST-SE converter")
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    converter = StateSEConverter(
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        device=device,
        max_set_len=args.max_set_len,
        use_amp=args.use_amp,
        amp_dtype=amp_dtype,
    )

    print("\n[3/5] Initializing distribution scorer")
    use_projection = projection is not None
    scorer = DistributionScorer(
        target_state=pair.target_embeddings,
        device=device,
        normalize=(not args.no_normalize_embeddings) and not use_projection,
        sinkhorn_metric=args.sinkhorn_metric,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        projection=projection,
        projection_auto_metric=True,
        projection_auto_epsilon=bool(use_projection and args.projection_auto_epsilon),
    )
    if use_projection and scorer.projection is not None:
        print(scorer.projection.summary())

    print("\n[4/5] Running sequential perturbation search")
    search_out = run_search(
        converter=converter,
        scorer=scorer,
        start_embeddings=pair.start_embeddings,
        cfg=cfg,
        perturbations=None,
        output_dir=dirs["search_dir"],
        start_cell=args.start_cell,
        target_cell=args.target_cell,
        robust_samples=robust_samples,
    )
    print(f"search results:    {search_out['results_tsv']}")
    print(f"search checkpoint: {search_out['checkpoint']}")

    # ------------------------------------------------------------------
    # 5. Generate generic search report
    # ------------------------------------------------------------------
    if args.skip_report:
        print("\n[5/6] Skipping generic search report because --skip-report was set")
    else:
        print("\n[5/6] Generating generic search report")
        from make_search_report import main as report_main

        old_argv = sys.argv[:]
        try:
            report_argv = [
                "make_search_report.py",
                "--search-dir", str(dirs["search_dir"]),
                "--output-dir", str(dirs["report_dir"]),
                "--target-npz", str(state_cache_path),
                "--top-n-paths", str(args.report_top_n_paths),
                "--top-n-drugs", str(args.report_top_n_drugs),
                "--seed", str(args.seed),
            ]
            if args.conversion_threshold is not None:
                report_argv.extend(["--conversion-threshold", str(args.conversion_threshold)])
            if args.device is not None:
                report_argv.extend(["--device", str(args.device)])
            if projection_cache_path is not None:
                report_argv.extend(["--projection-cache", projection_cache_path])

            sys.argv = report_argv
            report_main()
        finally:
            sys.argv = old_argv

    # ------------------------------------------------------------------
    # 6. Generate sample/drug metadata report
    # ------------------------------------------------------------------
    if args.skip_sample_drug_report:
        print("\n[6/6] Skipping sample/drug metadata report because --skip-sample-drug-report was set")
    else:
        print("\n[6/6] Generating sample/drug metadata report")

        metadata_dir = Path(args.metadata_dir)
        cell_metadata_path = Path(args.cell_metadata) if args.cell_metadata else metadata_dir / "cell_line_metadata.csv"
        drug_metadata_path = Path(args.drug_metadata) if args.drug_metadata else metadata_dir / "drug_metadata.csv"

        if not cell_metadata_path.exists() or not drug_metadata_path.exists():
            print(
                "Warning: sample/drug metadata report skipped because metadata files were not found.\n"
                f"  cell metadata: {cell_metadata_path}\n"
                f"  drug metadata: {drug_metadata_path}\n"
                "Provide --metadata-dir, --cell-metadata, or --drug-metadata to enable this report."
            )
        else:
            from make_sample_drug_report import main as sample_drug_report_main

            old_argv = sys.argv[:]
            try:
                sample_report_argv = [
                    "make_sample_drug_report.py",
                    "--run-dir", str(dirs["output_dir"]),
                    "--output-dir", str(dirs["sample_drug_report_dir"]),
                    "--start-cell", str(args.start_cell),
                    "--cell-metadata", str(cell_metadata_path),
                    "--drug-metadata", str(drug_metadata_path),
                    "--top-n-paths", str(args.sample_drug_top_n_paths),
                    "--background", str(args.sample_drug_background),
                ]

                if args.sample_drug_final_depth_only:
                    sample_report_argv.append("--final-depth-only")
                else:
                    sample_report_argv.append("--no-final-depth-only")

                sys.argv = sample_report_argv
                sample_drug_report_main()
            finally:
                sys.argv = old_argv

    print("\n=== Workflow complete ===")
    print(f"output:              {dirs['output_dir']}")
    print(f"search:              {dirs['search_dir']}")
    print(f"cache:               {dirs['cache_dir']}")
    if not args.skip_report:
        print(f"generic report:      {dirs['report_dir'] / 'summary.md'}")
    if not args.skip_sample_drug_report:
        print(f"sample/drug report:  {dirs['sample_drug_report_dir'] / 'summary.md'}")


if __name__ == "__main__":
    main()
