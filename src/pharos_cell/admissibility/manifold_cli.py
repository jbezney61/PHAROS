#!/usr/bin/env python
"""
PHAROS embedding-manifold admissibility CLI.

CLI wrapper for ST-SE embedding manifold QC.

Two-step workflow
-----------------
1. Build a reusable reference bundle:

   python embedding_manifold_qc_analysis.py build-reference \
     --reference-h5ad data/merged_5um_perturbations_plus_DMSO_100_per_cell_line_log1p_norm10k.SE600M.h5ad \
     --output-dir runs/tahoe_reference_manifold \
     --cell-line-metadata metadata/cell_line_metadata.csv \
     --k 50

2. Score a query h5ad against that reference:

   python embedding_manifold_qc_analysis.py score-query \
     --reference-dir runs/tahoe_reference_manifold \
     --query-h5ad data/query.SE600M.h5ad \
     --query-state-col cell_type \
     --output-dir runs/query_manifold_qc
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def prepare_output_dir(path: str | Path, overwrite: bool) -> Path:
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


def write_cli_manifest(args: argparse.Namespace, output_dir: Path) -> Path:
    path = output_dir / "embedding_manifold_qc_cli_args.json"
    payload = vars(args).copy()
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def add_common_compute_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpu-id", type=int, default=0, help="FAISS GPU id to use.")
    p.add_argument(
        "--require-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require FAISS GPU execution. Use --no-require-gpu only for tiny smoke tests.",
    )
    p.add_argument("--add-batch-size", type=int, default=100_000, help="Reference vectors added to FAISS per batch.")
    p.add_argument("--search-batch-size", type=int, default=16_384, help="Query/search vectors processed per FAISS batch.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pharos admissibility manifold",
        description="Build and run FAISS-based ST-SE embedding manifold QC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "build-reference",
        help="Build a reusable reference embedding bundle and calibration distributions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build.add_argument("--reference-h5ad", required=True, help="Reference h5ad containing ST-SE embeddings.")
    build.add_argument("--output-dir", required=True, help="Reference bundle output directory.")
    build.add_argument("--reference-cell-col", default="cell_name", help="Reference obs column containing cell-line name.")
    build.add_argument(
        "--reference-perturbation-col",
        default="drugname_drugconc",
        help="Reference obs column containing drug/concentration label.",
    )
    build.add_argument(
        "--reference-state-col",
        default=None,
        help="Optional precomputed reference state column. If omitted, state is cell_name | drugname_drugconc.",
    )
    build.add_argument("--embed-key", default="X_state", help="adata.obsm key containing ST-SE embeddings.")
    build.add_argument("--metric", choices=["l2", "cosine"], default="l2", help="FAISS distance metric.")
    build.add_argument("--k", type=int, default=50, help="k for kNN support scoring and calibration.")
    build.add_argument("--seed", type=int, default=42, help="Random seed for calibration sampling.")
    build.add_argument(
        "--calibration-cells-per-state",
        type=int,
        default=100,
        help="Cells sampled per reference state for state-level calibration.",
    )
    build.add_argument("--calibration-splits", type=int, default=3, help="Heldout calibration splits.")
    build.add_argument(
        "--calibration-state-fraction",
        type=float,
        default=0.10,
        help="Fraction of reference states held out in each heldout-state split.",
    )
    build.add_argument(
        "--calibration-max-states-per-split",
        type=int,
        default=250,
        help="Cap on heldout states scored in each heldout-state split.",
    )
    build.add_argument(
        "--calibration-cell-names-per-split",
        type=int,
        default=5,
        help="Cell lines held out in each heldout-cell-line split.",
    )
    build.add_argument(
        "--calibration-max-cell-name-states-per-split",
        type=int,
        default=250,
        help="Cap on states scored per heldout-cell-line split.",
    )
    build.add_argument(
        "--skip-heldout-state-calibration",
        action="store_true",
        help="Skip heldout whole-state calibration.",
    )
    build.add_argument(
        "--skip-heldout-cell-name-calibration",
        action="store_true",
        help="Skip heldout whole-cell-line calibration.",
    )
    build.add_argument(
        "--cell-line-metadata",
        default="metadata/cell_line_metadata.csv",
        help="CSV containing cell_name and Organ columns for nearest reference tissue annotation.",
    )
    build.add_argument(
        "--save-faiss-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also save a CPU FAISS index. This duplicates the reference embedding storage but can speed reuse.",
    )
    build.add_argument(
        "--max-reference-cells",
        type=int,
        default=None,
        help="Optional first-N reference cell limiter for smoke tests.",
    )
    build.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite output dir.")
    add_common_compute_args(build)

    score = sub.add_parser(
        "score-query",
        help="Score query cell states against a built reference bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    score.add_argument("--reference-dir", required=True, help="Directory produced by build-reference.")
    score.add_argument("--query-h5ad", required=True, help="Query h5ad containing ST-SE embeddings.")
    score.add_argument("--output-dir", required=True, help="Query QC output directory.")
    score.add_argument("--query-state-col", default="cell_type", help="Query obs column defining query cell states.")
    score.add_argument("--embed-key", default="X_state", help="adata.obsm key containing ST-SE embeddings.")
    score.add_argument(
        "--k",
        type=int,
        default=None,
        help="k for query scoring. Defaults to the reference calibration k; mismatches require --allow-k-mismatch.",
    )
    score.add_argument(
        "--query-cells-per-state",
        type=int,
        default=100,
        help="Maximum query cells scored per query state.",
    )
    score.add_argument("--seed", type=int, default=42, help="Random seed for query state downsampling.")
    score.add_argument(
        "--allow-k-mismatch",
        action="store_true",
        help="Allow query k to differ from the reference calibration k. Percentile calibration may be less interpretable.",
    )
    score.add_argument(
        "--use-saved-index",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use reference.faiss if the reference bundle saved one.",
    )
    score.add_argument(
        "--save-query-neighbors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save full query-to-reference kNN indices/distances for local UMAP overlay reports.",
    )
    score.add_argument("--skip-report", action="store_true", help="Run scoring only; skip report/figure generation.")
    score.add_argument("--report-output-dir", default=None, help="Report output directory. Default: <output-dir>/report.")
    score.add_argument("--report-top-n-states", type=int, default=40, help="Max query states shown in crowded figures.")
    score.add_argument(
        "--report-local-umap-neighbors-per-query",
        type=int,
        default=50,
        help="Number of nearest Tahoe neighbors per query cell used in the optional local reference UMAP.",
    )
    score.add_argument(
        "--report-local-umap-max-reference-cells",
        type=int,
        default=50_000,
        help="Maximum unique Tahoe reference cells plotted in the optional local reference UMAP.",
    )
    score.add_argument(
        "--report-local-umap-max-query-cells",
        type=int,
        default=25_000,
        help="Maximum query cells plotted in the optional local reference UMAP.",
    )
    score.add_argument("--report-local-umap-seed", type=int, default=42, help="Random seed for optional local UMAP plotting.")
    score.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite output dir.")
    add_common_compute_args(score)
    return parser


def run_build_reference(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    manifest = write_cli_manifest(args, output_dir)
    print(f"CLI manifest: {manifest}")

    from .manifold import build_reference_manifold

    return build_reference_manifold(
        reference_h5ad=args.reference_h5ad,
        output_dir=output_dir,
        reference_cell_col=args.reference_cell_col,
        reference_perturbation_col=args.reference_perturbation_col,
        reference_state_col=args.reference_state_col,
        embed_key=args.embed_key,
        metric=args.metric,
        k=args.k,
        seed=args.seed,
        add_batch_size=args.add_batch_size,
        search_batch_size=args.search_batch_size,
        calibration_cells_per_state=args.calibration_cells_per_state,
        calibration_splits=args.calibration_splits,
        calibration_state_fraction=args.calibration_state_fraction,
        calibration_max_states_per_split=args.calibration_max_states_per_split,
        calibration_cell_names_per_split=args.calibration_cell_names_per_split,
        calibration_max_cell_name_states_per_split=args.calibration_max_cell_name_states_per_split,
        skip_heldout_state_calibration=args.skip_heldout_state_calibration,
        skip_heldout_cell_name_calibration=args.skip_heldout_cell_name_calibration,
        cell_line_metadata=args.cell_line_metadata,
        gpu_id=args.gpu_id,
        require_gpu=args.require_gpu,
        save_faiss_index=args.save_faiss_index,
        max_reference_cells=args.max_reference_cells,
    )


def run_score_query(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    manifest = write_cli_manifest(args, output_dir)
    print(f"CLI manifest: {manifest}")

    from .manifold import run_query_manifold_qc

    out = run_query_manifold_qc(
        reference_dir=args.reference_dir,
        query_h5ad=args.query_h5ad,
        output_dir=output_dir,
        query_state_col=args.query_state_col,
        embed_key=args.embed_key,
        k=args.k,
        query_cells_per_state=args.query_cells_per_state,
        seed=args.seed,
        add_batch_size=args.add_batch_size,
        search_batch_size=args.search_batch_size,
        gpu_id=args.gpu_id,
        require_gpu=args.require_gpu,
        allow_k_mismatch=args.allow_k_mismatch,
        use_saved_index=args.use_saved_index,
        save_query_neighbors=args.save_query_neighbors,
    )

    if args.skip_report:
        print("\nReport skipped because --skip-report was set.")
    else:
        from .reports.manifold import make_embedding_manifold_qc_report

        report_out = make_embedding_manifold_qc_report(
            run_dir=output_dir,
            output_dir=args.report_output_dir,
            top_n_states=args.report_top_n_states,
            local_umap_neighbors_per_query=args.report_local_umap_neighbors_per_query,
            local_umap_max_reference_cells=args.report_local_umap_max_reference_cells,
            local_umap_max_query_cells=args.report_local_umap_max_query_cells,
            local_umap_seed=args.report_local_umap_seed,
        )
        print(f"report summary: {report_out['summary']}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-reference":
        out = run_build_reference(args)
        print("\n=== Reference build complete ===")
        print(f"output:   {out['output_dir']}")
        print(f"manifest: {out['manifest']}")
        print(f"states:   {out['metadata']['n_reference_states']}")
    elif args.command == "score-query":
        out = run_score_query(args)
        print("\n=== Query manifold QC complete ===")
        print(f"output:       {out['output_dir']}")
        print(f"state scores: {out['paths']['query_state_scores']}")
        print(f"cell scores:  {out['paths']['query_cell_scores']}")
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
