#!/usr/bin/env python
"""
target_calibration_qc_analysis.py

CLI wrapper for ST-SE target calibration QC.

Example
-------
python target_calibration_QC/target_calibration_qc_analysis.py \
  --adata data/merged_5um_perturbations_plus_DMSO_100_per_cell_line.SE600M.h5ad \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --output-dir runs/target_calibration_qc \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in [REPO_ROOT, SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def parse_csv_or_none(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null"}:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def write_cli_manifest(args: argparse.Namespace, output_dir: Path, cell_types: Optional[Sequence[str]]) -> Path:
    path = output_dir / "target_calibration_qc_cli_args.json"
    payload = vars(args).copy()
    payload["cell_types_parsed"] = list(cell_types) if cell_types else None
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ST-SE target calibration QC against observed 5uM perturbation targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = p.add_argument_group("Core inputs")
    required.add_argument("--adata", "--input-h5ad", dest="input_h5ad", required=True, help="h5ad with DMSO WT and observed 5uM perturbation embeddings.")
    required.add_argument("--model-dir", required=True, help="ST-SE training run directory.")
    required.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")

    data = p.add_argument_group("Data loading")
    data.add_argument("--checkpoint", default=None, help="Path to ST-SE checkpoint. Defaults to model_dir/checkpoints/final.ckpt.")
    data.add_argument("--cell-col", default="cell_type", help="adata.obs column containing cell-line labels.")
    data.add_argument("--perturbation-col", default="drugname_drugconc", help="adata.obs column containing perturbation labels.")
    data.add_argument(
        "--control-label",
        default="DMSO",
        help="Untreated baseline label. Exact labels or short names like DMSO are accepted.",
    )
    data.add_argument(
        "--target-calibration-mode",
        choices=["raw", "dmso_start_only", "dmso-start-only", "dmso_adapter", "dmso-adapter", "both", "all"],
        default="all",
        help=(
            "Target calibration scoring mode. 'both' keeps the legacy raw + DMSO-adapter pair; "
            "'all' emits raw, DMSO-start-only, and DMSO-adapter rows."
        ),
    )
    data.add_argument(
        "--dmso-adapter-label",
        default=None,
        help="Exact converter perturbation label used to DMSO-adapt WT and, for dmso_adapter mode, observed target embeddings. Defaults to auto-detect.",
    )
    data.add_argument("--embed-key", default="X_state", help="adata.obsm key containing SE embeddings.")
    data.add_argument("--cells-per-state", type=int, default=100, help="WT and target cells sampled per cell line/drug.")
    data.add_argument("--cell-types", default=None, help="Optional comma-separated cell types. Default: all control cell types.")
    data.add_argument("--max-cell-types", type=int, default=None, help="Optional first-N cell-type limiter for smoke tests.")
    data.add_argument("--max-drugs", type=int, default=None, help="Optional first-N matched 5uM drug limiter for smoke tests.")
    data.add_argument("--seed", type=int, default=42, help="Random seed for WT and target cell sampling.")
    data.add_argument("--no-replace-if-needed", action="store_true", help="Error if fewer cells than --cells-per-state are available.")

    drugs = p.add_argument_group("Perturbation matching")
    drugs.add_argument("--drug-concentration", type=float, default=5.0, help="Concentration selected from observed and converter labels.")
    drugs.add_argument("--drug-unit", default="uM", help="Concentration unit selected from observed and converter labels.")

    compute = p.add_argument_group("Compute")
    compute.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu. Defaults to cuda:0 if available.")
    compute.add_argument("--max-set-len", type=int, default=100, help="Maximum cells per ST-SE forward pass.")
    compute.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True, help="Use autocast mixed precision on CUDA.")
    compute.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16", help="Autocast dtype.")
    compute.add_argument("--converter-chunk-size", type=int, default=8, help="Perturbation labels converted per progress chunk.")

    scoring = p.add_argument_group("Sinkhorn OT scoring")
    scoring.add_argument("--sinkhorn-metric", choices=["cosine", "sqeuclidean", "euclidean"], default="cosine")
    scoring.add_argument("--sinkhorn-epsilon", type=float, default=0.05, help="Entropic regularization.")
    scoring.add_argument("--sinkhorn-iters", type=int, default=100, help="Sinkhorn iterations.")
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
        help="Whiten projected components by their training std. Default off, matching the search workflow.",
    )
    projection.add_argument(
        "--projection-fit-cap",
        type=int,
        default=4000,
        help="Max cells per class used to fit each per-cell-line/per-drug projection.",
    )
    projection.add_argument(
        "--projection-target-split",
        choices=["auto", "none", "holdout"],
        default="auto",
        help=(
            "'auto' fits on all sampled cells when min(start, target) <= threshold, else holdout. "
            "'holdout' splits each sampled class into fit/eval. 'none' fits and scores on all sampled cells."
        ),
    )
    projection.add_argument(
        "--projection-split-frac",
        type=float,
        default=0.5,
        help="Fraction of sampled cells per class used to fit the projection in holdout mode.",
    )
    projection.add_argument(
        "--projection-small-dataset-threshold",
        type=int,
        default=512,
        help="In 'auto' mode, fit on all sampled cells when min(start, target) <= this.",
    )
    projection.add_argument(
        "--projection-auto-epsilon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When projection is active, set Sinkhorn epsilon to 0.1 x median target pairwise cost.",
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
            "Select PCA/PLS component counts from held-out start/target geometry before fitting "
            "each final scoring projection. When enabled, --projection-components and "
            "--projection-pca-prefilter are replaced by the selected values."
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

    report = p.add_argument_group("Report")
    report.add_argument("--skip-report", action="store_true", help="Run scoring only; skip report generation.")
    report.add_argument("--report-output-dir", default=None, help="Report output directory. Default: <output-dir>/report.")
    report.add_argument("--top-n-drugs", type=int, default=10, help="Number of high/low ranked drugs shown in report violins.")
    report.add_argument("--rank-stat", choices=["mean", "median"], default="mean", help="Statistic used to rank cell lines and drugs.")

    out = p.add_argument_group("Output")
    out.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite a non-empty output directory.")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cell_types = parse_csv_or_none(args.cell_types)

    from target_calibration_qc import run_target_calibration_qc

    out = run_target_calibration_qc(
        input_h5ad=args.input_h5ad,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        cell_col=args.cell_col,
        perturbation_col=args.perturbation_col,
        control_label=args.control_label,
        target_calibration_mode=args.target_calibration_mode,
        dmso_adapter_label=args.dmso_adapter_label,
        embed_key=args.embed_key,
        cells_per_state=args.cells_per_state,
        drug_concentration=args.drug_concentration,
        drug_unit=args.drug_unit,
        seed=args.seed,
        replace_if_needed=not args.no_replace_if_needed,
        cell_types=cell_types,
        max_cell_types=args.max_cell_types,
        max_drugs=args.max_drugs,
        device=args.device,
        max_set_len=args.max_set_len,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        converter_chunk_size=args.converter_chunk_size,
        normalize_embeddings=not args.no_normalize_embeddings,
        sinkhorn_metric=args.sinkhorn_metric,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        projection_method=args.projection_method,
        projection_components=args.projection_components,
        projection_whiten=args.projection_whiten,
        projection_fit_cap=args.projection_fit_cap,
        projection_target_split=args.projection_target_split,
        projection_split_frac=args.projection_split_frac,
        projection_small_dataset_threshold=args.projection_small_dataset_threshold,
        projection_auto_epsilon=args.projection_auto_epsilon,
        projection_pca_prefilter=args.projection_pca_prefilter,
        projection_auto_select_components=args.projection_auto_select_components,
        projection_selection_pca_grid=args.projection_selection_pca_grid,
        projection_selection_pls_grid=args.projection_selection_pls_grid,
        overwrite=args.overwrite,
    )
    cli_manifest = write_cli_manifest(args, Path(out["output_dir"]), cell_types)
    print(f"CLI manifest: {cli_manifest}")

    if args.skip_report:
        print("\nReport skipped because --skip-report was set.")
    else:
        from make_target_calibration_qc_report import make_target_calibration_qc_report

        report_out = make_target_calibration_qc_report(
            run_dir=out["output_dir"],
            output_dir=args.report_output_dir,
            top_n_drugs=args.top_n_drugs,
            rank_stat=args.rank_stat,
        )
        print(f"report summary: {report_out['summary']}")

    print("\n=== Target calibration QC complete ===")
    print(f"output:   {out['output_dir']}")
    print(f"scores:   {out['paths']['scores']}")
    print(f"manifest: {out['paths']['manifest']}")


if __name__ == "__main__":
    main()
