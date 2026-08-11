#!/usr/bin/env python
"""
positive_control_2drug_analysis.py

CLI wrapper for positive-control 2-drug ST-SE analysis.

Example
-------
python positive_control_2drug_analysis.py \
  --adata positive_controls/GSE206741_qc_mad_scrublet_log1p.pano_alve.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_Alvespimycin" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --output-dir runs/PC_CPA_pano_alve \
  --2drug-pair "['Panobinostat', 'crizotinib']" \
  --MOA-pairs "['HDAC inhibitor', 'Multi-TK inhibitor']"

The remaining defaults match the validated settings used in the PHAROS paper,
including 100 random pairs, five standard batches, and conversion-aligned
PCA--PLS-DA scoring. High-sensitivity batch selection defaults to three batches.
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path
from typing import List, Sequence


def parse_list_arg(
    values: Sequence[str] | str,
    *,
    name: str,
    expected_len: int | None = 2,
    expected_lens: Sequence[int] | None = None,
) -> List[str]:
    """
    Parse flexible list syntax from the shell.

    Supported forms:
      --2drug-pair Panobinostat crizotinib
      --2drug-pair Trametinib
      --2drug-pair "Panobinostat, crizotinib"
      --2drug-pair "['Panobinostat', 'crizotinib']"
      --2drug-pair "['Trametinib']"
      --MOA-pairs ['HDAC inhibitor', 'Multi-TK inhibitor']
    """
    if isinstance(values, str):
        tokens = [values]
    else:
        tokens = list(values)

    allowed_lens = tuple(expected_lens) if expected_lens is not None else (int(expected_len),)
    if len(tokens) in allowed_lens and not any(any(ch in t for ch in "[],") for t in tokens):
        parts = [t.strip().strip("\"'") for t in tokens]
    else:
        raw = " ".join(tokens).strip()
        parts = []
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                parts = [str(x).strip() for x in parsed]
        except Exception:
            parts = []

        if not parts:
            cleaned = raw.strip().strip("[]()")
            if "," in cleaned:
                parts = [p.strip().strip("\"'") for p in cleaned.split(",") if p.strip()]
            else:
                parts = [p.strip().strip("\"'") for p in cleaned.split() if p.strip()]

    parts = [p for p in parts if p]
    if len(parts) not in allowed_lens:
        expected = (
            f"one of {list(allowed_lens)} values"
            if len(allowed_lens) > 1
            else f"exactly {allowed_lens[0]} values"
        )
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected}; parsed {parts!r}. "
            "Quote values with spaces, or pass a Python/JSON-style list string."
        )
    return parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run positive-control 2-drug ST-SE conversion analysis: explicit pair, "
            "MOA-matched pairs, and random 2-drug controls."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = p.add_argument_group("Core inputs")
    required.add_argument("--adata", required=True, help="Input h5ad containing SE embeddings in adata.obsm[embed_key].")
    required.add_argument("--start-cell", required=True, help="Starting state/cell label, e.g. DMSO_DMSO.")
    required.add_argument("--target-cell", required=True, help="Target state/cell label, e.g. panobinostat_Alvespimycin.")
    required.add_argument("--model-dir", required=True, help="ST-SE training run directory.")
    required.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    required.add_argument(
        "--2drug-pair",
        "--two-drug-pair",
        dest="two_drug_pair",
        nargs="+",
        default=None,
        help=(
            "Optional explicit 2-drug pair; both drug orders are scored. "
            "Accepts one or two tokens, comma string, or list string. "
            "With one drug, that fixed drug is paired against all converter-available drugs "
            "matching the second --MOA-pairs term, and the best pair is used as the explicit pair. "
            "If omitted, blue explicit-pair outputs are skipped and additive/sequential plots use the best MOA pair."
        ),
    )
    required.add_argument(
        "--MOA-pairs",
        "--moa-pairs",
        dest="moa_pairs",
        nargs="+",
        required=True,
        help=(
            "moa-fine terms matched to --2drug-pair order. If the reverse explicit drug order wins, "
            "these terms are reversed before selecting top MOA pairs."
        ),
    )

    data = p.add_argument_group("Data loading")
    data.add_argument("--checkpoint", default=None, help="Path to ST-SE checkpoint. Defaults to model_dir/checkpoints/final.ckpt.")
    data.add_argument("--cell-col", default="cell_name", help="adata.obs column containing start/target labels.")
    data.add_argument("--embed-key", default="X_state", help="adata.obsm key containing SE embeddings.")
    data.add_argument("--start-sample", default="256", help="Number of start cells to sample, or 'all'.")
    data.add_argument("--target-sample", default="256", help="Number of target cells to sample, or 'all'.")
    data.add_argument("--seed", type=int, default=42, help="Random seed for cell sampling and random pairs.")
    data.add_argument("--batch-seed-offset", type=int, default=0, help="Batch seeds are seed + offset + batch_index.")
    data.add_argument("--no-replace-if-needed", action="store_true", help="Error if requested sample size exceeds available cells.")
    data.add_argument(
        "--batch-selection",
        choices=["standard", "high-sensitivity"],
        default="standard",
        help=(
            "How start/target batches are selected. 'standard' keeps random sampling. "
            "'high-sensitivity' screens local nearest-neighbor batches and keeps the most separated ones."
        ),
    )
    data.add_argument(
        "--batch-candidates",
        type=int,
        default=1000,
        help="Number of local candidate batch pairs screened when --batch-selection high-sensitivity.",
    )
    data.add_argument(
        "--batch-overlap-penalty",
        type=float,
        default=0.0,
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

    analysis = p.add_argument_group("Positive-control analysis")
    analysis.add_argument("--random-pairs", type=int, default=100, help="Number of random 2-drug controls.")
    analysis.add_argument(
        "--random-type",
        "--random_type",
        choices=["matched", "legacy"],
        default="matched",
        help=(
            "Random-control construction. 'matched' samples random base-drug pairs with no explicit/MOA/metadata "
            "drug exclusions, scores both orders and all concentration pairs on batch 0, and keeps the best row per "
            "random pair before evaluating those controls across all batches. 'legacy' keeps the prior direct "
            "ordered-label sampling with the existing filters, then also evaluates those controls across all batches."
        ),
    )
    analysis.add_argument(
        "--batch",
        "--batches",
        dest="n_batches",
        type=int,
        default=None,
        help=(
            "Number of sampled batches for baseline and selected pairs. "
            "Defaults to 5 with standard sampling and 3 with high-sensitivity sampling."
        ),
    )
    analysis.add_argument("--converter-chunk-size", type=int, default=16, help="Second-drug labels scored per converter/scorer chunk.")
    analysis.add_argument(
        "--include-explicit-in-moa",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include the explicit pair among MOA pair candidates when it matches the requested MOAs.",
    )
    analysis.add_argument(
        "--moa-match-mode",
        choices=["exact", "contains"],
        default="exact",
        help="How --MOA-pairs terms match metadata/drug_metadata.csv moa-fine values.",
    )
    analysis.add_argument("--max-moa-pairs", type=int, default=None, help="Optional random subset limiter for MOA pair smoke tests.")
    analysis.add_argument(
        "--allow-random-metadata-missing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow metadata-missing drugs in legacy random controls. Ignored for --random-type matched, "
            "which does not exclude metadata-missing drugs."
        ),
    )

    compute = p.add_argument_group("Compute")
    compute.add_argument("--device", default=None, help="Device, e.g. cuda:0 or cpu. Defaults to cuda:0 if available.")
    compute.add_argument("--max-set-len", type=int, default=256, help="Maximum cells per ST-SE forward pass.")
    compute.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True, help="Use autocast mixed precision on CUDA.")
    compute.add_argument("--amp-dtype", choices=["bfloat16", "float16"], default="bfloat16", help="Autocast dtype.")

    scoring = p.add_argument_group("Scoring")
    scoring.add_argument("--sinkhorn-metric", choices=["cosine", "sqeuclidean", "euclidean"], default="cosine")
    scoring.add_argument("--sinkhorn-epsilon", type=float, default=0.05, help="Entropic regularization for Sinkhorn OT.")
    scoring.add_argument("--sinkhorn-iters", type=int, default=100, help="Sinkhorn iterations.")
    scoring.add_argument("--no-normalize-embeddings", action="store_true", help="Disable L2 normalization before scoring.")

    projection = p.add_argument_group("Projection (optional PLS-DA / PCA scoring subspace)")
    projection.add_argument(
        "--projection-method",
        choices=["none", "pls_da", "pca_pls_da", "pca"],
        default="pca_pls_da",
        help="Linear dimensionality reduction fit on start vs target and applied only at scoring time.",
    )
    projection.add_argument("--projection-components", type=int, default=128, help="Number of latent dimensions K.")
    projection.add_argument(
        "--projection-whiten",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whiten projected components to unit variance.",
    )
    projection.add_argument("--projection-fit-cap", type=int, default=4000, help="Max cells per class used to fit the projection.")
    projection.add_argument("--projection-pca-prefilter", type=int, default=256, help="PCA components for pca_pls_da prefilter.")
    projection.add_argument(
        "--projection-auto-select-components",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        default="64,96,128,192",
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
    projection.add_argument(
        "--projection-target-split",
        choices=["none", "holdout", "auto"],
        default="auto",
        help="'holdout' splits start and target cells into fit/eval halves so PLS never sees the scored cells; "
        "'auto' picks holdout for large datasets and none for small ones (see --projection-small-dataset-threshold).",
    )
    projection.add_argument("--projection-split-frac", type=float, default=0.5, help="Fraction of cells used to fit the projection in holdout mode.")
    projection.add_argument(
        "--projection-small-dataset-threshold",
        type=int,
        default=512,
        help="In 'auto' split, datasets with fewer cells per class than this fit the projection on all cells (no holdout).",
    )
    projection.add_argument(
        "--projection-auto-epsilon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set Sinkhorn epsilon from the median pairwise cost in projected space.",
    )

    metadata = p.add_argument_group("Metadata")
    metadata.add_argument("--metadata-dir", default="metadata", help="Directory containing drug_metadata.csv.")
    metadata.add_argument("--drug-metadata", default=None, help="Optional explicit path to drug_metadata.csv.")

    out = p.add_argument_group("Output and report")
    out.add_argument("--skip-report", action="store_true", help="Run scoring only; skip report/figure generation.")
    out.add_argument("--report-output-dir", default=None, help="Report output directory. Default: <output-dir>/report.")
    out.add_argument("--report-top-n-moa", type=int, default=3, help="Number of top MOA pairs shown in report figures.")
    out.add_argument(
        "--save-trajectory-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save explicit --2drug-pair WT, drug1, drug1+drug2, and target embeddings "
            "for the trajectory report."
        ),
    )
    out.add_argument(
        "--trajectory-report-output-dir",
        default=None,
        help="Trajectory report output directory. Defaults depend on --trajectory-embedding-space.",
    )
    out.add_argument(
        "--trajectory-embedding-space",
        choices=["full", "projection", "both"],
        default="projection",
        help="Embedding space for explicit trajectory report figures/metrics.",
    )
    out.add_argument(
        "--trajectory-projection-cache",
        default=None,
        help="Optional fitted projection .npz for projected trajectory figures. Defaults to the run projection cache.",
    )
    out.add_argument(
        "--trajectory-max-cells-per-group",
        type=int,
        default=1200,
        help="Maximum cells per visible group for trajectory scatter/UMAP plots.",
    )
    out.add_argument("--errorbar", choices=["std", "sem"], default="std", help="Error bars for the bar plot.")
    out.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite a non-empty output directory.")

    args = p.parse_args()
    if args.n_batches is None:
        args.n_batches = 3 if args.batch_selection == "high-sensitivity" else 5
    return args


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output_dir}\n"
                "Use --overwrite or choose a new --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_cli_manifest(args: argparse.Namespace, output_dir: Path, two_drug_pair: Sequence[str], moa_pairs: Sequence[str]) -> Path:
    path = output_dir / "positive_control_cli_args.json"
    payload = vars(args).copy()
    payload["two_drug_pair_parsed"] = list(two_drug_pair)
    payload["moa_pairs_parsed"] = list(moa_pairs)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def main() -> None:
    args = parse_args()
    try:
        two_drug_pair = (
            parse_list_arg(args.two_drug_pair, name="--2drug-pair", expected_lens=(1, 2))
            if args.two_drug_pair
            else []
        )
        moa_pairs = parse_list_arg(args.moa_pairs, name="--MOA-pairs")
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if (
        not args.skip_report
        and args.save_trajectory_embeddings
        and args.trajectory_embedding_space in {"projection", "both"}
        and args.projection_method == "none"
        and not args.trajectory_projection_cache
    ):
        raise SystemExit(
            "error: --trajectory-embedding-space projection/both requires either "
            "--projection-method != none or --trajectory-projection-cache."
        )

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    manifest = write_cli_manifest(args, output_dir, two_drug_pair, moa_pairs)
    print(f"CLI manifest: {manifest}")

    from positive_control_2drug import run_positive_control_2drug_analysis

    analysis_out = run_positive_control_2drug_analysis(
        adata=args.adata,
        start_cell=args.start_cell,
        target_cell=args.target_cell,
        cell_col=args.cell_col,
        embed_key=args.embed_key,
        model_dir=args.model_dir,
        checkpoint=args.checkpoint,
        output_dir=output_dir,
        two_drug_pair=two_drug_pair,
        moa_pairs=moa_pairs,
        random_pairs=args.random_pairs,
        random_type=args.random_type,
        n_batches=args.n_batches,
        start_sample=args.start_sample,
        target_sample=args.target_sample,
        batch_selection=args.batch_selection,
        batch_candidates=args.batch_candidates,
        batch_overlap_penalty=args.batch_overlap_penalty,
        batch_selection_score_chunk_size=args.batch_selection_score_chunk_size,
        seed=args.seed,
        batch_seed_offset=args.batch_seed_offset,
        replace_if_needed=not args.no_replace_if_needed,
        converter_chunk_size=args.converter_chunk_size,
        device=args.device,
        max_set_len=args.max_set_len,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        normalize_embeddings=not args.no_normalize_embeddings,
        sinkhorn_metric=args.sinkhorn_metric,
        sinkhorn_epsilon=args.sinkhorn_epsilon,
        sinkhorn_iters=args.sinkhorn_iters,
        metadata_dir=args.metadata_dir,
        drug_metadata=args.drug_metadata,
        include_explicit_in_moa=args.include_explicit_in_moa,
        moa_match_mode=args.moa_match_mode,
        max_moa_pairs=args.max_moa_pairs,
        allow_random_metadata_missing=args.allow_random_metadata_missing,
        projection_method=args.projection_method,
        projection_components=args.projection_components,
        projection_whiten=args.projection_whiten,
        projection_fit_cap=args.projection_fit_cap,
        projection_pca_prefilter=args.projection_pca_prefilter,
        projection_target_split=args.projection_target_split,
        projection_split_frac=args.projection_split_frac,
        projection_small_dataset_threshold=args.projection_small_dataset_threshold,
        projection_auto_epsilon=args.projection_auto_epsilon,
        projection_auto_select_components=args.projection_auto_select_components,
        projection_selection_pca_grid=args.projection_selection_pca_grid,
        projection_selection_pls_grid=args.projection_selection_pls_grid,
        projection_selection_fit_frac=args.projection_selection_fit_frac,
        projection_selection_repeats=args.projection_selection_repeats,
        projection_selection_small_cell_threshold=args.projection_selection_small_cell_threshold,
        projection_selection_fallback_pca=args.projection_selection_fallback_pca,
        projection_selection_fallback_pls=args.projection_selection_fallback_pls,
        projection_selection_rule=args.projection_selection_rule,
        save_trajectory_embeddings=args.save_trajectory_embeddings,
    )

    if args.skip_report:
        print("\nReport skipped because --skip-report was set.")
    else:
        from make_positive_control_report import make_positive_control_report

        report_out = make_positive_control_report(
            run_dir=output_dir,
            output_dir=args.report_output_dir,
            top_n_moa=args.report_top_n_moa,
            errorbar=args.errorbar,
        )
        print(f"report summary: {report_out['summary']}")

        trajectory_cache = analysis_out["paths"].get("explicit_pair_trajectory_embeddings")
        if args.save_trajectory_embeddings and trajectory_cache:
            from make_explicit_trajectory_report import make_explicit_trajectory_report

            trajectory_report_out = make_explicit_trajectory_report(
                run_dir=output_dir,
                output_dir=args.trajectory_report_output_dir,
                max_cells_per_group=args.trajectory_max_cells_per_group,
                seed=args.seed,
                embedding_space=args.trajectory_embedding_space,
                projection_cache=args.trajectory_projection_cache or analysis_out["paths"].get("projection_cache"),
            )
            print(f"trajectory report summary: {trajectory_report_out['summary']}")
        elif args.save_trajectory_embeddings:
            print("\nTrajectory report skipped because no explicit --2drug-pair trajectory cache was saved.")
        else:
            print("\nTrajectory report skipped because --no-save-trajectory-embeddings was set.")

    print("\n=== Workflow complete ===")
    print(f"output:             {analysis_out['output_dir']}")
    print(f"evaluation results: {analysis_out['paths']['evaluation_results']}")
    print(f"selected pairs:     {analysis_out['paths']['selected_pairs']}")


if __name__ == "__main__":
    main()
