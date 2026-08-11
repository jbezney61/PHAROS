#!/usr/bin/env python
"""
CLI for explicit FDA-approved 2-drug pair panel analysis.

This is a thin wrapper around run_positive_control_2drug_analysis. It keeps the
standard positive-control sampling, scoring, random-pair controls, and output
tables, but replaces the single --2drug-pair / MOA search with a file of
explicit 2-drug pairs. Each pair is evaluated in both orders and retained as
one selected explicit_pair row.

Example
-------
python positive_control_2drug/positive_control_2drug_panel_analysis.py \\
  --adata my_conversion.SE600M.h5ad \\
  --start-cell metastatic_patient_1 \\
  --target-cell matched_primary_1 \\
  --cell-col Sample \\
  --model-dir "$ST_RUN" \\
  --approved-pairs-file approved_breast_cancer_pairs.tsv \\
  --output-dir runs/patient_1_fda_pair_panel \\
  --random-pairs 1000 \\
  --batch 5

Pair file columns
-----------------
Required:
    drug_a, drug_b

Optional:
    pair_id, pair_group
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _path in [SCRIPT_DIR, REPO_ROOT]:
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd


def _read_table_auto(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Approved-pairs file not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def load_explicit_pairs_file(
    path: str | Path,
    *,
    pair_id_col: str = "pair_id",
    first_drug_col: str = "drug_a",
    second_drug_col: str = "drug_b",
    pair_group_col: str | None = "pair_group",
) -> List[Dict[str, str]]:
    df = _read_table_auto(path)
    missing = [c for c in [first_drug_col, second_drug_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required pair columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    records: List[Dict[str, str]] = []
    seen_pair_ids: set[str] = set()
    for row_index, row in df.iterrows():
        first = str(row[first_drug_col]).strip()
        second = str(row[second_drug_col]).strip()
        if not first or first.lower() in {"nan", "none", "null"}:
            raise ValueError(f"Missing first drug in row {row_index + 1} of {path}")
        if not second or second.lower() in {"nan", "none", "null"}:
            raise ValueError(f"Missing second drug in row {row_index + 1} of {path}")

        if pair_id_col in df.columns and pd.notna(row[pair_id_col]) and str(row[pair_id_col]).strip():
            pair_id = str(row[pair_id_col]).strip()
        else:
            pair_id = f"{first} + {second}"
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id {pair_id!r} in {path}")
        seen_pair_ids.add(pair_id)

        record = {"pair_id": pair_id, "drug_a": first, "drug_b": second}
        if pair_group_col and pair_group_col in df.columns:
            if pd.isna(row[pair_group_col]) or not str(row[pair_group_col]).strip():
                raise ValueError(f"Missing pair group in row {row_index + 1} of {path}")
            record["pair_group"] = str(row[pair_group_col]).strip()
        records.append(record)

    if not records:
        raise ValueError(f"No approved pairs found in {path}")
    return records


def parse_list_arg(values: Sequence[str] | str, *, name: str, expected_len: int = 2) -> List[str]:
    if isinstance(values, str):
        tokens = [values]
    else:
        tokens = list(values)
    if len(tokens) == expected_len:
        return [str(x).strip().strip("\"'") for x in tokens]
    raw = " ".join(tokens).strip().strip("[]()")
    parts = [p.strip().strip("\"'") for p in raw.split(",") if p.strip()]
    if len(parts) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} must contain exactly {expected_len} values; parsed {parts!r}")
    return parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run positive-control analysis for a panel of explicit FDA-approved "
            "2-drug pairs against random 2-drug controls."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = p.add_argument_group("Core inputs")
    required.add_argument("--adata", required=True, help="Input h5ad containing SE embeddings.")
    required.add_argument("--start-cell", required=True, help="Starting state/cell label.")
    required.add_argument("--target-cell", required=True, help="Target state/cell label.")
    required.add_argument("--model-dir", required=True, help="ST-SE training run directory.")
    required.add_argument("--output-dir", required=True, help="Directory where outputs will be written.")
    required.add_argument("--approved-pairs-file", required=True, help="TSV/CSV with FDA-approved pairs.")

    pairs = p.add_argument_group("Approved pair file")
    pairs.add_argument("--pair-id-col", default="pair_id", help="Optional pair ID column in --approved-pairs-file.")
    pairs.add_argument("--first-drug-col", default="drug_a", help="First drug column in --approved-pairs-file.")
    pairs.add_argument("--second-drug-col", default="drug_b", help="Second drug column in --approved-pairs-file.")
    pairs.add_argument(
        "--pair-group-col",
        default="pair_group",
        help="Optional cohort/group column in --approved-pairs-file, e.g. FDA_approved or failed_trial.",
    )
    pairs.add_argument("--fda-group-value", default="FDA_approved", help="pair_group value used for the FDA-approved cohort report.")
    pairs.add_argument("--failed-group-value", default="failed_trial", help="pair_group value used for the failed-trial cohort report.")

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
        help="How start/target batches are selected.",
    )
    data.add_argument("--batch-candidates", type=int, default=300, help="Candidates screened for high-sensitivity batches.")
    data.add_argument("--batch-overlap-penalty", type=float, default=0.02, help="Overlap penalty for high-sensitivity batches.")
    data.add_argument("--batch-selection-score-chunk-size", type=int, default=8, help="Candidate batches scored per chunk.")

    analysis = p.add_argument_group("Panel analysis")
    analysis.add_argument("--random-pairs", type=int, default=1000, help="Number of random 2-drug controls.")
    analysis.add_argument(
        "--random-type",
        "--random_type",
        choices=["matched", "legacy"],
        default="matched",
        help=(
            "Random-control construction. 'matched' samples random base-drug pairs with no explicit/MOA/metadata "
            "drug exclusions, scores both orders and all concentration pairs on batch 0, and keeps the best row per "
            "random pair. 'legacy' keeps the prior direct ordered-label sampling with the existing filters."
        ),
    )
    analysis.add_argument("--batch", "--batches", dest="n_batches", type=int, default=5, help="Number of sampled batches.")
    analysis.add_argument("--converter-chunk-size", type=int, default=16, help="Second-drug labels scored per converter/scorer chunk.")
    analysis.add_argument(
        "--MOA-pairs",
        "--moa-pairs",
        dest="moa_pairs",
        nargs="+",
        default=[],
        help="Optional MOA terms if --evaluate-moa-pairs is enabled.",
    )
    analysis.add_argument(
        "--evaluate-moa-pairs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the old MOA-matched pair search. Off by default for FDA panel runs.",
    )
    analysis.add_argument("--max-moa-pairs", type=int, default=None, help="Optional MOA pair subset limiter.")
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

    projection = p.add_argument_group("Projection")
    projection.add_argument("--projection-method", choices=["none", "pls_da", "pca_pls_da", "pca"], default="none")
    projection.add_argument("--projection-components", type=int, default=128)
    projection.add_argument("--projection-whiten", action=argparse.BooleanOptionalAction, default=False)
    projection.add_argument("--projection-fit-cap", type=int, default=4000)
    projection.add_argument("--projection-pca-prefilter", type=int, default=256)
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
    projection.add_argument("--projection-target-split", choices=["none", "holdout", "auto"], default="auto")
    projection.add_argument("--projection-split-frac", type=float, default=0.5)
    projection.add_argument("--projection-small-dataset-threshold", type=int, default=512)
    projection.add_argument("--projection-auto-epsilon", action=argparse.BooleanOptionalAction, default=True)

    metadata = p.add_argument_group("Metadata")
    metadata.add_argument("--metadata-dir", default="metadata", help="Directory containing drug_metadata.csv.")
    metadata.add_argument("--drug-metadata", default=None, help="Optional explicit path to drug_metadata.csv.")

    out = p.add_argument_group("Output and report")
    out.add_argument("--skip-report", action="store_true", help="Run scoring only; skip panel report generation.")
    out.add_argument("--report-output-dir", default=None, help="Report output directory. Default: <output-dir>/panel_report.")
    out.add_argument("--errorbar", choices=["std", "sem"], default="std", help="Error bars for report mean-gain plot.")
    out.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite a non-empty output directory.")

    return p.parse_args()


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory exists and is not empty: {output_dir}\n"
                "Use --overwrite or choose a new --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def write_cli_manifest(args: argparse.Namespace, output_dir: Path, explicit_pairs: Sequence[Dict[str, str]]) -> Path:
    path = output_dir / "positive_control_panel_cli_args.json"
    payload = vars(args).copy()
    payload["explicit_pairs_parsed"] = list(explicit_pairs)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def main() -> None:
    args = parse_args()
    explicit_pairs = load_explicit_pairs_file(
        args.approved_pairs_file,
        pair_id_col=args.pair_id_col,
        first_drug_col=args.first_drug_col,
        second_drug_col=args.second_drug_col,
        pair_group_col=args.pair_group_col,
    )
    if args.evaluate_moa_pairs:
        try:
            moa_pairs = parse_list_arg(args.moa_pairs, name="--MOA-pairs")
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(f"error: {exc}") from exc
    else:
        moa_pairs = []

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, overwrite=args.overwrite)
    manifest = write_cli_manifest(args, output_dir, explicit_pairs)
    print(f"CLI manifest: {manifest}")
    print(f"approved explicit pairs: {len(explicit_pairs)}")
    if any("pair_group" in pair for pair in explicit_pairs):
        pair_groups = sorted({str(pair["pair_group"]) for pair in explicit_pairs if "pair_group" in pair})
        print(f"pair groups:        {pair_groups}")

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
        explicit_drug_pairs=explicit_pairs,
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
        max_moa_pairs=args.max_moa_pairs,
        evaluate_moa_pairs=args.evaluate_moa_pairs,
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
        save_trajectory_embeddings=False,
        evaluate_additive_interaction=False,
    )

    if args.skip_report:
        print("\nPanel report skipped because --skip-report was set.")
    else:
        from make_positive_control_panel_report import make_positive_control_panel_report

        report_out = make_positive_control_panel_report(
            run_dir=output_dir,
            output_dir=args.report_output_dir,
            errorbar=args.errorbar,
            pair_group_col="pair_group",
            fda_group_value=args.fda_group_value,
            failed_group_value=args.failed_group_value,
        )
        print(f"panel report summary: {report_out['summary']}")

    print("\n=== Explicit pair panel workflow complete ===")
    print(f"output:             {analysis_out['output_dir']}")
    print(f"evaluation results: {analysis_out['paths']['evaluation_results']}")
    print(f"selected pairs:     {analysis_out['paths']['selected_pairs']}")


if __name__ == "__main__":
    main()
