"""CLI for per-run target-pair recovery vs Model B MC null."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pharos evaluate pair-recovery",
        description=(
            "For each search run, count assigned-pair abundance (n_A, n_B) and exact-pair "
            "recovery in the top beam, and compare to a Model B Monte Carlo null."
        )
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        type=Path,
        help=(
            "Search output directories or result files. Each entry may be a run directory "
            "containing results.tsv, a top-level run directory containing search/results.tsv, "
            "or a direct TSV path."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="Dataset labels in the same order as --run-dirs. Must match the pair-table run key.",
    )
    parser.add_argument(
        "--target-pairs",
        required=True,
        type=Path,
        help=(
            "TSV/CSV with one row per run. Must include a run-key column (default: label) "
            "plus drug_a and drug_b."
        ),
    )
    parser.add_argument(
        "--run-key-col",
        default="label",
        help="Column in --target-pairs that matches --labels (default: label).",
    )
    parser.add_argument("--drug-a-col", default="drug_a", help="Column for drug A.")
    parser.add_argument("--drug-b-col", default="drug_b", help="Column for drug B.")
    parser.add_argument(
        "--pair-id-col",
        default="pair_id",
        help="Optional display-name column for pairs. If absent, drug_a + drug_b is used.",
    )
    parser.add_argument(
        "--depth",
        choices=["1", "2", "both"],
        required=True,
        help="Depths included when counting. Use 2 for two-drug paths only.",
    )
    parser.add_argument(
        "--rank-threshold",
        type=int,
        required=True,
        help="Retain rows with rank <= this threshold (beam width for observed and null).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of Model B MC null replicates per run.",
    )
    parser.add_argument(
        "--num-drugs",
        type=int,
        default=379,
        help="Number of base drugs in the Model B search space.",
    )
    parser.add_argument(
        "--concentrations-per-drug",
        type=int,
        default=3,
        help="Number of concentration labels per base drug.",
    )
    parser.add_argument("--seed", type=int, default=1, help="RNG seed for MC null.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where tables, plots, and summary.md will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        from .pair_recovery import run_analysis, save_analysis_tables, write_summary_markdown
        from .pair_recovery_plotting import make_report_plots
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing pair-recovery dependency. This CLI requires numpy, pandas, "
            "and matplotlib in the Python environment used for analysis."
        ) from exc

    analysis = run_analysis(
        run_dirs=args.run_dirs,
        labels=args.labels,
        target_pairs_path=args.target_pairs,
        depth_mode=args.depth,
        rank_threshold=args.rank_threshold,
        permutations=args.permutations,
        output_dir=args.output_dir,
        num_drugs=args.num_drugs,
        concentrations_per_drug=args.concentrations_per_drug,
        seed=args.seed,
        run_key_col=args.run_key_col,
        drug_a_col=args.drug_a_col,
        drug_b_col=args.drug_b_col,
        pair_id_col=args.pair_id_col,
    )
    table_paths = save_analysis_tables(analysis)
    figure_paths = make_report_plots(analysis)
    summary_path = write_summary_markdown(analysis, table_paths, figure_paths)

    print(f"Wrote target-pair recovery report to: {args.output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Table: {table_paths['per_run_results']}")
    print()
    display = analysis["per_run_results_display"]
    print(
        f"{'target_pair':<36} {'#A':<14} {'#B':<14} {'exact':<14} {'exact_ot_rank':<13}"
    )
    for _, row in display.iterrows():
        print(
            f"{row['target_pair']:<36} "
            f"{row['n_A'] + ' ' + row['n_A_p']:<14} "
            f"{row['n_B'] + ' ' + row['n_B_p']:<14} "
            f"{row['exact'] + ' ' + row['exact_p']:<14} "
            f"{row['best_exact_rank']:<10}"
        )
    if analysis["notes"]:
        print("\nNotes:")
        for note in analysis["notes"]:
            print(f"  - {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
