#!/usr/bin/env python
"""
PHAROS separation admissibility CLI.

CLI for pre-search cell-line pair screening (UMAP, KNN purity QC, energy distances).

Example:
    export CUDA_VISIBLE_DEVICES=0

    python screen_cell_line_pairs.py \
      --adata WT_256_per_cell_name.SE600M.h5ad \
      --cell-col cell_name \
      --embed-key X_state \
      --cells-per-line 256 \
      --output-dir runs/pair_screening_WT256
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pharos admissibility separation",
        description="Screen start/target cell-line pairs before PHAROS search.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adata", required=True, help="h5ad with SE embeddings in obsm")
    p.add_argument("--cell-col", default="cell_name", help="obs column with cell-line labels")
    p.add_argument("--embed-key", default="X_state", help="obsm key with SE embeddings")
    p.add_argument("--output-dir", required=True, help="Output directory for tables/figures/cache")

    p.add_argument(
        "--cells-per-line",
        type=int,
        default=None,
        help="Sample exactly this many cells per retained cell line. Lines with fewer cells are skipped.",
    )
    p.add_argument(
        "--max-cells-per-line",
        type=int,
        default=None,
        help="Deprecated alias for --cells-per-line.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--knn-k", type=int, default=30)
    p.add_argument("--min-mean-knn-purity", type=float, default=0.8)
    p.add_argument("--min-cells-per-line", type=int, default=2, help="Skip lines with fewer cells")

    p.add_argument("--energy-batch-size", type=int, default=32, help="Batch size for pairwise energy distance computation")
    p.add_argument("--device", default=None, help="cuda:0 or cpu; auto-detect if omitted")

    p.add_argument("--umap-n-neighbors", type=int, default=15)
    p.add_argument("--umap-min-dist", type=float, default=0.3)
    p.add_argument("--top-n-summary", type=int, default=20, help="Top pairs listed in summary.md")
    return p.parse_args(argv)


def resolve_cells_per_line(args: argparse.Namespace) -> int:
    """Resolve the preferred option and its deprecated alias."""

    cells_per_line = args.cells_per_line
    if cells_per_line is None:
        cells_per_line = 256 if args.max_cells_per_line is None else args.max_cells_per_line
    elif args.max_cells_per_line is not None and args.max_cells_per_line != cells_per_line:
        raise ValueError("--cells-per-line and --max-cells-per-line were both set to different values")
    if cells_per_line <= 0:
        raise ValueError("--cells-per-line must be positive")
    return cells_per_line


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cells_per_line = resolve_cells_per_line(args)
    import torch

    from .separation import run_pair_screening

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out = run_pair_screening(
        h5ad_path=args.adata,
        output_dir=args.output_dir,
        cell_col=args.cell_col,
        embed_key=args.embed_key,
        max_cells_per_line=cells_per_line,
        seed=args.seed,
        knn_k=args.knn_k,
        min_mean_knn_purity=args.min_mean_knn_purity,
        energy_batch_size=args.energy_batch_size,
        device=device,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cells_per_line=args.min_cells_per_line,
        top_n_summary=args.top_n_summary,
    )

    ranked = out["ranked_pairs"]
    qc = out["cell_line_qc"]

    print("\n=== Pair screening complete ===")
    print(f"Output directory: {Path(args.output_dir).resolve()}")
    print(f"Cells sampled per retained line: {cells_per_line}")
    print(f"Cell lines sampled:      {len(qc)}")
    print(f"Cell lines skipped count: {len(out['cell_line_count_filter'])}")
    print(f"Cell lines passing QC:   {int(qc['pass_qc'].sum())}")
    print(f"Directed pairs ranked:   {len(ranked)}")

    if len(ranked):
        print("\nTop 5 pairs (lowest energy distance):")
        for _, row in ranked.head(5).iterrows():
            print(
                f"  {int(row['rank'])}. {row['start_cell_line']} -> {row['target_cell_line']}: "
                f"energy={row['distance_energy']:.6g}"
            )

    print("\nKey outputs:")
    print(f"  {args.output_dir}/tables/ranked_pairs.tsv")
    print(f"  {args.output_dir}/summary.md")
    print(f"  {args.output_dir}/figures/01_umap_by_cell_line.png")


if __name__ == "__main__":
    main()
