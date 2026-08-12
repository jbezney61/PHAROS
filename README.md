# PHAROS

PHAROS identifies combinatorial drug perturbations predicted to convert a
starting cell state into a target cell state.

## Package migration status

PHAROS is being migrated from its original research scripts to an installable
command-line application. The open-search workflow is now available through
the package, while the remaining scientific commands will be ported
incrementally.

```bash
uv tool install .
pharos --help
pharos --version
```

## Open search

`pharos open-search` runs the sequential drug-combination search previously
invoked with `python cell_converter.py`. Its defaults reproduce the validated
paper configuration. Only dataset, cell-state, model, and output paths are
required:

```bash
pharos open-search \
  --adata /path/to/embedded.h5ad \
  --start-cell "starting_state" \
  --target-cell "target_state" \
  --cell-col "cell_type" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --output-dir runs/example
```

For a conversion that fails separation QC, add high-sensitivity batch
selection. This automatically changes robust reranking from five batches to
three unless `--robust-n-samples` is supplied explicitly:

```bash
pharos open-search \
  --adata /path/to/embedded.h5ad \
  --start-cell "starting_state" \
  --target-cell "target_state" \
  --cell-col "cell_type" \
  --model-dir "$ST_RUN" \
  --output-dir runs/example_high_sensitivity \
  --batch-selection high-sensitivity
```

Run `pharos open-search --help` for the complete option reference. The legacy
`python cell_converter.py` entry point remains as a compatibility wrapper and
uses the same packaged implementation.

The pretrained STATE model and checkpoint are external runtime artifacts and
are not included in the PHAROS package.

## Admissibility checks

PHAROS evaluates target calibration, reference-manifold support, and
source-target separation before interpreting a search. The calibration and
manifold reference steps produce reusable reference outputs; query scoring and
separation are then run for each dataset.

Run target calibration against the observed perturbation reference dataset:

```bash
pharos admissibility calibrate \
  --adata /path/to/calibration_reference.h5ad \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --output-dir runs/target_calibration \
  --overwrite
```

Build the reusable embedding-manifold reference:

```bash
pharos admissibility manifold build-reference \
  --reference-h5ad /path/to/tahoe_reference.h5ad \
  --cell-line-metadata metadata/cell_line_metadata.csv \
  --output-dir runs/tahoe_manifold_reference \
  --overwrite
```

Score a query dataset against that reference:

```bash
pharos admissibility manifold score-query \
  --reference-dir runs/tahoe_manifold_reference \
  --query-h5ad /path/to/query.h5ad \
  --query-state-col cell_type \
  --output-dir runs/query_manifold_qc \
  --overwrite
```

Finally, test whether candidate source and target states are sufficiently
resolved in the query embedding:

```bash
pharos admissibility separation \
  --adata /path/to/query.h5ad \
  --cell-col cell_type \
  --output-dir runs/query_separation_qc
```

Use `pharos admissibility --help` and each subcommand's `--help` output for the
complete option reference. The corresponding scripts in
`target_calibration_QC/`, `embedding_manifold_QC/`, and
`umap_seperation_QC/` remain available as compatibility entry points.

## Hypothesis-driven analysis

Hypothesis-driven mode evaluates specified drug combinations against random
two-drug controls. A candidate may be an explicit pair, one fixed drug paired
with a requested mechanism, or two mechanism-of-action classes.

Evaluate one candidate pair and its mechanism-matched alternatives:

```bash
pharos hypothesis-driven pair \
  --adata /path/to/query.h5ad \
  --start-cell "starting_state" \
  --target-cell "target_state" \
  --cell-col "cell_type" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --drug-pair "['drug_a', 'drug_b']" \
  --moa-pairs "['mechanism_a', 'mechanism_b']" \
  --output-dir runs/hypothesis_pair \
  --overwrite
```

For a conversion that failed separation QC, add
`--batch-selection high-sensitivity`. This uses the same 1,000-candidate,
zero-overlap-penalty selection and three-batch default as open-search mode.

Evaluate a CSV or TSV panel containing `drug_a` and `drug_b` columns, with
optional `pair_id` and `pair_group` columns:

```bash
pharos hypothesis-driven panel \
  --adata /path/to/query.h5ad \
  --start-cell "starting_state" \
  --target-cell "target_state" \
  --cell-col "cell_type" \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_RUN/checkpoints/final.ckpt" \
  --pairs-file /path/to/drug_pairs.csv \
  --output-dir runs/hypothesis_panel \
  --overwrite
```

Collate pair analyses from multiple conversions and run the report-level
statistical comparisons:

```bash
pharos hypothesis-driven summarize \
  --run-dirs runs/conversion_a runs/conversion_b runs/conversion_c \
  --labels "Conversion A" "Conversion B" "Conversion C" \
  --output-dir runs/hypothesis_summary
```

The original options `--2drug-pair`, `--MOA-pairs`, and
`--approved-pairs-file` remain supported for command compatibility. The
scripts in `positive_control_2drug/` also remain available as compatibility
entry points.
