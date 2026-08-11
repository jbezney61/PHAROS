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
