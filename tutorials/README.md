# Tutorial: a single positive-control conversion

This tutorial runs the main PHAROS workflows on one known positive-control
conversion. The input is the Zenodo dataset
[`GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad.gz`](https://zenodo.org/records/21925263/files/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad.gz?download=1),
available from the [PHAROS Zenodo record](https://zenodo.org/records/21925263).
It contains one cell line in two observed states:

- start: DMSO control (`DMSO_DMSO`)
- target: combined panobinostat and crizotinib treatment
  (`panobinostat_crizotinib`)

The state labels are in `adata.obs["cell_type"]`. The dataset has already been
QC-filtered, normalized, log-transformed, and embedded with SE-600M; its
embeddings are in `adata.obsm["X_state"]`. Do not normalize or embed it again.

The commands below are written from the root of a PHAROS clone and use the
same model environment variables as the [main README](../README.md). Complete
its installation and model-download steps first, activate the `PHAROS` Conda
environment, and make sure `ST_RUN` and `ST_CKPT` are set.

## 1. Download the example and reference manifold

Create local data and output directories:

```bash
mkdir -p data runs
```

Download and decompress the positive-control dataset:

```bash
curl --location \
  "https://zenodo.org/records/21925263/files/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad.gz?download=1" \
  --output data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad.gz

gzip --decompress --keep \
  data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad.gz
```

PHAROS query-manifold scoring also needs the reusable Tahoe-100M reference
vector database. Download the already-built reference from the same Zenodo
record and extract it:

```bash
curl --location \
  "https://zenodo.org/records/21925263/files/tahoe100m_stse_manifold_reference.tar.gz?download=1" \
  --output data/tahoe100m_stse_manifold_reference.tar.gz

tar --extract --gzip --file data/tahoe100m_stse_manifold_reference.tar.gz \
  --directory data
```

The remaining examples assume these paths:

```text
data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad
data/tahoe100m_stse_manifold_reference/
```

If the archive creates one extra enclosing directory, pass the directory that
directly contains the reference manifest and parquet files to
`--reference-dir`.

An optional input check confirms that the embedded matrix and both exact,
case-sensitive state labels are present:

```bash
python - <<'PY'
import anndata as ad

path = "data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad"
adata = ad.read_h5ad(path, backed="r")
assert "X_state" in adata.obsm, "Missing adata.obsm['X_state']"
assert "cell_type" in adata.obs, "Missing adata.obs['cell_type']"
counts = adata.obs["cell_type"].value_counts()
for label in ("DMSO_DMSO", "panobinostat_crizotinib"):
    assert label in counts.index, f"Missing state: {label}"
print("X_state shape:", adata.obsm["X_state"].shape)
print(counts.loc[["DMSO_DMSO", "panobinostat_crizotinib"]])
PY
```

## 2. Run admissibility checks

These are the first PHAROS commands to run for a new conversion. Target
calibration and reference-manifold construction have already been performed;
this tutorial starts with `pharos admissibility manifold score-query`, as a
user should when bringing an SE-embedded dataset to the supplied reference.

### 2.1 Score the query against the Tahoe manifold

```bash
pharos admissibility manifold score-query \
  --reference-dir data/tahoe100m_stse_manifold_reference \
  --query-h5ad data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --query-state-col cell_type \
  --embed-key X_state \
  --output-dir runs/pano_criz_manifold_qc \
  --save-query-neighbors \
  --overwrite
```

Review `runs/pano_criz_manifold_qc/report/` and the state-level tables. An
out-of-distribution result does not prevent hypothesis generation, but it is a
reason to interpret downstream rankings more cautiously. Saving query
neighbors enables the report's local query/reference visualization.

### 2.2 Check start-target separation

Use 1,300 cells from each state, as specified for this positive control:

```bash
pharos admissibility separation \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --cell-col cell_type \
  --embed-key X_state \
  --cells-per-line 1300 \
  --output-dir runs/pano_criz_separation_qc
```

Inspect `runs/pano_criz_separation_qc/summary.md`, its KNN-purity QC, and its
pairwise energy-distance output before moving on. `--cells-per-line` requires
at least 1,300 cells in each retained state; if applying this template to a
smaller dataset, choose a value no larger than the smaller state.

If separation fails but the conversion is still scientifically important,
repeat downstream analyses with the high-sensitivity batch-selection examples
below and report those runs as sensitivity analyses.

## 3. Hypothesis-driven analysis

Hypothesis-driven mode compares prespecified drugs or mechanisms with 100
random two-drug controls by default. The examples below keep the requested
five batches and the exact STATE/Tahoe drug and `moa-fine` vocabulary. Drug
name capitalization is independent of the lowercase text in the target-state
label and must match `metadata/drug_metadata.csv`.

### 3.1 Test the known two-drug positive control

```bash
pharos hypothesis-driven pair \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --2drug-pair "['crizotinib', 'Panobinostat']" \
  --batch 5 \
  --MOA-pairs "['Multi-TK inhibitor', 'HDAC inhibitor']" \
  --output-dir runs/pano_criz_hypothesis_explicit_pair \
  --overwrite
```

PHAROS evaluates both sequential orders, so the list order does not force the
winning trajectory. It searches available concentration combinations,
compares the explicit pair with mechanism-matched pairs and matched random
controls, and generates the pair and trajectory reports automatically.

### 3.2 Fix crizotinib and search for an HDAC-inhibitor partner

Passing one drug fixes it as the first mechanism's member. The second
mechanism defines the eligible partner set:

```bash
pharos hypothesis-driven pair \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --2drug-pair "['crizotinib']" \
  --batch 5 \
  --MOA-pairs "['Multi-TK inhibitor', 'HDAC inhibitor']" \
  --output-dir runs/pano_criz_hypothesis_fixed_crizotinib \
  --overwrite
```

This tests whether PHAROS recovers Panobinostat, or another HDAC inhibitor, as
a strong partner without supplying the second drug's identity.

### 3.3 Test only the two mechanism classes

Omit `--2drug-pair` to ask whether Multi-TK-inhibitor/HDAC-inhibitor
combinations outperform random controls without naming either drug:

```bash
pharos hypothesis-driven pair \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --batch 5 \
  --MOA-pairs "['Multi-TK inhibitor', 'HDAC inhibitor']" \
  --output-dir runs/pano_criz_hypothesis_moa_only \
  --overwrite
```

Because this run has no explicit pair, explicit-pair trajectory outputs are
omitted; the best mechanism-matched pair is used for additive and sequential
comparisons.

### 3.4 Run the explicit-pair sensitivity analysis

Use this after weak separation QC, or as an additional robustness check:

```bash
pharos hypothesis-driven pair \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --2drug-pair "['crizotinib', 'Panobinostat']" \
  --batch 5 \
  --MOA-pairs "['Multi-TK inhibitor', 'HDAC inhibitor']" \
  --batch-selection high-sensitivity \
  --output-dir runs/pano_criz_hypothesis_explicit_pair_high_sensitivity \
  --overwrite
```

High-sensitivity selection screens 1,000 candidate local start/target batch
pairs with zero overlap penalty by default. Here `--batch 5` intentionally
overrides its usual three-batch default so that the requested five batches are
retained.

### 3.5 Optional panel mode on the same conversion

Panel mode is useful when this same start-to-target conversion has several
prespecified candidate pairs. A one-row file can smoke-test the known positive
control:

```bash
printf 'pair_id\tdrug_a\tdrug_b\tpair_group\npositive_control\tcrizotinib\tPanobinostat\tpositive_control\n' \
  > data/pano_criz_pairs.tsv

pharos hypothesis-driven panel \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --pairs-file data/pano_criz_pairs.tsv \
  --batch 5 \
  --output-dir runs/pano_criz_hypothesis_panel \
  --overwrite
```

The one-row panel produces an individual pair-versus-random comparison, but it
cannot support a meaningful between-cohort test. For a cohort comparison, add
multiple independently selected pairs to each `pair_group` and use the
`FDA_approved` and `failed_trial` labels (or set `--fda-group-value` and
`--failed-group-value`). The panel report then applies its one-sided
Mann-Whitney U test and Benjamini-Hochberg adjustment.

### 3.6 Read the hypothesis-driven statistics

Each `pair` command generates its report automatically. Use the report and
tables together to check:

- the raw start-to-target baseline versus the sequential conversion;
- both drug orders and the selected concentration labels;
- the explicit pair versus the MOA-matched and random-pair distributions;
- Sinkhorn and energy-distance results across all five batches;
- adjusted P-values rather than uncorrected P-values; and
- additive-versus-sequential interaction summaries and trajectory geometry.

These are prioritization and robustness tests. They do not establish drug
synergy, efficacy, or causality without experimental validation.

## 4. Open search

Open search does not receive the positive-control drugs. It searches the STATE
perturbation vocabulary for depth-two sequential combinations that move
`DMSO_DMSO` toward `panobinostat_crizotinib`.

### 4.1 Run the unbiased search

```bash
pharos open-search \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --max-depth 2 \
  --robust-n-samples 5 \
  --output-dir runs/pano_criz_open_search \
  --overwrite
```

`--robust-n-samples 5` makes the requested five-batch robustness setting
explicit for open search. The retained search paths are written under
`runs/pano_criz_open_search/search/`, and the general report is generated
automatically.

For a weakly separated conversion, run the corresponding sensitivity search:

```bash
pharos open-search \
  --adata data/GSE206741_qc_mad_scrublet_log1p.pano_criz.SE600M.h5ad \
  --start-cell "DMSO_DMSO" \
  --target-cell "panobinostat_crizotinib" \
  --cell-col "cell_type" \
  --embed-key X_state \
  --model-dir "$ST_RUN" \
  --checkpoint "$ST_CKPT" \
  --max-depth 2 \
  --batch-selection high-sensitivity \
  --robust-n-samples 5 \
  --output-dir runs/pano_criz_open_search_high_sensitivity \
  --overwrite
```

### 4.2 Locate the blinded positive-control pair

After the unbiased search is complete, audit whether the known pair was
retained and where it ranked:

```bash
pharos report open-search \
  --run-dir runs/pano_criz_open_search/search \
  --drug-a crizotinib \
  --drug-b Panobinostat \
  --output-dir runs/pano_criz_open_search/panobinostat_crizotinib_report
```

The report checks single-drug appearances, exact pair co-occurrence, both
ordered paths, and search, Sinkhorn, energy-distance, and adjusted-score ranks
and percentiles. These ranks are among retained depth-matched paths, not every
candidate expanded during search.

### 4.3 Test whether pair recovery exceeds chance

Create the target-pair table only after the open search has been run without
the drug identities as constraints:

```bash
printf 'label\tdrug_a\tdrug_b\tpair_id\npositive_control\tcrizotinib\tPanobinostat\tpanobinostat_crizotinib\n' \
  > data/pano_criz_target_pairs.tsv

pharos evaluate pair-recovery \
  --run-dirs runs/pano_criz_open_search/search \
  --labels positive_control \
  --target-pairs data/pano_criz_target_pairs.tsv \
  --depth 2 \
  --rank-threshold 128 \
  --output-dir runs/pano_criz_open_search/pair_recovery
```

By default, this evaluation compares the observed appearances of each drug,
their exact co-occurrence, and the best pair rank against 10,000 simulated
beam-matched null searches. It tests search-recovery enrichment, not
biological synergy.

## 5. Apply the template to another dataset

For a user's own SE-embedded AnnData object, replace only:

1. every `--adata` or `--query-h5ad` path;
2. `--start-cell` and `--target-cell` with exact values from the chosen
   observation column;
3. `--cell-col`/`--query-state-col` with that column's name;
4. the explicit drugs and `--MOA-pairs` with exact entries from
   [`metadata/drug_metadata.csv`](../metadata/drug_metadata.csv); and
5. each `--output-dir` so independent analyses do not overwrite one another.

Keep `--embed-key X_state` when the SE-600M embeddings use the standard key.
If `X_state` is absent, follow the embedding instructions in the
[main README](../README.md#embed-the-dataset-with-se-600m) before beginning
this workflow. Use the command-specific `--help` pages for all optional
parameters.
