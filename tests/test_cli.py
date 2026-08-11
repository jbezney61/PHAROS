"""Smoke tests for the installed PHAROS command."""

import pytest
from typer.testing import CliRunner

from pharos_cell import __version__
from pharos_cell.cli import app
from pharos_cell.open_search import parse_args

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Target-directed drug-combination screening" in result.stdout
    assert "open-search" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"pharos {__version__}"


def test_open_search_help_exposes_original_options() -> None:
    result = runner.invoke(app, ["open-search", "--help"])

    assert result.exit_code == 0
    assert "usage: pharos open-search" in result.stdout
    assert "--adata" in result.stdout
    assert "--batch-selection" in result.stdout
    assert "--projection-method" in result.stdout


def test_open_search_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[str] = []

    def fake_main(argv: list[str]) -> None:
        received.extend(argv)

    monkeypatch.setattr("pharos_cell.open_search.main", fake_main)
    result = runner.invoke(app, ["open-search", "--adata", "input.h5ad", "--overwrite"])

    assert result.exit_code == 0
    assert received == ["--adata", "input.h5ad", "--overwrite"]


def _required_open_search_args() -> list[str]:
    return [
        "--adata", "input.h5ad",
        "--start-cell", "start",
        "--target-cell", "target",
        "--model-dir", "state-model",
        "--output-dir", "run-output",
    ]


def test_open_search_paper_defaults() -> None:
    args = parse_args(_required_open_search_args())

    assert args.algorithm == "diverse_beam"
    assert args.path_overlap_penalty == 25.0
    assert args.max_depth == 2
    assert args.beam_size == 128
    assert args.prefilter_metric == "sinkhorn_low_iter"
    assert args.prefilter_multiplier == 10
    assert args.converter_chunk_size == 16
    assert args.start_sample == "256"
    assert args.target_sample == "256"
    assert args.sinkhorn_metric == "cosine"
    assert args.sinkhorn_epsilon == 0.05
    assert args.sinkhorn_iters == 100
    assert args.conversion_threshold == 0.025
    assert args.robust_rerank is True
    assert args.robust_n_samples == 5
    assert args.robust_metric == "sinkhorn"
    assert args.robust_aggregation == "mean_plus_std"
    assert args.robust_std_penalty == 0.5
    assert args.projection_method == "pca_pls_da"
    assert args.projection_auto_select_components is True
    assert args.projection_whiten is False
    assert args.projection_selection_pca_grid == "96,128,192,256"
    assert args.projection_selection_pls_grid == "64,96,128,192"
    assert args.drug_metadata == "metadata/drug_metadata_sciplex.csv"


def test_open_search_high_sensitivity_defaults_to_three_robust_batches() -> None:
    args = parse_args(_required_open_search_args() + ["--batch-selection", "high-sensitivity"])

    assert args.batch_candidates == 1000
    assert args.batch_overlap_penalty == 0.0
    assert args.robust_n_samples == 3
