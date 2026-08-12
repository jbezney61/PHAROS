"""Smoke tests for the installed PHAROS command."""

import pytest
from typer.testing import CliRunner

from pharos_cell import __version__
from pharos_cell.admissibility.calibration_cli import build_parser as build_calibration_parser
from pharos_cell.admissibility.manifold_cli import build_parser as build_manifold_parser
from pharos_cell.admissibility.separation_cli import parse_args as parse_separation_args
from pharos_cell.admissibility.separation_cli import resolve_cells_per_line
from pharos_cell.cli import app
from pharos_cell.hypothesis.pair_cli import parse_args as parse_hypothesis_pair_args
from pharos_cell.hypothesis.panel_cli import parse_args as parse_hypothesis_panel_args
from pharos_cell.hypothesis.reports.multi import parse_args as parse_hypothesis_summary_args
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


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["admissibility", "calibrate", "--help"], "--target-calibration-mode"),
        (["admissibility", "manifold", "--help"], "build-reference"),
        (["admissibility", "manifold", "build-reference", "--help"], "--reference-h5ad"),
        (["admissibility", "manifold", "score-query", "--help"], "--query-h5ad"),
        (["admissibility", "separation", "--help"], "--min-mean-knn-purity"),
    ],
)
def test_admissibility_help(command: list[str], expected: str) -> None:
    result = runner.invoke(app, command)

    assert result.exit_code == 0
    assert expected in result.stdout


def test_admissibility_group_help() -> None:
    result = runner.invoke(app, ["admissibility", "--help"])

    assert result.exit_code == 0
    assert "calibrate" in result.stdout
    assert "manifold" in result.stdout
    assert "separation" in result.stdout


def test_calibration_defaults() -> None:
    args = build_calibration_parser().parse_args(
        [
            "--adata", "calibration.h5ad",
            "--model-dir", "state-model",
            "--output-dir", "calibration-output",
        ]
    )

    assert args.target_calibration_mode == "all"
    assert args.cells_per_state == 100
    assert args.sinkhorn_metric == "cosine"
    assert args.sinkhorn_epsilon == 0.05
    assert args.sinkhorn_iters == 100
    assert args.projection_method == "none"


def test_manifold_defaults() -> None:
    parser = build_manifold_parser()
    reference = parser.parse_args(
        ["build-reference", "--reference-h5ad", "reference.h5ad", "--output-dir", "reference-output"]
    )
    query = parser.parse_args(
        [
            "score-query",
            "--reference-dir", "reference-output",
            "--query-h5ad", "query.h5ad",
            "--output-dir", "query-output",
        ]
    )

    assert reference.metric == "l2"
    assert reference.k == 50
    assert reference.calibration_cells_per_state == 100
    assert reference.require_gpu is True
    assert query.query_cells_per_state == 100
    assert query.require_gpu is True


def test_separation_defaults() -> None:
    args = parse_separation_args(["--adata", "query.h5ad", "--output-dir", "separation-output"])

    assert resolve_cells_per_line(args) == 256
    assert args.knn_k == 30
    assert args.min_mean_knn_purity == 0.8
    assert args.umap_n_neighbors == 15
    assert args.umap_min_dist == 0.3


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (["hypothesis-driven", "pair", "--help"], "--drug-pair"),
        (["hypothesis-driven", "panel", "--help"], "--pairs-file"),
        (["hypothesis-driven", "summarize", "--help"], "--run-dirs"),
    ],
)
def test_hypothesis_driven_help(command: list[str], expected: str) -> None:
    result = runner.invoke(app, command)

    assert result.exit_code == 0
    assert expected in result.stdout


def test_hypothesis_driven_group_help() -> None:
    result = runner.invoke(app, ["hypothesis-driven", "--help"])

    assert result.exit_code == 0
    assert "pair" in result.stdout
    assert "panel" in result.stdout
    assert "summarize" in result.stdout


def _required_hypothesis_args() -> list[str]:
    return [
        "--adata", "input.h5ad",
        "--start-cell", "start",
        "--target-cell", "target",
        "--model-dir", "state-model",
        "--output-dir", "run-output",
    ]


def test_hypothesis_pair_paper_defaults_and_alias() -> None:
    args = parse_hypothesis_pair_args(
        _required_hypothesis_args()
        + [
            "--drug-pair", "drug-a", "drug-b",
            "--moa-pairs", "['moa-a', 'moa-b']",
        ]
    )

    assert args.two_drug_pair == ["drug-a", "drug-b"]
    assert args.random_pairs == 100
    assert args.n_batches == 5
    assert args.converter_chunk_size == 16
    assert args.start_sample == "256"
    assert args.target_sample == "256"
    assert args.sinkhorn_metric == "cosine"
    assert args.sinkhorn_epsilon == 0.05
    assert args.sinkhorn_iters == 100
    assert args.projection_method == "pca_pls_da"
    assert args.projection_auto_select_components is True
    assert args.projection_whiten is False
    assert args.trajectory_embedding_space == "projection"


def test_hypothesis_pair_high_sensitivity_defaults_to_three_batches() -> None:
    args = parse_hypothesis_pair_args(
        _required_hypothesis_args()
        + ["--moa-pairs", "['moa-a', 'moa-b']", "--batch-selection", "high-sensitivity"]
    )

    assert args.batch_candidates == 1000
    assert args.batch_overlap_penalty == 0.0
    assert args.n_batches == 3


def test_hypothesis_panel_paper_defaults_and_alias() -> None:
    args = parse_hypothesis_panel_args(
        _required_hypothesis_args() + ["--pairs-file", "pairs.csv"]
    )

    assert args.approved_pairs_file == "pairs.csv"
    assert args.random_pairs == 100
    assert args.n_batches == 5
    assert args.converter_chunk_size == 16
    assert args.start_sample == "256"
    assert args.target_sample == "256"
    assert args.sinkhorn_metric == "cosine"
    assert args.sinkhorn_epsilon == 0.05
    assert args.sinkhorn_iters == 100
    assert args.projection_method == "pca_pls_da"
    assert args.projection_auto_select_components is True
    assert args.projection_whiten is False
    assert args.drug_metadata == "metadata/drug_metadata_sciplex.csv"


def test_hypothesis_panel_high_sensitivity_defaults_to_three_batches() -> None:
    args = parse_hypothesis_panel_args(
        _required_hypothesis_args()
        + ["--pairs-file", "pairs.csv", "--batch-selection", "high-sensitivity"]
    )

    assert args.batch_candidates == 1000
    assert args.batch_overlap_penalty == 0.0
    assert args.n_batches == 3


def test_hypothesis_summary_defaults() -> None:
    args = parse_hypothesis_summary_args(
        ["--run-dirs", "run-a", "run-b", "--output-dir", "summary-output"]
    )

    assert args.labels is None
    assert args.show_baseline_line is True
    assert args.title_prefix == "Positive-control 2-drug comparison"


def test_hypothesis_pair_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[str] = []

    def fake_main(argv: list[str]) -> None:
        received.extend(argv)

    monkeypatch.setattr("pharos_cell.hypothesis.pair_cli.main", fake_main)
    result = runner.invoke(app, ["hypothesis-driven", "pair", "--adata", "input.h5ad"])

    assert result.exit_code == 0
    assert received == ["--adata", "input.h5ad"]
