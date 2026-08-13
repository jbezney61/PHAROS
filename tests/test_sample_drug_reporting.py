import argparse
import json

import pandas as pd
import pytest

from pharos_cell.open_search import (
    save_run_manifest,
    update_report_status,
    write_skipped_sample_drug_summary,
)
from pharos_cell.reports.sample_drug import (
    StartCellMetadataNotFoundError,
    get_starting_cell_drivers,
)


def test_missing_start_cell_raises_specific_error(tmp_path) -> None:
    metadata_path = tmp_path / "cell_line_metadata.csv"
    cell_meta = pd.DataFrame(
        {
            "cell_name": ["known-cell"],
            "Organ": ["lung"],
            "Driver_Gene_Symbol": ["EGFR"],
        }
    )

    with pytest.raises(StartCellMetadataNotFoundError) as exc_info:
        get_starting_cell_drivers(
            cell_meta,
            "unknown-cell",
            metadata_path=metadata_path,
        )

    assert exc_info.value.start_cell == "unknown-cell"
    assert exc_info.value.metadata_path == metadata_path
    assert str(metadata_path) in str(exc_info.value)


def test_skipped_summary_explains_that_search_remains_valid(tmp_path) -> None:
    metadata_path = tmp_path / "cell_line_metadata.csv"
    summary_path = write_skipped_sample_drug_summary(
        tmp_path / "sample_drug_report",
        start_cell="unknown-cell",
        cell_metadata_path=metadata_path,
        reason="Starting cell `unknown-cell` was not found in the cell metadata.",
    )

    summary = summary_path.read_text()
    assert "**Status: skipped**" in summary
    assert "`unknown-cell`" in summary
    assert str(metadata_path) in summary
    assert "open-search results remain valid" in summary
    assert "generic report also remains valid" in summary
    assert "--cell-metadata" in summary


def test_run_manifest_records_report_status(tmp_path) -> None:
    args = argparse.Namespace(skip_report=False, skip_sample_drug_report=False)
    dirs = {"output_dir": tmp_path, "report_dir": tmp_path / "report"}
    manifest_path = save_run_manifest(args, {}, dirs)

    update_report_status(
        manifest_path,
        "sample_drug",
        status="skipped",
        reason="start_cell_not_found",
        start_cell="unknown-cell",
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["reports"]["generic"]["status"] == "pending"
    assert manifest["reports"]["sample_drug"] == {
        "status": "skipped",
        "reason": "start_cell_not_found",
        "start_cell": "unknown-cell",
    }
