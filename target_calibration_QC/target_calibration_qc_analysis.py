#!/usr/bin/env python
"""Compatibility entry point for PHAROS target-calibration QC."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharos_cell.admissibility.calibration_cli import main


if __name__ == "__main__":
    main()
