#!/usr/bin/env python
"""Compatibility entry point for the packaged sample/drug report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.reports.sample_drug import *  # noqa: F403
from pharos_cell.reports.sample_drug import main


if __name__ == "__main__":
    main()
