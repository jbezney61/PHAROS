#!/usr/bin/env python
"""Compatibility entry point for the packaged open-search report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.reports.search import *  # noqa: F403
from pharos_cell.reports.search import main


if __name__ == "__main__":
    main()
