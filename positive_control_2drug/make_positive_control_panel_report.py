#!/usr/bin/env python
"""Compatibility entry point for the hypothesis-driven panel report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharos_cell.hypothesis.reports.panel import *  # noqa: F403
from pharos_cell.hypothesis.reports.panel import main


if __name__ == "__main__":
    main()
