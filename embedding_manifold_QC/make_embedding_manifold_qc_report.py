#!/usr/bin/env python
"""Compatibility entry point for the embedding-manifold QC report."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharos_cell.admissibility.reports.manifold import *  # noqa: F403
from pharos_cell.admissibility.reports.manifold import main


if __name__ == "__main__":
    main()
