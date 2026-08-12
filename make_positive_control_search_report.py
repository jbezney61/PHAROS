#!/usr/bin/env python
"""Compatibility entry point for ``pharos report open-search``."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pharos_cell.reports.open_search import *  # noqa: F403
from pharos_cell.reports.open_search import main


if __name__ == "__main__":
    main()
