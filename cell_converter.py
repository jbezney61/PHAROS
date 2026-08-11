#!/usr/bin/env python
"""Compatibility entry point for the original open-search command."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.open_search import main


if __name__ == "__main__":
    main()
