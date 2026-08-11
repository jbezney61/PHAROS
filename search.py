"""Compatibility imports for the packaged PHAROS search module."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.search import *  # noqa: F403
