"""Compatibility imports for the packaged PHAROS projection module."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.projections import *  # noqa: F403
