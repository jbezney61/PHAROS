"""Compatibility imports for packaged PHAROS separation QC."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharos_cell.admissibility.separation import *  # noqa: F403
