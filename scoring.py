"""Compatibility imports for the packaged PHAROS scoring module."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.scoring import *  # noqa: F403
