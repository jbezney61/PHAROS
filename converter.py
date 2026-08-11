"""Compatibility imports for the packaged PHAROS converter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.converter import *  # noqa: F403
