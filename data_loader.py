"""Compatibility imports for the packaged PHAROS data loader."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pharos_cell.data_loader import *  # noqa: F403
from pharos_cell.data_loader import _sample_indices
