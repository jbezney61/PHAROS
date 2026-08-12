"""Compatibility imports for the packaged hypothesis-driven engine."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pharos_cell.hypothesis.engine import *  # noqa: F403
