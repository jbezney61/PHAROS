#!/usr/bin/env python
"""Compatibility imports for the packaged pair-recovery plotting helpers."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pharos_cell.evaluation.pair_recovery_plotting import *  # noqa: F403
