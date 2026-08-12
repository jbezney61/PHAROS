#!/usr/bin/env python
"""Compatibility entry point for ``pharos evaluate pair-recovery``."""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pharos_cell.evaluation.pair_recovery_cli import *  # noqa: F403
from pharos_cell.evaluation.pair_recovery_cli import main


if __name__ == "__main__":
    raise SystemExit(main())
