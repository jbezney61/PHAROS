"""Run the PHAROS command-line interface with ``python -m pharos_cell``.

Delegate to the same entry point used by the installed ``pharos`` command.
"""

from pharos_cell.cli import main

if __name__ == "__main__":
    main()

