"""PHAROS: target-directed drug-combination screening from cell-state embeddings.

Expose the installed ``pharos-cell`` package version, with a fallback version
for source checkouts that do not have installed distribution metadata.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pharos-cell")
except PackageNotFoundError:
    __version__ = "0.1.0a0"

__all__ = ["__version__"]

