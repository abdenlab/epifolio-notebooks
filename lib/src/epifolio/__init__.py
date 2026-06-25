"""epifolio: shared utilities for TCGA cancer regulatory genomics notebooks."""

from importlib.resources import files
from pathlib import Path

ASSETS = files("epifolio") / "assets"


def asset_path(*parts: str) -> Path:
    """Return a concrete path inside the packaged epifolio asset tree."""
    path = ASSETS
    for part in parts:
        path = path / part
    return Path(path)
