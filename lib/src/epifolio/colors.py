"""Color palette utilities for NMF visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.colors as pc
import plotly.express as px


def unique_sorted_unknown_last(values: list[str]) -> list[str]:
    """Return the unique values sorted alphabetically, with "Unknown" last.

    NA/NaN values are dropped so they never become a literal category.
    """
    unique_values = sorted({str(value) for value in values if pd.notna(value)})
    if "Unknown" in unique_values:
        unique_values = [value for value in unique_values if value != "Unknown"] + ["Unknown"]
    return unique_values


def component_palette(n: int) -> list[str]:
    """Generate a Viridis color scale with *n* evenly-spaced stops."""
    return pc.sample_colorscale("Viridis", [i / (n - 1) for i in range(n)])


def distinct_palette(n: int) -> list[str]:
    """Return *n* colors from Plotly's Alphabet qualitative palette.

    The 26-color base palette is cycled when ``n`` exceeds it, so the result
    always has exactly ``n`` entries (callers index it positionally).
    """
    base = px.colors.qualitative.Alphabet
    return [base[index % len(base)] for index in range(n)]


def load_color_map(path: str | Path) -> dict[str, str]:
    """Load a JSON color map (e.g. cancer types, components).

    Returns an empty dict when *path* is falsy or the file is missing.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def resolve_cancer_colors(
    cancer_types: list[str],
    user_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a complete cancer-type -> hex-color mapping.

    Colors from *user_map* take priority; any types not covered get
    auto-assigned from :func:`distinct_palette`.
    """
    user_map = user_map or {}
    unique = sorted(set(cancer_types))
    auto = distinct_palette(len(unique))
    return {ct: user_map.get(ct, auto[i]) for i, ct in enumerate(unique)}
