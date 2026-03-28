"""Color palette utilities for NMF visualizations."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.colors as pc
import plotly.express as px


def component_palette(n: int) -> list[str]:
    """Generate a Viridis color scale with *n* evenly-spaced stops."""
    return pc.sample_colorscale("Viridis", [i / (n - 1) for i in range(n)])


def distinct_palette(n: int) -> list[str]:
    """Return the first *n* colors from Plotly's Alphabet qualitative palette."""
    return px.colors.qualitative.Alphabet[:n]


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


def resolve_component_colors(
    n_components: int,
    comp_order: list[int] | None = None,
    color_map: dict[str, str] | None = None,
) -> list[str]:
    """Return an ordered list of component hex colors.

    If *color_map* is provided (e.g. loaded from a JSON file), colors
    are looked up by ``Comp_<i>`` keys.  Missing entries fall back to
    :func:`component_palette`.
    """
    if comp_order is None:
        comp_order = list(range(n_components))

    if not color_map:
        return component_palette(n_components)

    auto = component_palette(n_components)
    colors: list[str] = []
    for idx, i in enumerate(comp_order):
        color = (
            color_map.get(f"Comp_{i}")
            or color_map.get(f"Component {i}")
            or color_map.get(str(i))
        )
        colors.append(color or auto[idx % len(auto)])
    return colors
