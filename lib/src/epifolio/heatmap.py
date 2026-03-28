"""Multi-panel NMF heatmap figure builder (Plotly)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from epifolio.color_utils import (
    component_palette,
    distinct_palette,
    load_color_map,
    resolve_cancer_colors,
    resolve_component_colors,
)
from epifolio.data_utils import load_h_matrix, load_cfg
from epifolio.sort_utils import get_sample_order


# ---------------------------------------------------------------------------
# Grouping helpers (organ system / embryonic layer annotation strips)
# ---------------------------------------------------------------------------

def _load_grouping_data(path: str | Path) -> list[dict]:
    try:
        with open(path) as f:
            return json.load(f).get("organ_system_groupings", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _map_codes_to_groups(
    cancer_codes: list[str],
    grouping_data: list[dict],
) -> tuple[list[str], list[str]]:
    """Return parallel lists of (group_name, color) for each cancer code."""
    code_to_group: dict[str, str] = {}
    code_to_color: dict[str, str] = {}
    for g in grouping_data:
        for code in g["cancer_codes"]:
            code_to_group[code[:4]] = g["group_name"]
            code_to_color[code[:4]] = g["color"]

    names = [code_to_group.get(c, "Unknown") for c in cancer_codes]
    colors = [code_to_color.get(c, "#CCCCCC") for c in cancer_codes]
    return names, colors


# ---------------------------------------------------------------------------
# Trace builders
# ---------------------------------------------------------------------------

def _add_proportional_bars(
    fig: go.Figure,
    H_sorted: np.ndarray,
    comp_colors: list[str],
    comp_order: np.ndarray,
) -> None:
    H_prop = H_sorted / H_sorted.sum(axis=1, keepdims=True)
    for i, comp_idx in enumerate(comp_order):
        fig.add_trace(
            go.Bar(
                y=H_prop[:, i],
                name=f"Comp {comp_idx}",
                marker_color=comp_colors[i],
                hovertemplate=(
                    f"Sample: %{{x}}<br>Component: Comp {comp_idx}"
                    "<br>Proportion: %{y:.2f}<extra></extra>"
                ),
            ),
            row=2, col=1,
        )


def _add_categorical_strip(
    fig: go.Figure,
    values: list,
    colors: list[str] | dict[str, str],
    label: str,
    row: int,
) -> None:
    """Add a 1-pixel-high heatmap strip for a categorical annotation."""
    unique = sorted(set(values))
    val_to_idx = {v: i for i, v in enumerate(unique)}
    idx_arr = [val_to_idx[v] for v in values]

    if isinstance(colors, dict):
        color_list = [colors.get(v, "#CCCCCC") for v in unique]
    else:
        # Parallel list of per-sample colors — deduplicate by value
        seen: dict[str, str] = {}
        for v, c in zip(values, colors):
            if v not in seen:
                seen[v] = c
        color_list = [seen.get(v, "#CCCCCC") for v in unique]

    n = len(unique)
    if n <= 1:
        scale = [(0, color_list[0] if color_list else "#CCCCCC"),
                 (1, color_list[0] if color_list else "#CCCCCC")]
    else:
        scale = [(i / (n - 1), color_list[i]) for i in range(n)]

    fig.add_trace(
        go.Heatmap(
            z=[idx_arr],
            colorscale=scale,
            showscale=False,
            hovertemplate=f"Sample: %{{x}}<br>{label}: %{{customdata}}<extra></extra>",
            customdata=[values],
        ),
        row=row, col=1,
    )


def _add_component_strip(
    fig: go.Figure,
    H_ord: np.ndarray,
    samp_order: np.ndarray,
    comp_colors: list[str],
    comp_order: np.ndarray,
) -> None:
    n_comps = H_ord.shape[1]
    winners = np.argmax(H_ord[samp_order], axis=1)
    winner_ids = np.array([comp_order[i] for i in winners])

    if n_comps <= 1:
        scale = [(0, comp_colors[0]), (1, comp_colors[0])]
    else:
        scale = []
        for i in range(n_comps):
            if i == 0:
                scale.append((0, comp_colors[i]))
            else:
                scale.append(((i - 0.5) / (n_comps - 1), comp_colors[i - 1]))
                scale.append((i / (n_comps - 1), comp_colors[i]))
        scale.append((1, comp_colors[-1]))

    fig.add_trace(
        go.Heatmap(
            z=[winners],
            colorscale=scale,
            showscale=False,
            hovertemplate="Sample: %{x}<br>Dominant Component: Comp %{customdata}<extra></extra>",
            customdata=[winner_ids],
        ),
        row=3, col=1,
    )


# ---------------------------------------------------------------------------
# Main figure builder
# ---------------------------------------------------------------------------

def create_heatmap_figure(
    cfg_path: str | Path = "config.json",
    sort_method: str = "component",
    *,
    csv_path: Path | None = None,
    selection: list[int] | None = None,
) -> go.Figure:
    """Build the full multi-panel NMF heatmap.

    Panels (top to bottom):
    1. Main heatmap (sample x component activity)
    2. Stacked proportional bar chart
    3. Dominant component strip
    4. Cancer type strip
    5. Organ system strip
    6. Embryonic layer strip
    """
    cfg = load_cfg(cfg_path)

    # --- data ---
    if csv_path is None:
        csv_path = Path(cfg.get("DEFAULT_CSV_FILENAME", "data/all_H_component_contributions_k16.csv"))

    H, sample_ids, _comp_names = load_h_matrix(csv_path, selection=selection)
    n_samples, n_comps = H.shape

    # --- ordering ---
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    organ_path = cfg.get("JSON_FILENAME_ORGAN_SYSTEM")
    emb_path = cfg.get("JSON_FILENAME_EMBRYONIC_LAYER")
    cancer_types = [sid[:4] for sid in sample_ids]

    samp_order = get_sample_order(
        sort_method, H, sample_ids, cancer_types,
        organ_system_path=organ_path,
        embryonic_layer_path=emb_path,
    )
    H_sorted = H_ord[samp_order]
    x_labels_short = np.array([sid[:4] for sid in np.array(sample_ids)[samp_order]])

    # --- colors ---
    comp_color_map = load_color_map(cfg.get("JSON_FILENAME_COMPONENT_COLORS", ""))
    comp_colors = resolve_component_colors(n_comps, comp_order.tolist(), comp_color_map)
    cancer_color_map = resolve_cancer_colors(cancer_types, load_color_map(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS", "")))

    organ_data = _load_grouping_data(organ_path) if organ_path else []
    emb_data = _load_grouping_data(emb_path) if emb_path else []
    organ_names, organ_colors = _map_codes_to_groups(x_labels_short.tolist(), organ_data)
    emb_names, emb_colors = _map_codes_to_groups(x_labels_short.tolist(), emb_data)

    # --- figure ---
    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.20, 0.06, 0.06, 0.06, 0.06],
        vertical_spacing=0.04,
        subplot_titles=(
            "NMF Component Activities",
            "Proportional NMF Activity",
            "Dominant Component",
            "Cancer Type",
            "Organ System",
            "Embryonic Layer",
        ),
    )

    # 1) Main heatmap
    fig.add_trace(
        go.Heatmap(
            z=H_sorted.T,
            colorscale="Turbo",
            showscale=False,
            hovertemplate="Sample: %{x}<br>Component: %{y}<br>Activity: %{z}<extra></extra>",
        ),
        row=1, col=1,
    )

    # 2) Proportional bars
    _add_proportional_bars(fig, H_sorted, comp_colors, comp_order)

    # 3) Dominant component
    _add_component_strip(fig, H_ord, samp_order, comp_colors, comp_order)

    # 4) Cancer type
    ordered_cancer = list(np.array(cancer_types)[samp_order])
    _add_categorical_strip(fig, ordered_cancer, cancer_color_map, "Cancer Type", row=4)

    # 5) Organ system
    _add_categorical_strip(fig, organ_names, organ_colors, "Organ System", row=5)

    # 6) Embryonic layer
    _add_categorical_strip(fig, emb_names, emb_colors, "Embryonic Layer", row=6)

    # --- layout ---
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_comps)),
        ticktext=[f"Comp {i}" for i in comp_order],
        autorange="reversed",
        row=1, col=1,
    )
    for r in range(1, 6):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_samples)),
        ticktext=x_labels_short.tolist(),
        tickangle=90,
        row=6, col=1,
    )
    for r in range(2, 7):
        fig.update_yaxes(showticklabels=False, row=r, col=1)
    fig.update_yaxes(title="Proportion", range=[0, 1], row=2, col=1)

    fig.update_layout(
        height=max(900, n_comps * 25 + 300),
        width=1400,
        margin=dict(l=80, r=80, t=20, b=100),
        barmode="stack",
        showlegend=False,
        hovermode="x unified",
        selectdirection="h",
        dragmode="select",
    )

    return fig
