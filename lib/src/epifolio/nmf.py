"""NMF activity heatmap: a data-decoupled renderer plus config-driven helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from epifolio.colors import (
    component_palette,
    distinct_palette,
    unique_sorted_unknown_last,
)
from epifolio.tcga import patient_id_from_sample_id


@dataclass
class HeatmapStrip:
    """A single annotation row beneath the heatmap, decoupled from any schema.

    ``values`` and ``colors`` are per-sample and given in the original
    (unsorted) sample order; the renderer reorders them to match the heatmap.
    ``hover`` holds arbitrary extra per-sample fields shown in the tooltip.
    """

    label: str
    values: Sequence[str]
    colors: Sequence[str]
    hover: Mapping[str, Sequence[str]] = field(default_factory=dict)


def _palette_map(
    values: list[str],
    palette: list[str],
    unknown_color: str,
) -> dict[str, str]:
    unique_values = unique_sorted_unknown_last(values)
    if not unique_values:
        return {}

    color_map: dict[str, str] = {}
    palette_size = max(1, len(palette))
    for index, value in enumerate(unique_values):
        if value == "Unknown":
            color_map[value] = unknown_color
        else:
            color_map[value] = palette[index % palette_size]
    return color_map


def _cohort_color_map(
    values: list[str],
    cancer_color_map: dict[str, str],
    unknown_color: str,
) -> dict[str, str]:
    unique_values = unique_sorted_unknown_last(values)
    fallback_palette = distinct_palette(max(1, len(unique_values)))

    color_map: dict[str, str] = {}
    fallback_index = 0
    for value in unique_values:
        if value == "Unknown":
            color_map[value] = unknown_color
            continue

        preferred_keys = [value, f"{value}x", f"{value}xx"]
        color = None
        for key in preferred_keys:
            color = cancer_color_map.get(key)
            if color is not None:
                break
        if color is None:
            for key, candidate in cancer_color_map.items():
                if key.startswith(value):
                    color = candidate
                    break

        if color is None:
            color = fallback_palette[fallback_index % len(fallback_palette)]
            fallback_index += 1

        color_map[value] = color

    return color_map


def _resolve_strip_colors(
    strip_spec: dict,
    values: list[str],
    cancer_color_map: dict[str, str],
    unknown_color: str,
) -> tuple[list[str], dict[str, str]]:
    if strip_spec.get("palette_source") == "cancer_type_colors":
        color_map = _cohort_color_map(values, cancer_color_map, unknown_color)
    elif "color_map" in strip_spec:
        color_map = {str(key): str(value) for key, value in strip_spec["color_map"].items()}
        color_map.setdefault("Unknown", unknown_color)
    else:
        color_map = _palette_map(values, strip_spec.get("palette", []), unknown_color)

    return [color_map.get(value, unknown_color) for value in values], color_map


def _add_proportional_bar_chart(
    fig: go.Figure,
    H_sorted: np.ndarray,
    comp_colors: list[str],
    comp_order: np.ndarray,
    *,
    x_values: list[str] | np.ndarray,
    row: int = 2,
) -> None:
    H_proportional = H_sorted / H_sorted.sum(axis=1, keepdims=True)
    for i, comp_idx in enumerate(comp_order):
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=H_proportional[:, i],
                name=f"Comp {comp_idx}",
                marker_color=comp_colors[i],
                hovertemplate=(
                    "Sample: %{x}<br>"
                    f"Component: Comp {comp_idx}<br>"
                    "Proportion: %{y:.2f}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )


def _add_component_strip(
    fig: go.Figure,
    H_ord: np.ndarray,
    sample_order: np.ndarray,
    comp_colors: list[str],
    n_comps: int,
    comp_order: np.ndarray,
    *,
    x_values: list[str] | np.ndarray,
    row: int = 3,
) -> None:
    winning_comp_indices = np.argmax(H_ord[sample_order], axis=1)
    winning_comp_numbers = np.array([comp_order[index] for index in winning_comp_indices])

    if n_comps == 1:
        scale = [(0.0, comp_colors[0]), (1.0, comp_colors[0])]
    else:
        scale = [(0.0, comp_colors[0])]
        for index in range(1, n_comps):
            scale.append(((index - 0.5) / (n_comps - 1), comp_colors[index - 1]))
            scale.append((index / (n_comps - 1), comp_colors[index]))
        scale.append((1.0, comp_colors[-1]))

    fig.add_trace(
        go.Heatmap(
            x=x_values,
            z=[winning_comp_indices],
            colorscale=scale,
            showscale=False,
            customdata=[winning_comp_numbers],
            hovertemplate="Sample: %{x}<br>Dominant Component: Comp %{customdata}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _add_annotation_strip(
    fig: go.Figure,
    x_values: list[str] | np.ndarray,
    strip: HeatmapStrip,
    *,
    row: int,
) -> None:
    """Draw one annotation row from a :class:`HeatmapStrip` (display order)."""
    values = list(strip.values)
    colors = list(strip.colors)

    unique_values = sorted(set(values))
    value_to_idx = {value: index for index, value in enumerate(unique_values)}
    z_indices = [value_to_idx[value] for value in values]

    value_to_color: dict[str, str] = {}
    for value, color in zip(values, colors):
        value_to_color.setdefault(value, color)

    if len(unique_values) == 1:
        scale = [
            (0.0, value_to_color[unique_values[0]]),
            (1.0, value_to_color[unique_values[0]]),
        ]
    else:
        scale = [
            (index / (len(unique_values) - 1), value_to_color[value])
            for index, value in enumerate(unique_values)
        ]

    hover_fields = list(strip.hover.keys())
    customdata = np.array(
        [[
            [values[i], *[strip.hover[field][i] for field in hover_fields]]
            for i in range(len(values))
        ]],
        dtype=object,
    )
    hover_lines = [f"{strip.label}: %{{customdata[0]}}"]
    for offset, field_name in enumerate(hover_fields, start=1):
        hover_lines.append(f"{field_name}: %{{customdata[{offset}]}}")

    fig.add_trace(
        go.Heatmap(
            x=x_values,
            z=[z_indices],
            colorscale=scale,
            showscale=False,
            customdata=customdata,
            hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _add_selection_strip(
    fig: go.Figure,
    x_values: list[str] | np.ndarray,
    highlight_mask: list[int],
    *,
    row: int,
) -> None:
    """Draw a single-row strip highlighting the linked selection in gold.

    Unselected cells use a faint gray so the strip stays visible as a frame
    even when nothing is selected; selected cells use a saturated gold so a
    small selection stands out against the otherwise neutral strip.
    """
    fig.add_trace(
        go.Heatmap(
            x=x_values,
            z=[highlight_mask],
            colorscale=[(0.0, "#EEEEEE"), (1.0, "#FFC400")],
            zmin=0,
            zmax=1,
            showscale=False,
            customdata=[["selected" if value else "" for value in highlight_mask]],
            hovertemplate="Sample: %{x}<br>%{customdata}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def _metadata_group_sort(H: np.ndarray, group_values: list[str]) -> np.ndarray:
    final_order: list[int] = []
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    values = np.asarray(group_values)

    for group in unique_sorted_unknown_last(group_values):
        sub_indices = np.where(values == group)[0]
        if len(sub_indices) == 0:
            continue
        winners = np.argmax(H_ord[sub_indices], axis=1)
        sub_order = np.lexsort((-H_ord[sub_indices][np.arange(len(sub_indices)), winners], winners))
        final_order.extend(sub_indices[sub_order])
    return np.asarray(final_order, dtype=int)


def sample_sort_order(
    sort_method: str,
    H: np.ndarray,
    sample_ids: Sequence[str],
    *,
    cancer_types: Sequence[str] | None = None,
    group_values: Sequence[str] | None = None,
) -> np.ndarray:
    """Return a permutation of sample indices for the given sort strategy.

    - ``"component"`` (default/fallback): by dominant component, then activity.
    - ``"alphabetical"``: by sample ID.
    - ``"cancer_type"``: grouped by ``cancer_types`` (required for this mode).
    - ``"organ_system"``: grouped by ``group_values`` (required for this mode).
    """
    H = np.asarray(H, dtype=float)
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    winners = np.argmax(H_ord, axis=1)
    by_component = np.lexsort((-H_ord[np.arange(len(H_ord)), winners], winners))

    if sort_method == "alphabetical":
        return np.argsort(list(sample_ids))
    if sort_method == "cancer_type" and cancer_types is not None:
        return np.lexsort(
            (-H_ord[np.arange(len(H_ord)), winners], winners, list(cancer_types))
        )
    if sort_method == "organ_system" and group_values:
        return _metadata_group_sort(H, list(group_values))
    return by_component


def component_color_sequence(color_map: dict[str, str] | None, n_comps: int) -> list[str]:
    """Per-component-id colors (index ``i`` -> color for component ``i``).

    Resolves an already-loaded ``color_map`` (keys like ``Comp_0``); missing
    entries fall back to :func:`component_palette`. Pure transform, no I/O.
    """
    color_map = color_map or {}
    palette = component_palette(n_comps)
    colors = []
    for comp_id in range(n_comps):
        color = (
            color_map.get(f"Comp_{comp_id}")
            or color_map.get(f"Component {comp_id}")
            or color_map.get(f"Comp {comp_id}")
            or color_map.get(str(comp_id))
        )
        colors.append(color or palette[comp_id % len(palette)])
    return colors


def _configure_layout(
    fig: go.Figure,
    n_comps: int,
    n_samples: int,
    comp_order: np.ndarray,
    x_labels: np.ndarray,
    x_labels_short: np.ndarray,
    total_rows: int,
    n_strip_rows: int,
) -> None:
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_comps)),
        ticktext=[f"Comp {index}" for index in comp_order],
        autorange="reversed",
        row=1,
        col=1,
    )

    for row in range(1, total_rows):
        fig.update_xaxes(showticklabels=False, row=row, col=1)

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_labels.tolist(),
        ticktext=x_labels_short,
        tickangle=90,
        row=total_rows,
        col=1,
    )

    for row in range(2, total_rows + 1):
        fig.update_yaxes(showticklabels=False, row=row, col=1)

    fig.update_layout(
        height=max(1100, n_comps * 32 + 70 * n_strip_rows + 300),
        # No fixed width: let the figure stretch to the full container width so
        # the heatmap and every strip below it span the page (the strips are
        # shared-x subplots, so they always match the main heatmap's width).
        autosize=True,
        margin=dict(l=80, r=80, t=60, b=100),
        barmode="stack",
        hovermode="x unified",
        selectdirection="h",
        dragmode="select",
        showlegend=False,
    )


def nmf_activity_heatmap(
    H: np.ndarray,
    sample_ids: Sequence[str],
    *,
    component_colors: Sequence[str] | None = None,
    component_order: Sequence[int] | None = None,
    sample_order: Sequence[int] | None = None,
    column_tick_labels: Sequence[str] | None = None,
    strips: Sequence[HeatmapStrip] = (),
    highlight_mask: Sequence[int] | None = None,
) -> go.Figure:
    """Render the stacked NMF-activity heatmap from already-prepared data.

    This is a pure renderer: it knows nothing about config files, remote data,
    or any metadata schema. Callers prepare the matrix, ordering, colors and
    annotation :class:`HeatmapStrip`s and pass them in.

    Parameters
    ----------
    H:
        ``(n_samples, n_components)`` activity/proportion matrix.
    sample_ids:
        Column labels, length ``n_samples``.
    component_colors:
        Per-component-id colors (index ``i`` -> color for component ``i``).
        Defaults to a Viridis palette.
    component_order, sample_order:
        Permutations of component ids / sample indices. Default to
        descending total activity / dominant-component ordering.
    column_tick_labels:
        Short per-sample x-axis tick labels (e.g. cancer type). Defaults to
        ``sample_ids``.
    strips:
        Annotation rows (values/colors/hover given in original sample order).
    highlight_mask:
        Optional per-sample 0/1 flags (original order). When provided, a
        ``Selection`` highlight strip is drawn; when ``None`` it is omitted.
    """
    H = np.asarray(H, dtype=float)
    sample_ids = list(sample_ids)
    n_samples, n_comps = H.shape

    if component_order is None:
        component_order = np.argsort(-H.sum(axis=0))
    component_order = np.asarray(component_order)

    if sample_order is None:
        sample_order = sample_sort_order("component", H, sample_ids)
    sample_order = np.asarray(sample_order)

    if column_tick_labels is None:
        column_tick_labels = sample_ids
    column_tick_labels = list(column_tick_labels)

    if component_colors is None:
        palette = component_palette(n_comps)
        comp_colors = [palette[position % len(palette)] for position in range(n_comps)]
    else:
        comp_colors = [component_colors[comp_id] for comp_id in component_order]

    H_ord = H[:, component_order]
    H_sorted = H_ord[sample_order]
    x_labels = np.asarray(sample_ids)[sample_order]
    ordered_indices = list(sample_order)
    x_labels_short = np.asarray([column_tick_labels[index] for index in ordered_indices])

    has_selection = highlight_mask is not None

    subplot_titles = ["NMF Component Activities"]
    row_heights = [0.395 if has_selection else 0.43]
    if has_selection:
        subplot_titles.append("Selection")
        row_heights.append(0.035)
    subplot_titles += ["Proportional NMF Activity", "Dominant Component"]
    row_heights += [0.17 if has_selection else 0.18, 0.05]
    for strip in strips:
        subplot_titles.append(strip.label)
        row_heights.append(0.04)
    total_rows = len(row_heights)

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=tuple(subplot_titles),
    )

    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=[f"Comp {index}" for index in component_order],
            z=H_sorted.T,
            colorscale="Turbo",
            showscale=False,
            hovertemplate="Sample: %{x}<br>Component: %{y}<br>Activity: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    row = 2
    if has_selection:
        ordered_mask = [int(highlight_mask[index]) for index in ordered_indices]
        _add_selection_strip(fig, x_labels, ordered_mask, row=row)
        row += 1
    _add_proportional_bar_chart(fig, H_sorted, comp_colors, component_order, x_values=x_labels, row=row)
    row += 1
    _add_component_strip(
        fig, H_ord, sample_order, comp_colors, n_comps, component_order, x_values=x_labels, row=row
    )
    row += 1
    for strip in strips:
        ordered_strip = HeatmapStrip(
            label=strip.label,
            values=[strip.values[index] for index in ordered_indices],
            colors=[strip.colors[index] for index in ordered_indices],
            hover={
                field: [series[index] for index in ordered_indices]
                for field, series in strip.hover.items()
            },
        )
        _add_annotation_strip(fig, x_labels, ordered_strip, row=row)
        row += 1

    _configure_layout(
        fig,
        n_comps,
        n_samples,
        component_order,
        x_labels,
        x_labels_short,
        total_rows,
        len(strips) + (1 if has_selection else 0),
    )
    return fig


def build_metadata_strips(
    strip_specs: Sequence[dict],
    sample_ids: Sequence[str],
    strip_metadata: pd.DataFrame,
    *,
    id_column: str = "submitter_id",
    cancer_color_map: dict[str, str] | None = None,
    unknown_color: str = "#CCCCCC",
) -> list[HeatmapStrip]:
    """Build :class:`HeatmapStrip`s by joining loaded strip metadata onto samples.

    Pure transform — the caller loads ``strip_metadata`` (a per-patient table
    keyed by ``id_column``); this maps each sample to its patient via the TCGA
    barcode and resolves per-strip colors from each spec. No I/O, no config.
    """
    strip_specs = list(strip_specs)
    sample_ids = list(sample_ids)
    if not strip_specs:
        return []
    cancer_color_map = cancer_color_map or {}

    merged = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": [patient_id_from_sample_id(sample_id) for sample_id in sample_ids],
        }
    ).merge(strip_metadata, left_on="patient_id", right_on=id_column, how="left")

    submitter_ids = (
        merged.get(id_column, merged["patient_id"])
        .fillna(merged["patient_id"])
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )
    cohorts = (
        merged.get("cohort", pd.Series(["Unknown"] * len(sample_ids)))
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )

    strips: list[HeatmapStrip] = []
    for strip_spec in strip_specs:
        column = strip_spec["column"]
        if column in merged.columns:
            values = merged[column].fillna("Unknown").astype(str).tolist()
        else:
            values = ["Unknown"] * len(sample_ids)
        colors, _ = _resolve_strip_colors(strip_spec, values, cancer_color_map, unknown_color)
        strips.append(
            HeatmapStrip(
                label=strip_spec["label"],
                values=values,
                colors=colors,
                hover={"submitter_id": submitter_ids, "cohort": cohorts},
            )
        )
    return strips
