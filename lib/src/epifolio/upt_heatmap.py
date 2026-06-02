"""Multi-analysis NMF heatmap helpers for the production Marimo notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from epifolio.color_utils import component_palette, distinct_palette, load_color_map
from epifolio.data_utils import _get_prepared_data, prepare_grandscatter_data, resolve_analysis_cfg


def _patient_id_from_sample_id(sample_id: str) -> str | None:
    parts = str(sample_id).split("-")
    if "TCGA" in parts:
        idx = parts.index("TCGA")
        if idx + 2 < len(parts):
            return "-".join(parts[idx : idx + 3])
    if len(parts) >= 4 and parts[1] == "TCGA":
        return "-".join(parts[1:4])
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return None


def _load_strip_metadata(cfg: dict) -> pd.DataFrame:
    metadata_df = pd.read_csv(cfg["STRIP_METADATA_FILENAME"])
    id_column = cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id")
    return metadata_df.drop_duplicates(subset=[id_column]).copy()


def _unique_sorted_unknown_last(values: list[str]) -> list[str]:
    unique_values = sorted({str(value) for value in values})
    if "Unknown" in unique_values:
        unique_values = [value for value in unique_values if value != "Unknown"] + ["Unknown"]
    return unique_values


def _palette_map(
    values: list[str],
    palette: list[str],
    unknown_color: str,
) -> dict[str, str]:
    unique_values = _unique_sorted_unknown_last(values)
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
    unique_values = _unique_sorted_unknown_last(values)
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


def _add_metadata_annotation_strip(
    fig: go.Figure,
    x_values: list[str] | np.ndarray,
    group_names: list[str],
    group_colors: list[str],
    label: str,
    submitter_ids: list[str],
    cohorts: list[str],
    winning_components: list[str],
    *,
    row: int,
) -> None:
    unique_groups = sorted(set(group_names))
    group_to_idx = {group: index for index, group in enumerate(unique_groups)}
    group_idx_arr = [group_to_idx[group] for group in group_names]

    unique_colors: dict[str, str] = {}
    for group_name, color in zip(group_names, group_colors):
        if group_name not in unique_colors:
            unique_colors[group_name] = color

    if len(unique_groups) == 1:
        group_scale = [
            (0.0, unique_colors[unique_groups[0]]),
            (1.0, unique_colors[unique_groups[0]]),
        ]
    else:
        group_scale = [
            (index / (len(unique_groups) - 1), unique_colors[group])
            for index, group in enumerate(unique_groups)
        ]

    customdata = np.array(
        [[
            [group_name, submitter_id, cohort, winning_component]
            for group_name, submitter_id, cohort, winning_component in zip(
                group_names,
                submitter_ids,
                cohorts,
                winning_components,
            )
        ]],
        dtype=object,
    )

    fig.add_trace(
        go.Heatmap(
            x=x_values,
            z=[group_idx_arr],
            colorscale=group_scale,
            showscale=False,
            customdata=customdata,
            hovertemplate=(
                f"{label}: %{{customdata[0]}}"
                "<br>submitter_id: %{customdata[1]}"
                "<br>cohort: %{customdata[2]}"
                "<br>%{customdata[3]}<extra></extra>"
            ),
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


def _build_strip_values(sample_ids: list[str], cfg: dict) -> tuple[dict[str, list[str]], pd.DataFrame]:
    strip_specs = cfg.get("HEATMAP_STRIPS", [])
    metadata_df = _load_strip_metadata(cfg)
    id_column = cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id")

    strip_frame = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "patient_id": [_patient_id_from_sample_id(sample_id) for sample_id in sample_ids],
        }
    )
    merged = strip_frame.merge(
        metadata_df,
        left_on="patient_id",
        right_on=id_column,
        how="left",
    )

    values: dict[str, list[str]] = {}
    for strip_spec in strip_specs:
        column = strip_spec["column"]
        if column in merged.columns:
            values[column] = merged[column].fillna("Unknown").astype(str).tolist()
        else:
            values[column] = ["Unknown"] * len(sample_ids)
    return values, merged


def _metadata_group_sort(H: np.ndarray, group_values: list[str]) -> np.ndarray:
    final_order: list[int] = []
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    values = np.asarray(group_values)

    for group in _unique_sorted_unknown_last(group_values):
        sub_indices = np.where(values == group)[0]
        if len(sub_indices) == 0:
            continue
        winners = np.argmax(H_ord[sub_indices], axis=1)
        sub_order = np.lexsort((-H_ord[sub_indices][np.arange(len(sub_indices)), winners], winners))
        final_order.extend(sub_indices[sub_order])
    return np.asarray(final_order, dtype=int)


def _get_sample_order(
    sort_method: str,
    H: np.ndarray,
    sample_ids: list[str],
    cancer_types: list[str],
    strip_values: dict[str, list[str]],
) -> np.ndarray:
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]
    winners = np.argmax(H_ord, axis=1)

    if sort_method == "component":
        return np.lexsort((-H_ord[np.arange(len(H_ord)), winners], winners))
    if sort_method == "alphabetical":
        return np.argsort(sample_ids)
    if sort_method == "cancer_type":
        return np.lexsort((-H_ord[np.arange(len(H_ord)), winners], winners, cancer_types))
    if sort_method == "organ_system" and strip_values.get("organ_system"):
        return _metadata_group_sort(H, strip_values["organ_system"])
    return np.lexsort((-H_ord[np.arange(len(H_ord)), winners], winners))


def _load_component_colors(
    component_color_file: str | Path,
    n_comps: int,
    comp_order: np.ndarray,
) -> list[str]:
    color_map = load_color_map(component_color_file)
    auto_colors = component_palette(n_comps)
    colors = []
    for index, comp_idx in enumerate(comp_order):
        color = (
            color_map.get(f"Comp_{comp_idx}")
            or color_map.get(f"Component {comp_idx}")
            or color_map.get(f"Comp {comp_idx}")
            or color_map.get(str(comp_idx))
        )
        colors.append(color or auto_colors[index % len(auto_colors)])
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


def create_heatmap_figure(
    cfg_path: str | Path,
    sort_method: str = "component",
    selected_sample_ids: list[int] | None = None,
    analysis_name: str | None = None,
) -> go.Figure:
    cfg = resolve_analysis_cfg(cfg_path, analysis_name)
    heatmap_csv = cfg.get("HEATMAP_CSV_FILENAME", cfg.get("DEFAULT_CSV_FILENAME"))
    metadata_path = cfg.get("ANALYSIS_METADATA_FILENAME")

    H, sample_ids_from_file, cancer_types = _get_prepared_data(
        heatmap_csv,
        selection=selected_sample_ids,
        metadata_path=metadata_path,
        metadata_sample_id_column=str(cfg.get("ANALYSIS_METADATA_SAMPLE_ID_COLUMN", "sample_id")),
        cancer_type_column=str(cfg.get("ANALYSIS_CANCER_TYPE_COLUMN", "cancer_type")),
    )

    n_samples, n_comps = H.shape
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    strip_specs = cfg.get("HEATMAP_STRIPS", [])
    strip_values, merged_strip_metadata = _build_strip_values(sample_ids_from_file, cfg)
    sample_order = _get_sample_order(
        sort_method,
        H,
        sample_ids_from_file,
        cancer_types,
        strip_values,
    )

    H_sorted = H_ord[sample_order]
    x_labels = np.asarray(sample_ids_from_file)[sample_order]
    ordered_indices = list(sample_order)
    x_labels_short = np.asarray([cancer_types[index] for index in ordered_indices])

    comp_colors = _load_component_colors(
        cfg.get("JSON_FILENAME_COMPONENT_COLORS", "conf/nmf_component_color_map.json"),
        n_comps,
        comp_order,
    )
    cancer_color_map = load_color_map(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS", ""))
    unknown_color = str(cfg.get("STRIP_UNKNOWN_COLOR", "#CCCCCC"))

    submitter_id_column = str(cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id"))
    submitter_ids = (
        merged_strip_metadata.get(submitter_id_column, merged_strip_metadata["patient_id"])
        .fillna(merged_strip_metadata["patient_id"])
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )
    cohorts = (
        merged_strip_metadata.get("cohort", pd.Series(["Unknown"] * len(sample_ids_from_file)))
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )
    winning_components = [f"Comp {index}" for index in np.argmax(H, axis=1).tolist()]

    subplot_titles = (
        "NMF Component Activities",
        "Proportional NMF Activity",
        "Dominant Component",
        *[strip_spec["label"] for strip_spec in strip_specs],
    )
    row_heights = [0.43, 0.18, 0.05, *([0.04] * len(strip_specs))]
    total_rows = len(row_heights)

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=[f"Comp {index}" for index in comp_order],
            z=H_sorted.T,
            colorscale="Turbo",
            showscale=False,
            hovertemplate="Sample: %{x}<br>Component: %{y}<br>Activity: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _add_proportional_bar_chart(fig, H_sorted, comp_colors, comp_order, x_values=x_labels)
    _add_component_strip(
        fig,
        H_ord,
        sample_order,
        comp_colors,
        n_comps,
        comp_order,
        x_values=x_labels,
    )

    for offset, strip_spec in enumerate(strip_specs, start=4):
        column = strip_spec["column"]
        ordered_values = [strip_values[column][index] for index in ordered_indices]
        ordered_colors, _ = _resolve_strip_colors(
            strip_spec,
            ordered_values,
            cancer_color_map,
            unknown_color,
        )
        ordered_submitter_ids = [submitter_ids[index] for index in ordered_indices]
        ordered_cohorts = [cohorts[index] for index in ordered_indices]
        ordered_winning_components = [winning_components[index] for index in ordered_indices]
        _add_metadata_annotation_strip(
            fig,
            x_labels,
            ordered_values,
            ordered_colors,
            strip_spec["label"],
            ordered_submitter_ids,
            ordered_cohorts,
            ordered_winning_components,
            row=offset,
        )

    _configure_layout(
        fig,
        n_comps,
        n_samples,
        comp_order,
        x_labels,
        x_labels_short,
        total_rows,
        len(strip_specs),
    )
    return fig


def create_linked_heatmap_figure(
    cfg_path: str | Path,
    sort_method: str = "component",
    highlight_sample_ids: list[str] | None = None,
    analysis_name: str | None = None,
) -> go.Figure:
    """Full NMF activity heatmap with a reactive selection-highlight strip.

    Same rows and metadata strips as :func:`create_heatmap_figure` (the published
    layout: ``Proportional NMF Activity``, ``Dominant Component`` and every
    ``HEATMAP_STRIPS`` annotation), but *all* samples stay visible and the
    current selection is marked by a ``Selection`` highlight strip rather than
    filtering the matrix down to the selected columns.

    Parameters
    ----------
    highlight_sample_ids:
        Sample-ID strings to highlight in the Selection strip. ``None`` or an
        empty list renders the strip blank (no samples highlighted).
    """
    cfg = resolve_analysis_cfg(cfg_path, analysis_name)
    heatmap_csv = cfg.get("HEATMAP_CSV_FILENAME", cfg.get("DEFAULT_CSV_FILENAME"))
    metadata_path = cfg.get("ANALYSIS_METADATA_FILENAME")

    # Always load every sample; selection is shown as a highlight, not a filter.
    H, sample_ids_from_file, cancer_types = _get_prepared_data(
        heatmap_csv,
        metadata_path=metadata_path,
        metadata_sample_id_column=str(cfg.get("ANALYSIS_METADATA_SAMPLE_ID_COLUMN", "sample_id")),
        cancer_type_column=str(cfg.get("ANALYSIS_CANCER_TYPE_COLUMN", "cancer_type")),
    )

    n_samples, n_comps = H.shape
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    strip_specs = cfg.get("HEATMAP_STRIPS", [])
    strip_values, merged_strip_metadata = _build_strip_values(sample_ids_from_file, cfg)
    sample_order = _get_sample_order(
        sort_method,
        H,
        sample_ids_from_file,
        cancer_types,
        strip_values,
    )

    H_sorted = H_ord[sample_order]
    x_labels = np.asarray(sample_ids_from_file)[sample_order]
    ordered_indices = list(sample_order)
    x_labels_short = np.asarray([cancer_types[index] for index in ordered_indices])

    comp_colors = _load_component_colors(
        cfg.get("JSON_FILENAME_COMPONENT_COLORS", "conf/nmf_component_color_map.json"),
        n_comps,
        comp_order,
    )
    cancer_color_map = load_color_map(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS", ""))
    unknown_color = str(cfg.get("STRIP_UNKNOWN_COLOR", "#CCCCCC"))

    submitter_id_column = str(cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id"))
    submitter_ids = (
        merged_strip_metadata.get(submitter_id_column, merged_strip_metadata["patient_id"])
        .fillna(merged_strip_metadata["patient_id"])
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )
    cohorts = (
        merged_strip_metadata.get("cohort", pd.Series(["Unknown"] * len(sample_ids_from_file)))
        .fillna("Unknown")
        .astype(str)
        .tolist()
    )
    winning_components = [f"Comp {index}" for index in np.argmax(H, axis=1).tolist()]

    # Selection highlight mask, in sorted order.
    highlight_set = {str(sample_id) for sample_id in (highlight_sample_ids or [])}
    highlight_mask = [1 if str(label) in highlight_set else 0 for label in x_labels]

    # Row layout: main heatmap, Selection strip, proportional bar, dominant
    # component, then one row per configured metadata strip.
    subplot_titles = (
        "NMF Component Activities",
        "Selection",
        "Proportional NMF Activity",
        "Dominant Component",
        *[strip_spec["label"] for strip_spec in strip_specs],
    )
    row_heights = [0.395, 0.035, 0.17, 0.05, *([0.04] * len(strip_specs))]
    total_rows = len(row_heights)

    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
    )

    fig.add_trace(
        go.Heatmap(
            x=x_labels,
            y=[f"Comp {index}" for index in comp_order],
            z=H_sorted.T,
            colorscale="Turbo",
            showscale=False,
            hovertemplate="Sample: %{x}<br>Component: %{y}<br>Activity: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _add_selection_strip(fig, x_labels, highlight_mask, row=2)
    _add_proportional_bar_chart(fig, H_sorted, comp_colors, comp_order, x_values=x_labels, row=3)
    _add_component_strip(
        fig,
        H_ord,
        sample_order,
        comp_colors,
        n_comps,
        comp_order,
        x_values=x_labels,
        row=4,
    )

    for offset, strip_spec in enumerate(strip_specs, start=5):
        column = strip_spec["column"]
        ordered_values = [strip_values[column][index] for index in ordered_indices]
        ordered_colors, _ = _resolve_strip_colors(
            strip_spec,
            ordered_values,
            cancer_color_map,
            unknown_color,
        )
        ordered_submitter_ids = [submitter_ids[index] for index in ordered_indices]
        ordered_cohorts = [cohorts[index] for index in ordered_indices]
        ordered_winning_components = [winning_components[index] for index in ordered_indices]
        _add_metadata_annotation_strip(
            fig,
            x_labels,
            ordered_values,
            ordered_colors,
            strip_spec["label"],
            ordered_submitter_ids,
            ordered_cohorts,
            ordered_winning_components,
            row=offset,
        )

    _configure_layout(
        fig,
        n_comps,
        n_samples,
        comp_order,
        x_labels,
        x_labels_short,
        total_rows,
        len(strip_specs) + 1,
    )
    return fig


def get_grandscatter_initial_projection(
    cfg_path: str | Path,
    analysis_name: str | None = None,
) -> dict:
    """Return the initial 2D projection grandscatter uses on first render."""
    df, axis_fields, _ = prepare_grandscatter_data(cfg_path, analysis_name=analysis_name)
    ndim = len(axis_fields)
    H = df[axis_fields].to_numpy(dtype=np.float64)

    scale = np.sqrt(2.0 / ndim)
    angles = 2.0 * np.pi * np.arange(ndim) / ndim
    basis_x = scale * np.cos(angles)
    basis_y = scale * np.sin(angles)

    x_proj = H @ basis_x
    y_proj = H @ basis_y
    max_r = float(np.max(np.sqrt(x_proj**2 + y_proj**2))) or 1.0
    x_n = (x_proj / max_r).tolist()
    y_n = (y_proj / max_r).tolist()
    dominant_idx = np.argmax(H, axis=1).tolist()
    dominant_comp = [axis_fields[index] for index in dominant_idx]
    cancer_types = df["cancer_type"].astype(str).tolist()

    points = [
        {"x": x_n[i], "y": y_n[i], "ct": cancer_types[i], "comp": dominant_comp[i]}
        for i in range(len(df))
    ]
    return {"points": points, "max_r": max_r}
