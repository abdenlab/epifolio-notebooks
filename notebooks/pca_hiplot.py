# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "anywidget>=0.9.18",
#     "wigglystuff>=0.5.9",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "pyarrow>=18.0.0",
#     "plotly>=6.1.2",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json

    import marimo as mo
    import pandas as pd
    from wigglystuff import ParallelCoordinates

    from epifolio import ASSETS
    from epifolio.colors import load_color_map
    from epifolio.embedding import (
        available_color_fields,
        build_strip_color_lookup,
        coerce_visible_features,
        get_feature_columns,
        prepare_embedding_for_display,
        prepare_embedding_sample_frame,
    )
    from epifolio.metadata import (
        build_metadata_views,
        metadata_presentation,
        metadata_tooltips as build_metadata_tooltips,
        subset_by_sample_ids,
    )

    return (
        ASSETS,
        ParallelCoordinates,
        available_color_fields,
        build_metadata_tooltips,
        build_metadata_views,
        build_strip_color_lookup,
        coerce_visible_features,
        get_feature_columns,
        json,
        load_color_map,
        metadata_presentation,
        mo,
        pd,
        prepare_embedding_for_display,
        prepare_embedding_sample_frame,
        subset_by_sample_ids,
    )


@app.cell(hide_code=True)
def _(mo):
    title = mo.md("# Explore published ITCR PCA outcomes in HiPlot")
    title
    return


@app.cell
def _(ASSETS, json, pd):
    _METADATA = "https://projects.abdenlab.org/itcr/epifolio/metadata"
    strip_cfg = json.loads((ASSETS / "conf" / "nmf_strips.json").read_text())
    strip_specs = strip_cfg.get("HEATMAP_STRIPS", [])
    strip_label_to_column = {
        spec.get("label", spec["column"]): spec["column"] for spec in strip_specs
    }
    strip_spec_by_column = {spec["column"]: spec for spec in strip_specs}
    strip_id_column = strip_cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id")
    strip_metadata = (
        pd.read_csv(f"{_METADATA}/strip.corces_submitter_metadata.csv")
        .drop_duplicates(subset=[strip_id_column])
        .rename(columns={strip_id_column: "patient_id"})
    )
    strip_unknown_color = strip_cfg.get("STRIP_UNKNOWN_COLOR", "#CCCCCC")
    return strip_label_to_column, strip_metadata, strip_spec_by_column, strip_unknown_color


@app.cell
def _():
    _EMBEDDINGS = "https://projects.abdenlab.org/itcr/epifolio/embeddings"
    pca_result_options = {
        "Canonical PCA": f"{_EMBEDDINGS}/tcga.atac.pca.sample.pq",
        "CN-aware adjusted PCA": f"{_EMBEDDINGS}/adjusted.pca.sample.pq",
        "CN-aware confounded PCA": f"{_EMBEDDINGS}/confounded.pca.sample.pq",
    }
    pca_result_labels = list(pca_result_options)
    return pca_result_labels, pca_result_options


@app.cell
def _(
    ASSETS,
    available_color_fields,
    build_metadata_tooltips,
    load_color_map,
    metadata_presentation,
    pd,
    strip_label_to_column,
    strip_metadata,
):
    _METADATA = "https://projects.abdenlab.org/itcr/epifolio/metadata"
    cancer_color_map = load_color_map(ASSETS / "conf" / "cancer_type_color_map.json")
    clinical_metadata = pd.read_csv(f"{_METADATA}/unified_clinical_metadata.csv").merge(
        strip_metadata,
        on="patient_id",
        how="left",
    )
    metadata_tooltips = build_metadata_tooltips(
        pd.read_csv(f"{_METADATA}/unified_clinical_metadata_dictionary.csv")
    )
    metadata_groups, base_tooltips, wrapped_columns, format_mapping = metadata_presentation()
    base_fields = available_color_fields(clinical_metadata)
    color_field_options = []
    for option in [
        "Cancer Type",
        *[field for field in base_fields if field != "Cancer Type"],
        *strip_label_to_column.keys(),
    ]:
        if option not in color_field_options:
            color_field_options.append(option)
    color_field_to_column = {field: field for field in base_fields} | strip_label_to_column
    return (
        base_tooltips,
        cancer_color_map,
        clinical_metadata,
        color_field_options,
        color_field_to_column,
        format_mapping,
        metadata_groups,
        metadata_tooltips,
        wrapped_columns,
    )


@app.cell
def _(color_field_options, mo, pca_result_labels):
    pca_result_dropdown = mo.ui.dropdown(
        options=pca_result_labels,
        value=pca_result_labels[0],
        label="PCA result set:",
    )
    color_by_dropdown = mo.ui.dropdown(
        options=color_field_options,
        value="Cancer Type",
        label="Color samples by:",
    )
    return color_by_dropdown, pca_result_dropdown


@app.cell
def _(pd, prepare_embedding_sample_frame, pca_result_dropdown, pca_result_options):
    pca_embedding_df = prepare_embedding_sample_frame(
        pd.read_parquet(pca_result_options[pca_result_dropdown.value])
    )
    return (pca_embedding_df,)


@app.cell
def _(
    build_strip_color_lookup,
    cancer_color_map,
    clinical_metadata,
    color_by_dropdown,
    color_field_to_column,
    pca_embedding_df,
    prepare_embedding_for_display,
    strip_spec_by_column,
    strip_unknown_color,
):
    selected_color_column = color_field_to_column[color_by_dropdown.value]
    pca_display_df, pca_label_colors = prepare_embedding_for_display(
        pca_embedding_df,
        clinical_metadata,
        selected_color_column,
        cancer_color_map,
    )
    if selected_color_column in strip_spec_by_column:
        pca_label_colors = build_strip_color_lookup(
            pca_display_df["Color Label"].tolist(),
            selected_color_column,
            strip_spec_by_column,
            cancer_color_map,
            strip_unknown_color,
        )
    return pca_display_df, pca_label_colors


@app.cell
def _(coerce_visible_features, get_feature_columns, pca_display_df):
    pca_feature_columns = get_feature_columns(pca_display_df)
    default_features = pca_feature_columns[: min(10, len(pca_feature_columns))]
    optional_metadata_columns = [
        column
        for column in [
            "tissue_or_organ_of_origin",
            "primary_diagnosis",
            "molecular_subtype",
            "patient_id",
        ]
        if column in pca_display_df.columns
    ]
    hiplot_column_options = [*pca_feature_columns, *optional_metadata_columns]
    default_hiplot_features = coerce_visible_features(
        hiplot_column_options,
        default_features,
        limit=10,
    )
    return default_hiplot_features, hiplot_column_options


@app.cell
def _(default_hiplot_features, hiplot_column_options, mo):
    hiplot_feature_picker = mo.ui.multiselect(
        options=hiplot_column_options,
        value=default_hiplot_features,
        label="Visible HiPlot columns:",
    )
    return (hiplot_feature_picker,)


@app.cell
def _(coerce_visible_features, hiplot_column_options, hiplot_feature_picker):
    selected_hiplot_features = coerce_visible_features(
        hiplot_column_options,
        list(hiplot_feature_picker.value),
        limit=10,
    )
    return (selected_hiplot_features,)


@app.cell
def _(ParallelCoordinates, mo, pca_display_df, pca_label_colors, selected_hiplot_features):
    # Parallel-coordinates view via wigglystuff's HiPlot port — a real anywidget,
    # so it links like the other plots (no iframe, no JS bridge).
    # The color_by column must be in the data (ignore would strip it out and
    # break coloring), so "Color Label" rides along as the leftmost dimension.
    # Selection maps by row index into pca_display_df, so Sample ID isn't needed.
    _axis_columns = [
        c for c in selected_hiplot_features
        if c in pca_display_df.columns and c != "Color Label"
    ]
    _frame = pca_display_df[["Color Label", *_axis_columns]]
    pca_widget = mo.ui.anywidget(
        ParallelCoordinates(
            _frame,
            color_by="Color Label",
            color_map=pca_label_colors,
            height=820,
        )
    )

    # ParallelCoordinates has no built-in legend; render one from the color map
    # (only categories present in the data, "Unknown" last).
    _present = {str(label) for label in pca_display_df["Color Label"]} & set(pca_label_colors)
    _ordered = sorted(label for label in _present if label != "Unknown")
    if "Unknown" in _present:
        _ordered.append("Unknown")
    pca_legend = mo.Html(
        '<div style="display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;'
        'padding:6px 2px;align-items:center">'
        + "".join(
            '<span style="display:inline-flex;align-items:center">'
            f'<span style="width:12px;height:12px;border-radius:2px;background:'
            f'{pca_label_colors[label]};display:inline-block;margin-right:5px;'
            'border:1px solid rgba(0,0,0,0.15)"></span>'
            f"{label}</span>"
            for label in _ordered
        )
        + "</div>"
    )
    return pca_legend, pca_widget


@app.cell
def _(pca_display_df, pca_widget):
    # Map brushed/selected row indices back to Sample IDs. Prefer the axis-brush
    # filter when it narrows the set, else the active selection.
    _ids_in_order = pca_display_df["Sample ID"].astype(str).tolist()
    _total = len(_ids_in_order)
    _state = pca_widget.value
    filtered_idx = [i for i in (_state.get("filtered_indices") or []) if 0 <= i < _total]
    selected_idx = [i for i in (_state.get("selected_indices") or []) if 0 <= i < _total]

    if filtered_idx and len(filtered_idx) < _total:
        selected_hiplot_sample_ids = [_ids_in_order[i] for i in filtered_idx]
    elif selected_idx and len(selected_idx) < _total:
        selected_hiplot_sample_ids = [_ids_in_order[i] for i in selected_idx]
    else:
        selected_hiplot_sample_ids = []

    hiplot_selection_summary = {
        "selected_count": len(selected_idx),
        "filtered_count": len(filtered_idx) if filtered_idx else _total,
        "is_filtered": bool(filtered_idx) and len(filtered_idx) < _total,
    }
    return hiplot_selection_summary, selected_hiplot_sample_ids


@app.cell
def _(build_metadata_views, pca_display_df, selected_hiplot_sample_ids):
    filtered_metadata_df, duplicate_patients_df, metadata_summary = build_metadata_views(
        pca_display_df,
        selected_hiplot_sample_ids,
    )
    return duplicate_patients_df, filtered_metadata_df, metadata_summary


@app.cell
def _(
    base_tooltips,
    color_by_dropdown,
    duplicate_patients_df,
    filtered_metadata_df,
    format_mapping,
    hiplot_feature_picker,
    metadata_groups,
    metadata_summary,
    metadata_tooltips,
    mo,
    pca_legend,
    pca_result_dropdown,
    pca_widget,
    hiplot_selection_summary,
    selected_hiplot_sample_ids,
    wrapped_columns,
):
    header_tooltips = {**metadata_tooltips, **base_tooltips}

    def make_table(dataframe, columns, label):
        available_columns = [column for column in columns if column in dataframe.columns]
        return mo.ui.table(
            dataframe[available_columns],
            pagination=True,
            page_size=14,
            show_data_types=False,
            show_column_summaries=False,
            freeze_columns_left=[
                column for column in ["selection_index", "Sample ID", "patient_id"] if column in available_columns
            ],
            wrapped_columns=[
                column for column in wrapped_columns if column in available_columns
            ],
            header_tooltip={
                column: header_tooltips[column]
                for column in available_columns
                if column in header_tooltips
            },
            format_mapping={
                column: format_mapping[column]
                for column in available_columns
                if column in format_mapping
            },
            max_height=420,
            label=label,
        )

    metadata_tabs = mo.ui.tabs(
        {
            name: make_table(filtered_metadata_df, columns, f"{name} metadata")
            for name, columns in metadata_groups.items()
        }
        | {
            "All metadata": make_table(
                filtered_metadata_df,
                list(filtered_metadata_df.columns),
                "All PCA metadata",
            )
        }
    )

    duplicate_view = (
        mo.accordion(
            {
                "Patients with multiple ATAC-seq samples in this view": make_table(
                    duplicate_patients_df,
                    ["patient_id", "sample_count", "cancer_types", "sample_ids"],
                    "Repeated patients",
                )
            },
            lazy=True,
        )
        if not duplicate_patients_df.empty
        else mo.md("_No repeated patient IDs in the current view._")
    )

    selection_summary = (
        "Showing all samples in the current PCA result."
        if not selected_hiplot_sample_ids
        else f"Showing {len(selected_hiplot_sample_ids)} selected samples from HiPlot."
    )
    if hiplot_selection_summary["is_filtered"]:
        selection_summary = (
            f"Showing {len(selected_hiplot_sample_ids)} filtered samples from HiPlot "
            f"({hiplot_selection_summary['selected_count']} explicitly selected)."
        )

    mo.vstack(
        [
            mo.hstack([pca_result_dropdown, color_by_dropdown], widths=[1, 1]),
            hiplot_feature_picker,
            pca_widget,
            pca_legend,
            mo.md("### Expanded metadata table"),
            mo.md(selection_summary),
            mo.hstack(
                [
                    mo.stat(metadata_summary["sample_count"], label="Samples shown", bordered=True),
                    mo.stat(metadata_summary["unique_patient_count"], label="Unique patients shown", bordered=True),
                    mo.stat(metadata_summary["cancer_type_count"], label="Cancer types shown", bordered=True),
                ],
                widths=[1, 1, 1],
            ),
            metadata_tabs,
            duplicate_view,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
