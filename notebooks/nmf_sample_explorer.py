# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "pyarrow>=18.0.0",
#     "plotly>=6.1.2",
#     "jupyter-scatter[all]>=0.22.0",
#     "grandscatter>=0.2.1",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    lib_src = repo_root / "lib" / "src"
    if lib_src.exists():
        lib_src_str = str(lib_src)
        if lib_src_str not in sys.path:
            sys.path.insert(0, lib_src_str)
    return


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    import pandas as pd

    from epifolio import ASSETS
    from epifolio.colors import load_color_map
    from epifolio.embedding import (
        available_color_fields,
        build_strip_color_lookup,
        prepare_embedding_for_display,
    )
    from epifolio.metadata import (
        build_metadata_views,
        metadata_presentation,
        metadata_tooltips as build_metadata_tooltips,
    )
    from epifolio.nmf import (
        build_metadata_strips,
        component_color_sequence,
        nmf_activity_heatmap,
        sample_sort_order,
    )

    return (
        ASSETS,
        available_color_fields,
        build_metadata_strips,
        build_metadata_tooltips,
        build_metadata_views,
        build_strip_color_lookup,
        component_color_sequence,
        json,
        load_color_map,
        metadata_presentation,
        mo,
        nmf_activity_heatmap,
        np,
        pd,
        prepare_embedding_for_display,
        sample_sort_order,
    )


@app.cell
def _():
    # Explicit data sources — visible data flow, no config indirection.
    BASE = "https://projects.abdenlab.org/itcr/epifolio"
    metadata_base = f"{BASE}/metadata"
    _nb = f"{BASE}/notebooks/nmf_explorer"
    analysis_sources = {
        "Canonical": {
            "nmf": f"{BASE}/embeddings/tcga.atac.nmf.sample.pq",
            "umap": f"{_nb}/canonical/tcga_atac_nmf_k24_umap_coordinates.csv",
            "metadata": f"{_nb}/canonical/tcga_atac_nmf_k24_metadata_sorted.csv",
        },
        "CN-aware adjusted": {
            "nmf": f"{BASE}/embeddings/adjusted.nmf.sample.pq",
            "umap": f"{_nb}/cn-aware-adjusted/adjusted_nmf_k24_umap_coordinates.csv",
            "metadata": f"{_nb}/cn-aware-adjusted/adjusted_nmf_k24_metadata_sorted.csv",
        },
        "CN-aware unadjusted": {
            "nmf": f"{BASE}/embeddings/confounded.nmf.sample.pq",
            "umap": f"{_nb}/cn-aware-unadjusted/confounded_nmf_k24_umap_coordinates.csv",
            "metadata": f"{_nb}/cn-aware-unadjusted/confounded_nmf_k24_metadata_sorted.csv",
        },
    }
    return analysis_sources, metadata_base


@app.cell
def _(analysis_name, analysis_sources, pd):
    # Load the selected analysis explicitly: raw NMF loadings -> proportions.
    _src = analysis_sources[analysis_name.value]
    _nmf = pd.read_parquet(_src["nmf"])
    activity_sample_ids = _nmf["sample"].astype(str).tolist()
    _raw = _nmf[[c for c in _nmf.columns if c.startswith("NMF")]].to_numpy(dtype=float)
    activity_H = _raw / _raw.sum(axis=1, keepdims=True)
    _cancer = pd.read_csv(_src["metadata"]).set_index("sample_id")["cancer_type"]
    activity_cancer_types = [str(_cancer.get(s, "Unknown")) for s in activity_sample_ids]
    return activity_H, activity_cancer_types, activity_sample_ids


@app.cell(hide_code=True)
def _(mo):
    title = mo.md("# Explore cancer sample cCRE activity signatures")
    title
    return


@app.cell
def _(analysis_sources, mo):
    analysis_options = list(analysis_sources)
    analysis_name = mo.ui.dropdown(
        options=analysis_options,
        value=analysis_options[0],
        label="NMF result set:",
    )
    return analysis_name, analysis_options


@app.cell
def _(mo):
    sort_method = mo.ui.dropdown(
        options=["component", "alphabetical", "cancer_type", "organ_system"],
        value="component",
        label="Sort by:",
    )
    return (sort_method,)


@app.cell
def _(ASSETS, json, metadata_base, pd):
    strip_cfg = json.loads((ASSETS / "conf" / "nmf_strips.json").read_text())
    strip_specs = strip_cfg.get("HEATMAP_STRIPS", [])
    strip_label_to_column = {
        spec.get("label", spec["column"]): spec["column"] for spec in strip_specs
    }
    strip_spec_by_column = {spec["column"]: spec for spec in strip_specs}
    strip_id_column = strip_cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id")
    strip_unknown_color = strip_cfg.get("STRIP_UNKNOWN_COLOR", "#CCCCCC")
    strip_metadata = (
        pd.read_csv(f"{metadata_base}/strip.corces_submitter_metadata.csv")
        .drop_duplicates(subset=[strip_id_column])
        .rename(columns={strip_id_column: "patient_id"})
    )
    return (
        strip_label_to_column,
        strip_metadata,
        strip_spec_by_column,
        strip_specs,
        strip_unknown_color,
    )


@app.cell
def _(
    ASSETS,
    available_color_fields,
    build_metadata_tooltips,
    load_color_map,
    metadata_base,
    metadata_presentation,
    pd,
    strip_label_to_column,
    strip_metadata,
):
    cancer_color_map = load_color_map(ASSETS / "conf" / "cancer_type_color_map.json")
    clinical_metadata = pd.read_csv(f"{metadata_base}/unified_clinical_metadata.csv").merge(
        strip_metadata,
        on="patient_id",
        how="left",
    )
    metadata_tooltips = build_metadata_tooltips(
        pd.read_csv(f"{metadata_base}/unified_clinical_metadata_dictionary.csv")
    )
    metadata_groups, base_tooltips, wrapped_columns, format_mapping = (
        metadata_presentation()
    )
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
def _(color_field_options, mo):
    color_by = mo.ui.dropdown(
        options=color_field_options,
        value="Cancer Type",
        label="Color by:",
    )
    return (color_by,)


@app.cell
def _(mo):
    shared_selected_ids, set_shared_selected_ids = mo.state([])
    return set_shared_selected_ids, shared_selected_ids


@app.cell
def _(mo):
    metadata_tab_name, set_metadata_tab_name = mo.state("Overview")
    return metadata_tab_name, set_metadata_tab_name


@app.cell
def _(np):
    def normalize_selection(selection):
        if selection is None:
            return []
        if isinstance(selection, np.ndarray):
            values = selection.tolist()
        else:
            values = list(selection)
        return sorted({int(value) for value in values})

    return (normalize_selection,)


@app.cell
def _(analysis_name, analysis_sources, pd):
    _src = analysis_sources[analysis_name.value]
    umap_data = pd.read_csv(_src["umap"]).rename(
        columns={"UMAP_1": "UMAP-1", "UMAP_2": "UMAP-2", "Sample_ID": "Sample ID"}
    )[["Sample ID", "UMAP-1", "UMAP-2"]]
    _cancer = pd.read_csv(_src["metadata"]).set_index("sample_id")["cancer_type"]
    umap_data["Cancer Type"] = (
        umap_data["Sample ID"].astype(str).map(_cancer).fillna("Unknown").astype(str)
    )
    return (umap_data,)


@app.cell
def _(analysis_name, set_shared_selected_ids):
    selected_analysis_name = analysis_name.value

    def clear_if_needed(current_ids):
        return [] if current_ids else current_ids

    set_shared_selected_ids(clear_if_needed)
    return (selected_analysis_name,)


@app.cell
def _(umap_data):
    sample_id_to_index = {
        str(sample_id): int(index)
        for index, sample_id in enumerate(umap_data["Sample ID"].astype(str).tolist())
    }
    return (sample_id_to_index,)


@app.cell
def _(
    cancer_color_map,
    clinical_metadata,
    color_by,
    color_field_to_column,
    prepare_embedding_for_display,
    umap_data,
):
    selected_color_column = color_field_to_column[color_by.value]
    scatter_display_df, base_sample_color_map = prepare_embedding_for_display(
        umap_data,
        clinical_metadata,
        selected_color_column,
        cancer_color_map,
    )
    return base_sample_color_map, scatter_display_df, selected_color_column


@app.cell
def _(
    build_strip_color_lookup,
    base_sample_color_map,
    cancer_color_map,
    scatter_display_df,
    selected_color_column,
    strip_spec_by_column,
    strip_unknown_color,
):
    sample_color_map = base_sample_color_map
    if selected_color_column in strip_spec_by_column:
        sample_color_map = build_strip_color_lookup(
            scatter_display_df["Color Label"].tolist(),
            selected_color_column,
            strip_spec_by_column,
            cancer_color_map,
            strip_unknown_color,
        )
    return (sample_color_map,)


@app.cell
def _(color_by, sample_color_map, scatter_display_df):
    import jscatter

    tooltip_fields = ["Sample ID", "Cancer Type", "Color Label"]
    scatter = jscatter.Scatter(
        data=scatter_display_df,
        x="UMAP-1",
        y="UMAP-2",
        color_by="Color Label",
        color_map=sample_color_map,
        height=600,
        width=600,
        lasso_callback=True,
        selection_mode="lasso",
    )
    scatter.tooltip(enable=True, properties=tooltip_fields)
    scatter.size(default=5)
    # Assign (rather than leave a bare expression) so marimo does not render this
    # cell's scatter on its own — the UMAP is only shown via the final layout.
    _ = scatter.options(
        {
            "aspectRatio": 1.0,
            "regl_scatterplot_options": {
                "showLegend": True,
                "xAxis": {"showGrid": True},
                "yAxis": {"showGrid": True},
                "title": f"UMAP of NMF Components (colored by {color_by.value})",
            },
        }
    )
    return (scatter,)


@app.cell
def _(scatter):
    # Display the raw jscatter widget (not wrapped in mo.ui.anywidget) so its
    # lasso selection fires the trait observer registered below — matching the
    # grandscatter wiring and the proven latest_design layout.
    scatter_widget = scatter.widget
    return (scatter_widget,)


@app.cell
def _(normalize_selection, scatter_widget, set_shared_selected_ids):
    def sync_scatter_selection(change):
        incoming_ids = normalize_selection(change.get("new"))

        def update_if_changed(current_ids):
            normalized_current = normalize_selection(current_ids)
            return current_ids if normalized_current == incoming_ids else incoming_ids

        set_shared_selected_ids(update_if_changed)

    scatter_widget.observe(sync_scatter_selection, names="selection")
    return


@app.cell
def _(normalize_selection, scatter_widget, shared_selected_ids):
    scatter_target_ids = shared_selected_ids()
    if normalize_selection(scatter_widget.selection) != scatter_target_ids:
        scatter_widget.selection = scatter_target_ids
    return


@app.cell
def _(
    ASSETS,
    activity_H,
    activity_cancer_types,
    activity_sample_ids,
    build_metadata_strips,
    cancer_color_map,
    component_color_sequence,
    load_color_map,
    mo,
    nmf_activity_heatmap,
    sample_sort_order,
    scatter_display_df,
    shared_selected_ids,
    sort_method,
    strip_metadata,
    strip_specs,
    strip_unknown_color,
):
    # The notebook orchestrates; nmf_activity_heatmap just renders.
    activity_strips = build_metadata_strips(
        strip_specs,
        activity_sample_ids,
        strip_metadata,
        id_column="patient_id",
        cancer_color_map=cancer_color_map,
        unknown_color=strip_unknown_color,
    )
    _organ_strip = next((s for s in activity_strips if s.label == "Organ System"), None)
    activity_order = sample_sort_order(
        sort_method.value,
        activity_H,
        activity_sample_ids,
        cancer_types=activity_cancer_types,
        group_values=_organ_strip.values if _organ_strip else None,
    )

    component_colors = component_color_sequence(
        load_color_map(ASSETS / "conf" / "nmf_component_color_map.json"),
        activity_H.shape[1],
    )

    selection = shared_selected_ids()
    if selection:
        _highlighted = set(
            scatter_display_df.iloc[selection]["Sample ID"].astype(str).tolist()
        )
        caption = (
            f"Highlighting {len(selection)} selected samples; "
            "all samples remain visible in the heatmap."
        )
    else:
        _highlighted = set()
        caption = (
            "Showing all samples. Select samples in the scatter, heatmap, "
            "grandscatter, or metadata table to highlight them in the linked views."
        )
    highlight_mask = [1 if sid in _highlighted else 0 for sid in activity_sample_ids]

    fig = nmf_activity_heatmap(
        activity_H,
        activity_sample_ids,
        component_colors=component_colors,
        sample_order=activity_order,
        column_tick_labels=activity_cancer_types,
        strips=activity_strips,
        highlight_mask=highlight_mask,
    )
    heatmap_plot = mo.ui.plotly(fig)
    return caption, heatmap_plot


@app.cell
def _(heatmap_plot, sample_id_to_index):
    def extract_selected_sample_ids(plot_points):
        seen = set()
        for point in plot_points or []:
            if not isinstance(point, dict):
                continue
            sample_id = point.get("x")
            if sample_id is None:
                continue
            sample_id = str(sample_id)
            if sample_id not in sample_id_to_index:
                continue
            seen.add(sample_id)
        return sorted(seen, key=lambda sample_id: sample_id_to_index[sample_id])

    heatmap_selected_sample_ids = extract_selected_sample_ids(heatmap_plot.value)
    return extract_selected_sample_ids, heatmap_selected_sample_ids


@app.cell
def _(heatmap_selected_sample_ids, sample_id_to_index, set_shared_selected_ids):
    incoming_heatmap_ids = sorted(
        {sample_id_to_index[sample_id] for sample_id in heatmap_selected_sample_ids}
    )

    if heatmap_selected_sample_ids:
        def heatmap_update_selection(current_ids):
            return current_ids if current_ids == incoming_heatmap_ids else incoming_heatmap_ids

        set_shared_selected_ids(heatmap_update_selection)
    return


@app.cell
def _(
    activity_H,
    activity_sample_ids,
    pd,
    sample_color_map,
    scatter_display_df,
):
    from grandscatter import Scatter

    axis_fields = [f"Comp_{index}" for index in range(activity_H.shape[1])]
    gs_df = pd.DataFrame(activity_H, columns=axis_fields)
    gs_df.insert(0, "Sample ID", activity_sample_ids)
    # Align rows to the UMAP scatter's order so a selection index refers to the
    # same sample in both widgets (the NMF parquet and UMAP files are sorted
    # differently). Merging onto scatter_display_df (left) preserves its order.
    grandscatter_df = scatter_display_df[["Sample ID", "Color Label"]].merge(
        gs_df,
        on="Sample ID",
        how="left",
    )
    grandscatter_df["Color Label"] = (
        grandscatter_df["Color Label"].fillna("Unknown").astype(str).astype("category")
    )
    # Display the raw grandscatter widget so its box-select fires the trait
    # observer registered below (no mo.ui.anywidget wrapper in between).
    gs_widget = Scatter(
        grandscatter_df[[*axis_fields, "Color Label"]],
        axis_fields=axis_fields,
        label_field="Color Label",
        label_colors=sample_color_map,
        base_point_size=6,
    )
    return axis_fields, grandscatter_df, gs_widget


@app.cell
def _(gs_widget, normalize_selection, set_shared_selected_ids):
    def sync_grandscatter_selection(change):
        incoming_ids = normalize_selection(change.get("new"))

        def update_if_changed(current_ids):
            normalized_current = normalize_selection(current_ids)
            return current_ids if normalized_current == incoming_ids else incoming_ids

        set_shared_selected_ids(update_if_changed)

    gs_widget.observe(sync_grandscatter_selection, names="selected_points")
    return


@app.cell
def _(gs_widget, normalize_selection, shared_selected_ids):
    grandscatter_target_ids = shared_selected_ids()
    if normalize_selection(gs_widget.selected_points) != grandscatter_target_ids:
        gs_widget.selected_points = grandscatter_target_ids
    return


@app.cell
def _(build_metadata_views, scatter_display_df, shared_selected_ids):
    selected_indices = shared_selected_ids()
    current_selected_sample_ids = (
        scatter_display_df.iloc[selected_indices]["Sample ID"].astype(str).tolist()
        if selected_indices
        else []
    )
    selected_metadata_df, duplicate_patients_df, metadata_summary = build_metadata_views(
        scatter_display_df,
        current_selected_sample_ids,
    )
    return (
        current_selected_sample_ids,
        duplicate_patients_df,
        metadata_summary,
        selected_metadata_df,
    )


@app.cell
def _(pd):
    def extract_selection_indices(table_value):
        if table_value is None:
            return []

        if isinstance(table_value, pd.DataFrame):
            if "selection_index" not in table_value.columns:
                return []
            values = table_value["selection_index"].dropna().tolist()
        elif isinstance(table_value, list):
            values = []
            for row in table_value:
                if not isinstance(row, dict):
                    continue
                value = row.get("selection_index")
                if value is not None and pd.notna(value):
                    values.append(value)
        else:
            return []

        return sorted({int(value) for value in values})

    return (extract_selection_indices,)


@app.cell
def _(
    analysis_name,
    base_tooltips,
    caption,
    color_by,
    duplicate_patients_df,
    extract_selection_indices,
    format_mapping,
    gs_widget,
    heatmap_plot,
    metadata_groups,
    metadata_summary,
    metadata_tab_name,
    metadata_tooltips,
    mo,
    normalize_selection,
    scatter_widget,
    selected_metadata_df,
    set_metadata_tab_name,
    set_shared_selected_ids,
    sort_method,
    wrapped_columns,
):
    header_tooltips = {**metadata_tooltips, **base_tooltips}
    table_initial_selection = (
        list(range(len(selected_metadata_df))) if metadata_summary["is_filtered"] else None
    )

    def make_table(
        dataframe,
        columns,
        label,
        page_size=12,
        max_height=360,
        *,
        selection=None,
        initial_selection=None,
    ):
        available_columns = [column for column in columns if column in dataframe.columns]

        def sync_table_selection(table_value):
            selected_indices = extract_selection_indices(table_value)

            def update_if_changed(current_ids):
                normalized_current = normalize_selection(current_ids)
                return current_ids if normalized_current == selected_indices else selected_indices

            set_shared_selected_ids(update_if_changed)

        return mo.ui.table(
            dataframe[available_columns],
            selection=selection,
            initial_selection=initial_selection,
            pagination=True,
            page_size=page_size,
            show_data_types=False,
            show_column_summaries=False,
            freeze_columns_left=[
                column
                for column in ["selection_index", "Sample ID", "patient_id"]
                if column in available_columns
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
            max_height=max_height,
            label=label,
            on_change=sync_table_selection if selection is not None else None,
        )

    metadata_tables = {
        name: make_table(
            selected_metadata_df,
            columns,
            f"{name} metadata",
            selection="multi",
            initial_selection=table_initial_selection,
        )
        for name, columns in metadata_groups.items()
    }
    metadata_tables["All metadata"] = make_table(
        selected_metadata_df,
        list(selected_metadata_df.columns),
        "All linked metadata",
        page_size=10,
        max_height=420,
        selection="multi",
        initial_selection=table_initial_selection,
    )

    active_metadata_tab = metadata_tab_name()
    if active_metadata_tab not in metadata_tables:
        active_metadata_tab = next(iter(metadata_tables))

    metadata_tabs = mo.ui.tabs(
        metadata_tables,
        value=active_metadata_tab,
        on_change=lambda selected_tab: set_metadata_tab_name(
            lambda current_tab: (
                current_tab if current_tab == selected_tab else selected_tab
            )
        ),
    )

    duplicate_patients_view = (
        mo.accordion(
            {
                "Patients with multiple ATAC-seq samples in this view": make_table(
                    duplicate_patients_df,
                    ["patient_id", "sample_count", "cancer_types", "sample_ids"],
                    "Repeated patients",
                    page_size=8,
                    max_height=260,
                )
            },
            lazy=True,
        )
        if not duplicate_patients_df.empty
        else mo.md("_No repeated patient IDs in the current view._")
    )

    metadata_scope = (
        "Current selection" if metadata_summary["is_filtered"] else "All assay samples"
    )
    metadata_caption = (
        f"{metadata_summary['matched_sample_count']}/{metadata_summary['sample_count']} "
        "rows matched the unified clinical metadata table. "
        f"Cancer types in view: {metadata_summary['cancer_types_label']}."
    )
    embedding_views = mo.hstack(
        [
            scatter_widget,
            gs_widget,
        ],
        widths=[0.42, 0.58],
    )

    mo.vstack(
        [
            mo.hstack([analysis_name, color_by], widths=[0.5, 0.5]),
            mo.md("### Embedding scatter plots"),
            mo.md(
                f"**{analysis_name.value}** result set. {caption} "
                "Use the lasso tool in the UMAP or grandscatter view to keep the "
                "linked views synchronized."
            ),
            embedding_views,
            mo.md("### Component scores"),
            mo.md(
                "_Drag axis handles in grandscatter to rotate the 24-dimensional "
                "NMF proportion space. Use box-select on the heatmap to link back "
                "to the scatter plots and metadata tables._"
            ),
            sort_method,
            heatmap_plot,
            mo.md("### Linked Clinical Metadata"),
            mo.md(f"_{metadata_scope}: {metadata_caption}_"),
            mo.hstack(
                [
                    mo.stat(
                        metadata_summary["sample_count"],
                        label="Samples in view",
                        caption="Assay rows linked to the current selection.",
                        bordered=True,
                    ),
                    mo.stat(
                        metadata_summary["unique_patient_count"],
                        label="Unique patients",
                        caption="Distinct `patient_id` values in view.",
                        bordered=True,
                    ),
                    mo.stat(
                        metadata_summary["cancer_type_count"],
                        label="Cancer types",
                        caption="Distinct cancer labels represented in view.",
                        bordered=True,
                    ),
                    mo.stat(
                        metadata_summary["duplicate_patient_count"],
                        label="Repeated patients",
                        caption="Patients appearing in more than one assay row.",
                        bordered=True,
                    ),
                ],
                widths=[1, 1, 1, 1],
            ),
            duplicate_patients_view,
            metadata_tabs,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
