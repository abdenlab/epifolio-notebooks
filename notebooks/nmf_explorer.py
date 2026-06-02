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
    import marimo as mo
    import numpy as np
    import pandas as pd

    from epifolio import ASSETS
    from epifolio.data_utils import (
        get_available_analyses,
        load_analysis_umap_data,
        prepare_grandscatter_data,
        resolve_analysis_cfg,
    )
    from epifolio.itcr_embedding_utils import (
        available_color_fields,
        build_metadata_views,
        build_strip_color_lookup,
        load_cancer_color_map,
        load_clinical_metadata,
        load_metadata_tooltips,
        metadata_presentation,
        prepare_embedding_for_display,
    )
    from epifolio.upt_heatmap import (
        create_linked_heatmap_figure,
        get_grandscatter_initial_projection,
    )

    return (
        ASSETS,
        available_color_fields,
        build_metadata_views,
        build_strip_color_lookup,
        create_linked_heatmap_figure,
        get_available_analyses,
        get_grandscatter_initial_projection,
        load_analysis_umap_data,
        load_cancer_color_map,
        load_clinical_metadata,
        load_metadata_tooltips,
        metadata_presentation,
        mo,
        np,
        pd,
        prepare_embedding_for_display,
        prepare_grandscatter_data,
        resolve_analysis_cfg,
    )


@app.cell
def _(ASSETS):
    cfg_path = ASSETS / "conf" / "upt_pub_nmf_config.json"
    return (cfg_path,)


@app.cell(hide_code=True)
def _(mo):
    title = mo.md("# Explore cancer cCRE signatures")
    title
    return


@app.cell
def _(cfg_path, get_available_analyses, mo):
    analysis_options = get_available_analyses(cfg_path)
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
def _(cfg_path, pd, resolve_analysis_cfg):
    base_cfg = resolve_analysis_cfg(cfg_path)
    strip_specs = base_cfg.get("HEATMAP_STRIPS", [])
    strip_label_to_column = {
        spec.get("label", spec["column"]): spec["column"] for spec in strip_specs
    }
    strip_spec_by_column = {spec["column"]: spec for spec in strip_specs}
    strip_id_column = base_cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id")
    strip_metadata = (
        pd.read_csv(base_cfg["STRIP_METADATA_FILENAME"])
        .drop_duplicates(subset=[strip_id_column])
        .rename(columns={strip_id_column: "patient_id"})
    )
    return base_cfg, strip_label_to_column, strip_metadata, strip_spec_by_column


@app.cell
def _(
    available_color_fields,
    load_cancer_color_map,
    load_clinical_metadata,
    load_metadata_tooltips,
    metadata_presentation,
    strip_label_to_column,
    strip_metadata,
):
    cancer_color_map = load_cancer_color_map()
    clinical_metadata = load_clinical_metadata().merge(
        strip_metadata,
        on="patient_id",
        how="left",
    )
    metadata_tooltips = load_metadata_tooltips()
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
def _(analysis_name, load_analysis_umap_data, cfg_path):
    umap_data = load_analysis_umap_data(cfg_path, analysis_name=analysis_name.value)
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
    base_cfg,
    build_strip_color_lookup,
    base_sample_color_map,
    cancer_color_map,
    scatter_display_df,
    selected_color_column,
    strip_spec_by_column,
):
    sample_color_map = base_sample_color_map
    if selected_color_column in strip_spec_by_column:
        sample_color_map = build_strip_color_lookup(
            scatter_display_df["Color Label"].tolist(),
            selected_color_column,
            strip_spec_by_column,
            cancer_color_map,
            base_cfg.get("STRIP_UNKNOWN_COLOR", "#CCCCCC"),
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

    _previous_observer = getattr(scatter_widget, "_marimo_selection_observer", None)
    if _previous_observer is not None:
        scatter_widget.unobserve(_previous_observer, names="selection")

    scatter_widget.observe(sync_scatter_selection, names="selection")
    scatter_widget._marimo_selection_observer = sync_scatter_selection
    return


@app.cell
def _(normalize_selection, scatter_widget, shared_selected_ids):
    scatter_target_ids = shared_selected_ids()
    if normalize_selection(scatter_widget.selection) != scatter_target_ids:
        scatter_widget.selection = scatter_target_ids
    return


@app.cell
def _(
    analysis_name,
    cfg_path,
    create_linked_heatmap_figure,
    mo,
    scatter_display_df,
    shared_selected_ids,
    sort_method,
):
    selection = shared_selected_ids()
    if selection:
        highlight_sample_ids = (
            scatter_display_df.iloc[selection]["Sample ID"].astype(str).tolist()
        )
        caption = (
            f"Highlighting {len(selection)} selected samples; "
            "all samples remain visible in the heatmap."
        )
    else:
        highlight_sample_ids = None
        caption = (
            "Showing all samples. Select samples in the scatter, heatmap, "
            "grandscatter, or metadata table to highlight them in the linked views."
        )

    fig = create_linked_heatmap_figure(
        cfg_path=cfg_path,
        sort_method=sort_method.value,
        highlight_sample_ids=highlight_sample_ids,
        analysis_name=analysis_name.value,
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
    analysis_name,
    cfg_path,
    prepare_grandscatter_data,
    sample_color_map,
    scatter_display_df,
):
    from grandscatter import Scatter

    gs_df, axis_fields, _ = prepare_grandscatter_data(
        cfg_path,
        analysis_name=analysis_name.value,
    )
    grandscatter_df = gs_df.rename(columns={"sample_id": "Sample ID"}).merge(
        scatter_display_df[["Sample ID", "Color Label"]],
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
def _(analysis_name, cfg_path, get_grandscatter_initial_projection, mo):
    import json

    projection = get_grandscatter_initial_projection(
        cfg_path,
        analysis_name=analysis_name.value,
    )
    points_json = json.dumps(projection["points"])
    hover_overlay = mo.Html(
        f"""
<style>
  .gs-tooltip {{
    position: fixed;
    background: rgba(15, 15, 15, 0.82);
    color: #f0f0f0;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 12px;
    font-family: monospace;
    pointer-events: none;
    display: none;
    z-index: 9999;
    white-space: nowrap;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
  }}
</style>
<div class="gs-tooltip" id="gs-hover-tip"></div>
<script type="module">
const POINTS = {points_json};
const tip = document.getElementById("gs-hover-tip");

function nearest(nx, ny) {{
  let best = null;
  let bestDistance = Infinity;
  for (let i = 0; i < POINTS.length; i += 1) {{
    const point = POINTS[i];
    const distance = (point.x - nx) ** 2 + (point.y - ny) ** 2;
    if (distance < bestDistance) {{
      bestDistance = distance;
      best = point;
    }}
  }}
  return best ? {{ ...best, dist: bestDistance }} : null;
}}

function attach() {{
  const widget = document.querySelector(".grandscatter-widget");
  if (!widget) {{
    requestAnimationFrame(attach);
    return;
  }}
  const canvas = widget.querySelector("canvas");
  if (!canvas) {{
    requestAnimationFrame(attach);
    return;
  }}

  canvas.addEventListener("mousemove", (event) => {{
    const rect = canvas.getBoundingClientRect();
    const nx = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const ny = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    const hit = nearest(nx, ny);
    if (hit && hit.dist < 0.03) {{
      tip.innerHTML = "<b>" + hit.ct + "</b> &nbsp;|&nbsp; " + hit.comp;
      tip.style.display = "block";
      tip.style.left = event.clientX + 14 + "px";
      tip.style.top = event.clientY - 36 + "px";
    }} else {{
      tip.style.display = "none";
    }}
  }});

  canvas.addEventListener("mouseleave", () => {{
    tip.style.display = "none";
  }});
}}

attach();
</script>
"""
    )
    return (hover_overlay,)


@app.cell
def _(gs_widget, normalize_selection, set_shared_selected_ids):
    def sync_grandscatter_selection(change):
        incoming_ids = normalize_selection(change.get("new"))

        def update_if_changed(current_ids):
            normalized_current = normalize_selection(current_ids)
            return current_ids if normalized_current == incoming_ids else incoming_ids

        set_shared_selected_ids(update_if_changed)

    _previous_observer = getattr(gs_widget, "_marimo_selection_observer", None)
    if _previous_observer is not None:
        gs_widget.unobserve(_previous_observer, names="selected_points")

    gs_widget.observe(sync_grandscatter_selection, names="selected_points")
    gs_widget._marimo_selection_observer = sync_grandscatter_selection
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
    hover_overlay,
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
            mo.vstack([hover_overlay, gs_widget]),
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
