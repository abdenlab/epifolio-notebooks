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
    import marimo as mo
    import numpy as np
    import pandas as pd

    return mo, np, pd


@app.cell
def _(mo):
    mo.md("""
    # Explore cancer cCRE signatures
    """)
    return


@app.cell
def _():
    from epifolio import ASSETS
    from epifolio.data_utils import load_h_matrix
    from epifolio.color_utils import load_color_map, resolve_cancer_colors
    from epifolio.heatmap import create_heatmap_figure, load_grouping_data

    return (
        ASSETS,
        create_heatmap_figure,
        load_color_map,
        load_grouping_data,
        load_h_matrix,
        resolve_cancer_colors,
    )


@app.cell
def _(
    ASSETS,
    load_color_map,
    load_grouping_data,
    load_h_matrix,
    np,
    pd,
    resolve_cancer_colors,
):
    H, sample_ids, comp_names = load_h_matrix(
        ASSETS / "data" / "all_H_component_contributions.csv"
    )
    cancer_types = [sid[:4] for sid in sample_ids]

    cancer_colors = resolve_cancer_colors(
        cancer_types,
        load_color_map(ASSETS / "conf" / "cancer_type_color_map.json"),
    )
    comp_color_map = load_color_map(ASSETS / "conf" / "nmf_component_color_map.json")
    organ_system_data = load_grouping_data(ASSETS / "conf" / "tissue_source_tcga.json")
    embryonic_layer_data = load_grouping_data(ASSETS / "conf" / "emb.json")

    df = pd.DataFrame(H.astype(np.float32), columns=comp_names)
    df["cancer_type"] = cancer_types
    umap_df = pd.read_parquet(ASSETS / "data" / "umap.parquet")
    return (
        H,
        cancer_colors,
        cancer_types,
        comp_color_map,
        comp_names,
        df,
        embryonic_layer_data,
        organ_system_data,
        sample_ids,
        umap_df,
    )


@app.cell
def _(mo):
    get_selection, set_selection = mo.state(None)
    return get_selection, set_selection


@app.cell
def _(cancer_colors, set_selection, umap_df):
    import jscatter

    umap = (
        jscatter.Scatter(
            data=umap_df,
            x="UMAP-1",
            y="UMAP-2",
            color_by="Cancer Type",
            color_map=cancer_colors,
            height=600,
            width=600,
            lasso_callback=True,
            selection_mode="lasso",
        )
        .size(default=5)
        .axes(grid=True, labels=True)
        .tooltip(
            enable=True,
            properties=["Sample ID", "Cancer Type"],
        )
        .widget
    )

    def _on_umap_selection(_):
        sel = umap.selection.tolist()
        set_selection(sel if len(sel) > 0 else None)

    umap.observe(_on_umap_selection, names=["selection"])
    return (umap,)


@app.cell
def _(cancer_colors, comp_names, df, set_selection):
    import grandscatter

    gs = grandscatter.Scatter(
        df,
        axis_fields=comp_names,
        label_field="cancer_type",
        label_colors=cancer_colors,
    )

    def _on_gs_selection(_):
        sel = gs.selected_points
        set_selection(sel if sel else None)

    gs.observe(_on_gs_selection, names=["selected_points"])
    return (gs,)


@app.cell
def _(mo):
    sort_method = mo.ui.dropdown(
        options=["component", "alphabetical", "cancer_type", "organ_system"],
        value="component",
        label="Sort by:",
    )
    return (sort_method,)


@app.cell
def _(
    H,
    active_selection,
    cancer_colors,
    cancer_types,
    comp_color_map,
    create_heatmap_figure,
    embryonic_layer_data,
    mo,
    organ_system_data,
    sample_ids,
    sort_method,
):
    fig = create_heatmap_figure(
        H,
        sample_ids,
        cancer_types,
        sort_method=sort_method.value,
        comp_color_map=comp_color_map,
        cancer_color_map=cancer_colors,
        organ_system_data=organ_system_data,
        embryonic_layer_data=embryonic_layer_data,
        selection=active_selection,
    )

    heatmap_widget = mo.ui.plotly(fig)
    return (heatmap_widget,)


@app.cell
def _(get_selection, gs, mo, umap, umap_df):
    active_selection = get_selection()
    umap.selection = active_selection or []
    gs.selected_points = active_selection or []
    table = mo.ui.table(
        umap_df.iloc[active_selection].reset_index(drop=True)
        if active_selection
        else umap_df
    )
    return active_selection, table


@app.cell(hide_code=True)
def _(active_selection, gs, heatmap_widget, mo, sort_method, table, umap):
    _caption = (
        f"Showing {len(active_selection)} selected samples."
        if active_selection
        else "Showing all samples. Use the lasso tool on a scatter to filter."
    )

    mo.vstack(
        [
            sort_method,
            mo.md(f"**{_caption}**"),
            heatmap_widget,
            mo.md("### Embeddings"),
            mo.md(
                "_Use the lasso tool on either scatter to select samples and "
                "filter the heatmap & table._"
            ),
            mo.hstack([umap, gs], widths=[0.5, 0.5]),
            mo.md("### Selected Samples"),
            table,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
