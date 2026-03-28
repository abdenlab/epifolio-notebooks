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
    mo.md("# Explore cancer cCRE signatures")
    return


@app.cell
def _():
    from epifolio import ASSETS
    from epifolio.data_utils import load_cfg, load_h_matrix, load_umap
    from epifolio.color_utils import load_color_map, resolve_cancer_colors
    from epifolio.heatmap import create_heatmap_figure
    from epifolio.scatter import (
        create_umap_scatter,
        create_grandscatter_widget,
    )

    CFG_PATH = str(ASSETS / "conf" / "config.json")
    return (
        ASSETS,
        CFG_PATH,
        create_grandscatter_widget,
        create_heatmap_figure,
        create_umap_scatter,
        load_cfg,
        load_color_map,
        load_h_matrix,
        load_umap,
        resolve_cancer_colors,
    )


@app.cell
def _(mo):
    sort_method = mo.ui.dropdown(
        options=["component", "alphabetical", "cancer_type", "organ_system"],
        value="component",
        label="Sort by:",
    )
    return (sort_method,)


@app.cell
def _(mo):
    shared_selected_ids, set_shared_selected_ids = mo.state([])
    return shared_selected_ids, set_shared_selected_ids


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@app.cell
def _(CFG_PATH, load_cfg, load_color_map, load_h_matrix, load_umap, resolve_cancer_colors, pd):
    from pathlib import Path

    cfg = load_cfg(CFG_PATH)
    cfg_dir = Path(CFG_PATH).parent.parent  # assets root

    csv_path = cfg_dir / cfg.get("DEFAULT_CSV_FILENAME", "data/all_H_component_contributions.csv")
    H, sample_ids, comp_names = load_h_matrix(csv_path)

    cancer_types = [sid[:4] for sid in sample_ids]
    cancer_colors = resolve_cancer_colors(
        cancer_types,
        load_color_map(str(cfg_dir / "conf" / "cancer_type_color_map.json")),
    )

    umap_df = load_umap(
        {**cfg, "UMAP_FILENAME": str(cfg_dir / cfg.get("UMAP_FILENAME", "data/umap.parquet"))},
        sample_ids,
        cancer_types,
    )
    return H, cancer_colors, cancer_types, cfg, cfg_dir, comp_names, csv_path, sample_ids, umap_df, Path


# ---------------------------------------------------------------------------
# UMAP scatter (jscatter with lasso selection)
# ---------------------------------------------------------------------------


@app.cell
def _(cancer_colors, create_umap_scatter, umap_df):
    scatter = create_umap_scatter(umap_df, cancer_colors)
    return (scatter,)


@app.cell
def _(mo, scatter):
    scatter_widget = mo.ui.anywidget(scatter.widget)
    return (scatter_widget,)


@app.cell
def _(np):
    def normalize_selection(selection):
        if selection is None:
            return []
        if isinstance(selection, np.ndarray):
            values = selection.tolist()
        else:
            values = list(selection)
        return [int(v) for v in values]
    return (normalize_selection,)


@app.cell
def _(normalize_selection, scatter_widget, set_shared_selected_ids):
    _incoming = normalize_selection(scatter_widget.selection)

    def _update(current):
        return current if _incoming == current else _incoming

    set_shared_selected_ids(_update)
    return


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


@app.cell
def _(CFG_PATH, create_heatmap_figure, csv_path, mo, shared_selected_ids, sort_method):
    _selection = shared_selected_ids()
    _selected = list(_selection) if _selection else None
    _caption = (
        f"Showing {len(_selection)} selected samples."
        if _selected
        else "Showing all samples. Use the lasso tool on the scatter to filter."
    )

    _fig = create_heatmap_figure(
        CFG_PATH,
        sort_method.value,
        csv_path=csv_path,
        selection=_selected,
    )
    heatmap_plot = mo.ui.plotly(_fig)
    caption = _caption
    return caption, heatmap_plot


# ---------------------------------------------------------------------------
# Grandscatter (16-D NMF proportion explorer)
# ---------------------------------------------------------------------------


@app.cell
def _(CFG_PATH, create_grandscatter_widget, mo):
    try:
        _gs = create_grandscatter_widget(CFG_PATH)
        grandscatter_plot = mo.ui.anywidget(_gs)
        gs_widget = _gs
    except Exception as e:
        grandscatter_plot = mo.md(
            f"**Grandscatter error:** {type(e).__name__}: {str(e)[:200]}"
        )
        gs_widget = None
    return grandscatter_plot, gs_widget


@app.cell
def _(gs_widget, normalize_selection, set_shared_selected_ids):
    if gs_widget is not None:
        _incoming = normalize_selection(gs_widget.selected_points)

        def _update(current):
            return current if _incoming == current else _incoming

        set_shared_selected_ids(_update)
    return


@app.cell
def _(gs_widget, normalize_selection, shared_selected_ids):
    _target = shared_selected_ids()
    if gs_widget is not None:
        _current = normalize_selection(gs_widget.selected_points)
        if _current != _target:
            gs_widget.selected_points = _target
    return


@app.cell
def _(normalize_selection, scatter_widget, shared_selected_ids):
    _target = shared_selected_ids()
    _current = normalize_selection(scatter_widget.selection)
    if _current != _target:
        scatter_widget.selection = _target
    return


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@app.cell
def _(caption, grandscatter_plot, heatmap_plot, mo, scatter_widget, shared_selected_ids, sort_method, umap_df):
    _selection = shared_selected_ids()
    _selected_df = (
        umap_df.iloc[_selection].reset_index(drop=True)
        if _selection
        else umap_df
    )

    mo.vstack([
        sort_method,
        mo.md(f"**{caption}**"),
        mo.hstack([scatter_widget, heatmap_plot], widths=[0.4, 0.6]),
        mo.md("### Multi-Dimensional NMF Proportions"),
        mo.md(
            "_Drag axis handles to rotate and explore the 16-dimensional NMF "
            "proportion space. Use the lasso tool to select samples and filter "
            "the heatmap & table._"
        ),
        grandscatter_plot,
        mo.md("### Selected Samples"),
        mo.ui.table(_selected_df),
    ])
    return


if __name__ == "__main__":
    app.run()
