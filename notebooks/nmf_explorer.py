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
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from urllib.request import urlopen

    import marimo as mo
    import numpy as np
    import pandas as pd

    return json, mo, np, pd, urlopen


@app.cell
def _(json, np, pd, urlopen):
    DATA_BASE = "https://projects.abdenlab.org/itcr/epifolio"

    # NMF sample embeddings (H matrix): 404 samples x 24 components
    nmf_df = pd.read_parquet(f"{DATA_BASE}/embeddings/tcga.atac.nmf.sample.pq")
    nmf_cols = [c for c in nmf_df.columns if c.startswith("NMF")]
    H = nmf_df[nmf_cols].values.astype(np.float32)
    sample_ids = nmf_df["sample"].tolist()

    # UMAP coordinates
    umap_df = pd.read_parquet(f"{DATA_BASE}/embeddings/tcga.atac.nmf.umap.pq")

    # Metadata
    meta_df = pd.read_parquet(f"{DATA_BASE}/metadata/samples.atac.parquet")
    cancer_groups_df = pd.read_parquet(f"{DATA_BASE}/metadata/cancer_groups.parquet")

    # Colormaps
    def _load_json(url):
        with urlopen(url) as r:
            return json.loads(r.read())

    cancer_group_colors = _load_json(f"{DATA_BASE}/metadata/colors.cancer_group.json")
    organ_system_colors = _load_json(f"{DATA_BASE}/metadata/colors.organ_system.json")
    nmf_component_colors = _load_json(f"{DATA_BASE}/metadata/colors.nmf_component.json")
    return (
        H,
        cancer_group_colors,
        cancer_groups_df,
        meta_df,
        nmf_cols,
        nmf_component_colors,
        nmf_df,
        organ_system_colors,
        umap_df,
    )


@app.cell
def _(cancer_groups_df, meta_df, nmf_cols, nmf_df, np, umap_df):
    # Build combined dataframe for scatter plots and heatmap
    # Join sample embeddings with metadata
    df = (
        nmf_df.merge(
            meta_df[["sample_barcode", "group_id", "cohort"]],
            left_on="sample",
            right_on="sample_barcode",
            how="inner",
        )
        .merge(
            cancer_groups_df[["group_id", "organ_system", "group_label", "primary_site", "histology"]],
            on="group_id",
            how="left",
        )
        .merge(
            umap_df,
            on="sample",
            how="inner",
        )
    )

    # Dominant component per sample
    dominant = np.argmax(df[nmf_cols].values, axis=1) + 1
    df["dominant_component"] = [f"NMF{d}" for d in dominant]

    # Scatter-friendly dataframe
    scatter_df = df[
        ["sample", "group_id", "organ_system", "group_label", "primary_site",
         "histology", "dominant_component", "UMAP1", "UMAP2"]
    ].copy()
    return df, scatter_df


@app.cell
def _(df, mo, nmf_cols):
    mo.md(f"""
    **Loaded {len(df)} samples**, {len(nmf_cols)} NMF components,
    {df["group_id"].nunique()} cancer groups,
    {df["organ_system"].nunique()} organ systems.
    """)
    return


@app.cell
def _(mo):
    get_selection, set_selection = mo.state(None)
    return get_selection, set_selection


@app.cell
def _(cancer_group_colors, scatter_df, set_selection):
    import jscatter

    umap = (
        jscatter.Scatter(
            data=scatter_df,
            x="UMAP1",
            y="UMAP2",
            color_by="group_id",
            color_map=cancer_group_colors,
            height=600,
            width=600,
            lasso_callback=True,
            selection_mode="lasso",
        )
        .size(default=5)
        .axes(grid=True, labels=True)
        .tooltip(
            enable=True,
            properties=["sample", "group_id", "organ_system", "histology"],
        )
        .widget
    )

    def _on_umap_selection(_):
        sel = umap.selection.tolist()
        set_selection(sel if len(sel) > 0 else None)

    umap.observe(_on_umap_selection, names=["selection"])
    return (umap,)


@app.cell
def _(cancer_group_colors, df, nmf_cols, set_selection):
    import grandscatter

    gs = grandscatter.Scatter(
        df,
        axis_fields=nmf_cols[:min(9, len(nmf_cols))],
        label_field="group_id",
        label_colors=cancer_group_colors,
    )

    def _on_gs_selection(_):
        sel = gs.selected_points
        set_selection(sel if sel else None)

    gs.observe(_on_gs_selection, names=["selected_points"])
    return (gs,)


@app.cell
def _(mo):
    sort_method = mo.ui.dropdown(
        options=["component", "group_id", "organ_system"],
        value="component",
        label="Sort by:",
    )
    return (sort_method,)


@app.cell
def _(
    H,
    active_selection,
    cancer_group_colors,
    cancer_groups_df,
    df,
    mo,
    nmf_cols,
    nmf_component_colors,
    np,
    organ_system_colors,
    set_selection,
    sort_method,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # -- ordering --
    n_samples, n_comps = H[:len(df)].shape

    _g2o = dict(zip(cancer_groups_df["group_id"], cancer_groups_df["organ_system"]))
    _group_ids = df["group_id"].tolist()
    _organ_systems = [_g2o.get(g, "Unknown") for g in _group_ids]

    comp_order = np.argsort(-H[:len(df)].sum(axis=0))
    H_ord = H[:len(df)][:, comp_order]

    if sort_method.value == "component":
        winners = np.argmax(H_ord, axis=1)
        samp_order = np.lexsort((-H_ord[np.arange(n_samples), winners], winners))
    elif sort_method.value == "group_id":
        winners = np.argmax(H_ord, axis=1)
        samp_order = np.lexsort((
            -H_ord[np.arange(n_samples), winners],
            winners,
            _group_ids,
        ))
    elif sort_method.value == "organ_system":
        winners = np.argmax(H_ord, axis=1)
        samp_order = np.lexsort((
            -H_ord[np.arange(n_samples), winners],
            winners,
            _organ_systems,
        ))
    else:
        samp_order = np.arange(n_samples)

    H_sorted = H_ord[samp_order]
    ordered_groups = [_group_ids[i] for i in samp_order]
    ordered_organs = [_organ_systems[i] for i in samp_order]
    comp_names_ordered = [nmf_cols[i] for i in comp_order]
    comp_colors = [nmf_component_colors.get(c, "#999999") for c in comp_names_ordered]

    # -- selection highlight mask (mapped through sort order) --
    sel = active_selection
    if sel is not None:
        _sel_set = set(sel)
        _sel_mask = [1 if samp_order[i] in _sel_set else 0 for i in range(n_samples)]
    else:
        _sel_mask = [0] * n_samples

    # -- on_change: map plotly selection back to original indices --
    _samp_order = samp_order

    def _on_heatmap_select(value):
        if not value:
            return
        x_positions = sorted({pt["x"] for pt in value if "x" in pt})
        if not x_positions:
            set_selection(None)
            return
        original_indices = [int(_samp_order[x]) for x in x_positions if x < len(_samp_order)]
        set_selection(original_indices if original_indices else None)

    # -- build figure: main heatmap on top (row 1), selection strip below (row 2) --
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.52, 0.03, 0.06, 0.06, 0.06],
        vertical_spacing=0.04,
        subplot_titles=("NMF Component Activities", "Selection", "Dominant Component", "Cancer Group", "Organ System"),
    )

    # Main heatmap (row 1)
    W_norm = H_sorted / (H_sorted.sum(axis=1, keepdims=True) + 1e-12)
    fig.add_trace(
        go.Heatmap(
            z=W_norm.T,
            x=list(range(n_samples)),
            y=comp_names_ordered,
            colorscale="Turbo",
            showscale=False,
            hovertemplate="Sample: %{x}<br>Component: %{y}<br>Activity: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(range(n_comps)),
        ticktext=comp_names_ordered,
        autorange="reversed",
        row=1,
        col=1,
    )

    # Selection highlight strip (row 2)
    fig.add_trace(
        go.Heatmap(
            z=[_sel_mask],
            x=list(range(n_samples)),
            y=["sel"],
            colorscale=[[0, "white"], [1, "#FFDD57"]],
            showscale=False,
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[["selected" if v else "" for v in _sel_mask]],
            zmin=0,
            zmax=1,
        ),
        row=2,
        col=1,
    )

    # Dominant component strip (row 3)
    winners_sorted = np.argmax(H_sorted, axis=1)
    _cscale = []
    for i in range(n_comps):
        frac = i / max(n_comps - 1, 1)
        _cscale.append((frac, comp_colors[i]))
    fig.add_trace(
        go.Heatmap(
            z=[winners_sorted.tolist()],
            x=list(range(n_samples)),
            y=["dominant"],
            colorscale=_cscale,
            showscale=False,
            hovertemplate="Dominant: %{customdata}<extra></extra>",
            customdata=[[comp_names_ordered[w] for w in winners_sorted]],
        ),
        row=3,
        col=1,
    )

    # Cancer group strip (row 4)
    _unique_groups = sorted(set(ordered_groups))
    _g2i = {g: i for i, g in enumerate(_unique_groups)}
    _gidx = [_g2i[g] for g in ordered_groups]
    _gcolors = [cancer_group_colors.get(g, "#CCCCCC") for g in _unique_groups]
    _ng = len(_unique_groups)
    _gscale = [(i / max(_ng - 1, 1), _gcolors[i]) for i in range(_ng)]
    fig.add_trace(
        go.Heatmap(
            z=[_gidx],
            x=list(range(n_samples)),
            y=["group"],
            colorscale=_gscale,
            showscale=False,
            hovertemplate="Cancer Group: %{customdata}<extra></extra>",
            customdata=[ordered_groups],
        ),
        row=4,
        col=1,
    )

    # Organ system strip (row 5)
    _unique_organs = sorted(set(ordered_organs))
    _o2i = {o: i for i, o in enumerate(_unique_organs)}
    _oidx = [_o2i[o] for o in ordered_organs]
    _ocolors = [organ_system_colors.get(o, "#CCCCCC") for o in _unique_organs]
    _no = len(_unique_organs)
    _oscale = [(i / max(_no - 1, 1), _ocolors[i]) for i in range(_no)]
    fig.add_trace(
        go.Heatmap(
            z=[_oidx],
            x=list(range(n_samples)),
            y=["organ"],
            colorscale=_oscale,
            showscale=False,
            hovertemplate="Organ System: %{customdata}<extra></extra>",
            customdata=[ordered_organs],
        ),
        row=5,
        col=1,
    )

    # -- axis cleanup --
    for r in range(1, 5):
        fig.update_xaxes(showticklabels=False, row=r, col=1)
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(n_samples)),
        ticktext=ordered_groups,
        tickangle=90,
        tickfont=dict(size=7),
        row=5,
        col=1,
    )
    for r in (2, 3, 4, 5):
        fig.update_yaxes(showticklabels=False, row=r, col=1)

    fig.update_layout(
        height=max(900, n_comps * 25 + 300),
        margin=dict(l=80, r=80, t=30, b=100),
        showlegend=False,
        autosize=True,
        plot_bgcolor="white",
        dragmode="select",
        selectdirection="h",
    )

    heatmap_fig = mo.ui.plotly(fig, on_change=_on_heatmap_select)

    return (heatmap_fig,)


@app.cell
def _(get_selection, gs, mo, scatter_df, umap):
    active_selection = get_selection()
    umap.selection = active_selection or []
    gs.selected_points = active_selection or []
    table = mo.ui.table(
        scatter_df.iloc[active_selection].reset_index(drop=True)
        if active_selection
        else scatter_df,
    )
    return active_selection, table


@app.cell(hide_code=True)
def _(active_selection, gs, heatmap_fig, mo, sort_method, table, umap):
    _num_selected = len(active_selection) if active_selection else 'all'

    mo.vstack(
        [
            mo.md("### Embedding scatter plots"),
            mo.md(
                f"Use the lasso tool on either scatter to select samples. [**{_num_selected}** samples selected]"
            ),
            mo.hstack([umap, gs], widths=[0.5, 0.5]),

            mo.md("### Component scores"),
            mo.md(f"Use box-select to select samples. [**{_num_selected}** samples selected]"),
            sort_method,
            heatmap_fig,
        
            mo.md("### Selected samples"),
            table,
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
