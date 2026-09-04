# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.21.1",
#     "anywidget>=0.9.18",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "pyarrow>=18.0.0",
#     "plotly>=6.1.2",
#     "polars>=1.42.0",
#     "bioframe>=0.8.0",
#     "jupyter-scatter[all]>=1.0.1",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
#
# [tool.marimo.runtime]
# output_max_bytes = 50_000_000
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import jscatter
    import marimo as mo
    import numpy as np
    import pyarrow.compute as pc

    import epifolio.cnv as cnv

    return cnv, jscatter, mo, np, pc


@app.cell
def _(cnv):
    DATA = cnv.open_dataset(
        [cnv.source(name) for name in (cnv.TRACKS, cnv.DISTRIBUTIONS)]
    )
    return (DATA,)


@app.cell
def _(DATA, cnv):
    cnv.assert_edges_match(DATA.schema_for("index"))
    return


@app.cell
def _(DATA, cnv, mo):
    COHORT = mo.cache(cnv.load_cohort)(DATA)
    CANCERS = sorted(COHORT)
    return CANCERS, COHORT


@app.cell
def _(DATA, cnv, mo):
    @mo.cache
    def read_row(biosample: str, *tags: str) -> dict:
        return cnv.load_panel(DATA, biosample, *tags)

    return (read_row,)


@app.cell
def _(CANCERS, COHORT, mo):
    cancer_dd = mo.ui.dropdown(
        options=CANCERS,
        value=CANCERS[0],
        label="Cancer type",
        searchable=True,
    )
    return (cancer_dd,)


@app.cell
def _(COHORT, cancer_dd, mo):
    patients = sorted(COHORT[cancer_dd.value])
    patient_dd = mo.ui.dropdown(
        options=patients,
        value=patients[0],
        label=f"Patient — {len(patients)} in {cancer_dd.value}",
        searchable=True,
    )
    return patient_dd, patients


@app.cell
def _(COHORT, cancer_dd, mo, patient_dd):
    aliquots = COHORT[cancer_dd.value][patient_dd.value]
    aliquot_dd = mo.ui.dropdown(
        options=aliquots,
        value=aliquots[0],
        label=f"Aliquot — {len(aliquots)}",
        searchable=True,
    )
    return aliquot_dd, aliquots


@app.cell
def _(aliquot_dd):
    biosample = aliquot_dd.value
    return (biosample,)


@app.cell
def _(biosample, read_row):
    meta = read_row(biosample, "header", "shared")
    cn_order = [str(level) for level in meta["cn_order"]]
    return cn_order, meta


@app.cell
def _(cnv, mo):
    get_window, set_window = mo.state(cnv.FULL_GENOME, allow_self_loops=True)
    return get_window, set_window


@app.cell
def _(cnv):
    def clamp_window(plotly_ranges):
        # H5: read "x" key, not "xaxis.range[0]"
        xr = (plotly_ranges or {}).get("x")
        if not xr:
            # H8: return None so the caller skips set_window (no snap-back)
            return None
        lo = max(0.0, float(min(xr)))
        hi = min(float(cnv.GENOME_LEN), float(max(xr)))
        return (lo, hi) if hi > lo else None

    return (clamp_window,)


@app.cell
def _(np):
    def lasso_indices(selection):
        if selection is None:
            return np.array([], dtype=np.int64)
        if isinstance(selection, np.ndarray):
            return selection.astype(np.int64)
        if isinstance(selection, memoryview):
            return np.frombuffer(selection, dtype=np.uint32).astype(np.int64)
        if isinstance(selection, bytes):
            return np.frombuffer(selection, dtype=np.uint32).astype(np.int64)
        if isinstance(selection, dict) and "view" in selection:
            dtype = selection.get("dtype", "uint32")
            return np.frombuffer(selection["view"], dtype=dtype).astype(np.int64)
        return np.asarray(list(selection), dtype=np.int64)

    return (lasso_indices,)


@app.cell
def _(biosample, clamp_window, cnv, get_window, mo, read_row, set_window):
    p1_row = read_row(biosample, "1")

    # H4: read .ranges off the UI element, not the callback arg (value.ranges would be
    # a list[dict] of points, not a range dict)
    def sync1(_value):
        moved = clamp_window(panel1.ranges)
        if moved:
            set_window(moved)

    panel1 = mo.ui.plotly(
        cnv.panel_1(p1_row, get_window(), biosample),
        on_change=sync1,
        config={"responsive": True},
    )
    return panel1, p1_row, sync1


@app.cell
def _(biosample, clamp_window, cnv, get_window, mo, read_row, set_window):
    p2_row = read_row(biosample, "2")

    def sync2(_value):
        moved = clamp_window(panel2.ranges)
        if moved:
            set_window(moved)

    panel2 = mo.ui.plotly(
        cnv.panel_2(p2_row, get_window(), biosample),
        on_change=sync2,
        config={"responsive": True},
    )
    return panel2, p2_row, sync2


@app.cell
def _(biosample, cn_order, cnv, mo, read_row):
    # H1: read the correct row for panel 3 (tag "3", not "1")
    p3_row = read_row(biosample, "3")
    panel3 = mo.ui.plotly(cnv.panel_3(p3_row, cn_order))
    return p3_row, panel3


@app.cell
def _(cn_order, cnv, panel3):
    # panel3.points MUST be read outside panel3's defining cell: marimo subtracts
    # get_defining_cells() from the cells it reruns on a UI value change
    # (runtime.py:1943), so an in-cell read stays frozen at the initial value.
    picked_cn = cnv.bar_levels(panel3.points, cn_order)
    active_cn = picked_cn or cn_order
    return active_cn, picked_cn


@app.cell
def _(active_cn, biosample, cn_order, cnv, mo, read_row):
    p4_row = read_row(biosample, "4")
    panel4 = mo.ui.plotly(cnv.panel_4(p4_row, active_cn, cn_order, biosample))
    return p4_row, panel4


@app.cell
def _(biosample, read_row):
    p6_row = read_row(biosample, "6")
    return (p6_row,)


@app.cell
def _(mo):
    # M2b: mo.state hoisted into its own cell with no dependency on active_cn,
    # so a panel-3 click no longer recreates it and silently drops the
    # selection (F2). Keyed by (cancer_z, healthy_z) bin coordinate rather
    # than positional index, so it also survives a dens_tbl/cn_tbl6 rebuild
    # on a CN-filter change instead of addressing the wrong points
    # (cnvpostpr.md §7 decision 4) — this is the single selection shared by
    # panels 5 and 6 (M3).
    get_sel_bins, set_sel_bins = mo.state(frozenset())
    return get_sel_bins, set_sel_bins


@app.cell
def _(active_cn, cn_order, cnv, jscatter, p6_row):
    # D6/M3a: density_table_with_index does the same np.unique aggregation as
    # density_table, but also returns inv — the panel-6-row -> panel-5-bin map
    # that used to be discarded (cnvpostpr.md §3).
    density = cnv.density_table_with_index(p6_row, active_cn, cn_order)
    dens_tbl, dens_inv = density.table, density.inv

    scatter5 = jscatter.Scatter(
        data=dens_tbl,  # E3: pass Arrow table directly
        x="cancer_z",
        y="healthy_z",
        x_scale=cnv.Z_LIM,
        y_scale=cnv.Z_LIM,
        color_by="log_cCREs",  # D22/D23
        color_map="inferno",
        size=3,
        opacity=0.9,
        height=460,
        axes_labels=["Cancer tissue ATAC z-score", "Healthy tissue ATAC z-score"],
        tooltip=True,
        tooltip_properties=["cancer_z", "healthy_z", "cCREs", "log_cCREs"],
        legend=True,
        lasso_initiator=True,
    )

    # M2a: the raw widget, not mo.ui.anywidget. mo.ui.anywidget turns
    # jscatter's synced `hovering` trait into a reactive dependency, forcing a
    # kernel round-trip and a full remount of both scatter canvases on every
    # mouse move (cnvpostpr.md F1). Selection still reaches the kernel via
    # .observe() below, which does not go through mo.ui.anywidget at all.
    panel5_widget = scatter5.widget

    return dens_inv, dens_tbl, panel5_widget, scatter5


@app.cell
def _(active_cn, cn_order, cnv, jscatter, p6_row):
    # D5: add color_map with correct palette hexes
    palette6 = cnv.cn_colors(cn_order)
    color_map6 = {cnv.cn_label(c): hex_val for c, hex_val in palette6.items()}

    cn_tbl6 = cnv.cn_table(p6_row, active_cn, cn_order)

    scatter6 = jscatter.Scatter(
        data=cn_tbl6,  # E3: pass Arrow table directly
        x="cancer_z",
        y="healthy_z",
        x_scale=cnv.Z_LIM,
        y_scale=cnv.Z_LIM,
        color_by="copy_number",
        color_map=color_map6,  # D5
        size=3,
        opacity=0.8,
        height=460,
        axes_labels=["Cancer tissue ATAC z-score", "Healthy tissue ATAC z-score"],
        tooltip=True,
        tooltip_properties=["cancer_z", "healthy_z", "copy_number", "cCREs"],
        legend=True,
        lasso_initiator=True,
    )

    # M2a: raw widget — see panel5_widget above
    panel6_widget = scatter6.widget

    return cn_tbl6, color_map6, palette6, panel6_widget, scatter6


@app.cell
def _(dens_tbl, lasso_indices, panel5_widget, set_sel_bins):
    # A panel-5 lasso sets the shared selection directly, as bin coordinates.
    def sync5(change):
        idx = lasso_indices(change.get("new"))
        rows = dens_tbl.take(idx) if len(idx) else dens_tbl.slice(0, 0)
        set_sel_bins(frozenset(zip(rows["cancer_z"].to_pylist(), rows["healthy_z"].to_pylist())))

    panel5_widget.observe(sync5, names="selection")
    return (sync5,)


@app.cell
def _(dens_inv, dens_tbl, lasso_indices, np, panel6_widget, set_sel_bins):
    # M3b: a panel-6 lasso maps to panel-5 bins through dens_inv before
    # joining the shared selection.
    def sync6(change):
        idx = lasso_indices(change.get("new"))
        bin_pos = np.unique(dens_inv[idx]) if len(idx) else np.array([], dtype=np.int64)
        rows = dens_tbl.take(bin_pos) if len(bin_pos) else dens_tbl.slice(0, 0)
        set_sel_bins(frozenset(zip(rows["cancer_z"].to_pylist(), rows["healthy_z"].to_pylist())))

    panel6_widget.observe(sync6, names="selection")
    return (sync6,)


@app.cell
def _(cn_tbl6, dens_inv, dens_tbl, get_sel_bins, np):
    # Resolve the shared bin-coordinate selection against the *current*
    # dens_tbl/cn_tbl6 (rebuilt on every active_cn change). bin_pos indexes
    # dens_tbl; cn6_pos indexes cn_tbl6 via dens_inv (M3c).
    sel_bins = get_sel_bins()
    if sel_bins:
        pairs5 = list(zip(dens_tbl["cancer_z"].to_pylist(), dens_tbl["healthy_z"].to_pylist()))
        bin_pos = np.array([i for i, pair in enumerate(pairs5) if pair in sel_bins], dtype=np.int64)
    else:
        bin_pos = np.array([], dtype=np.int64)
    cn6_pos = np.flatnonzero(np.isin(dens_inv, bin_pos)) if len(bin_pos) else np.array([], dtype=np.int64)

    # D25: show whole-sample stats/highlight until something is selected
    selected_dens = dens_tbl.take(bin_pos) if len(bin_pos) else dens_tbl
    selected_cn6 = cn_tbl6.take(cn6_pos) if len(cn6_pos) else cn_tbl6
    return bin_pos, cn6_pos, selected_cn6, selected_dens


@app.cell
def _(bin_pos, panel5_widget, sel_bins):
    # Apply the resolved selection back onto panel 5 — first render, a
    # CN-filter rebuild, or a panel-6-driven cross-highlight (M3b) all land
    # here. Guarded so a no-op assignment can't retrigger sync5 in a loop
    # (matching nmf_sample_explorer.py:389-392).
    _target = bin_pos.tolist() if len(sel_bins) else []
    _current = panel5_widget.selection
    if list([] if _current is None else _current) != _target:
        panel5_widget.selection = _target
    return


@app.cell
def _(cn6_pos, panel6_widget, sel_bins):
    # M3c: apply the resolved selection back onto panel 6, same guard as panel 5.
    _target = cn6_pos.tolist() if len(sel_bins) else []
    _current = panel6_widget.selection
    if list([] if _current is None else _current) != _target:
        panel6_widget.selection = _target
    return


@app.cell
def _(bin_pos, mo, pc, selected_cn6, selected_dens):
    # M2c: Arrow compute instead of .to_pylist() + a Python loop — this runs
    # on every legitimate lasso, not just the flicker-prone hover path.
    total_cCREs = int((pc.sum(selected_dens["cCREs"]).as_py() or 0))
    gain_cCREs = int(
        pc.sum(
            pc.if_else(
                pc.greater(selected_dens["cancer_z"], selected_dens["healthy_z"]),
                selected_dens["cCREs"],
                0,
            )
        ).as_py()
        or 0
    )
    gain_frac = gain_cCREs / total_cCREs if total_cCREs else 0.0

    counts_by_cn = (
        selected_cn6.group_by("copy_number")
        .aggregate([("cCREs", "sum")])
        .rename_columns({"cCREs_sum": "cCREs"})
        .sort_by("copy_number")
        .to_pandas()
    )

    # M3d: one shared cCRE-weighted readout for the linked panel 5/6
    # selection, replacing the two independent readouts.
    readout56 = mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(value=f"{total_cCREs:,}", label="cCREs in selection"),
                    mo.stat(value=f"{gain_frac:.1%}", label="gain fraction (cCRE-weighted)"),
                    mo.stat(value=f"{len(bin_pos):,}", label="bins selected"),
                ]
            ),
            mo.ui.table(counts_by_cn, selection=None),
        ],
        gap=1,
    )
    return (readout56,)


@app.cell
def _(
    aliquot_dd,
    cancer_dd,
    mo,
    panel1,
    panel2,
    panel3,
    panel4,
    panel5_widget,
    panel6_widget,
    patient_dd,
    readout56,
):
    mo.vstack(
        [
            mo.hstack([cancer_dd, patient_dd, aliquot_dd], justify="start", gap=1.5),
            panel1,
            panel2,
            mo.hstack([panel3, panel4], widths="equal", gap=1.5),
            mo.hstack([panel5_widget, panel6_widget], widths="equal", gap=1.5),
            readout56,
        ],
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
