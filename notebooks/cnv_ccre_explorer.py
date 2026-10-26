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
    import functools
    import json

    import jscatter
    import marimo as mo
    import numpy as np
    import pyarrow.compute as pc
    import polars as pl

    import epifolio.cnv as cnv

    return cnv, functools, jscatter, json, mo, np, pc, pl


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
    try:
        detail_index = cnv.load_detail_index(DATA.schema_for("2"))
        detail_error = None
    except Exception as exc:  # panel 2 stays on the binned overview without the detail tier
        detail_index = None
        detail_error = str(exc)

    # A sample switch reads only its value columns: the coordinates are read
    # once, and a healthy column is shared by every sample on that ENCODE
    # reference. LRU bounds keep memory flat while browsing samples.
    @mo.cache
    def read_coords():
        return cnv.load_coords(detail_index)

    @mo.lru_cache(maxsize=4)
    def read_healthy_column(column: str):
        return cnv.load_detail_column(detail_index, column)

    @mo.lru_cache(maxsize=4)
    def read_zscores(biosample: str):
        if detail_index is None:
            return None
        healthy = detail_index.healthy[biosample]
        return cnv.zscore_frame(
            read_coords(),
            cnv.load_detail_column(detail_index, detail_index.columns[biosample]),
            None if healthy is None else read_healthy_column(healthy),
        )

    return detail_error, detail_index, read_coords, read_healthy_column, read_zscores


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
def _(cnv, mo, pl):
    @mo.cache
    def read_loci():
        return cnv.load_loci()

    try:
        loci = read_loci()
        loci_error = None
    except Exception as exc:  # keep the raster dashboard usable when the index host is down
        loci = pl.DataFrame(
            {
                "cCRE_index": pl.Series([], dtype=pl.UInt32),
                "chrom": pl.Series([], dtype=pl.String),
                "start": pl.Series([], dtype=pl.Int64),
                "end": pl.Series([], dtype=pl.Int64),
                "rDHS": pl.Series([], dtype=pl.String),
                "cCRE": pl.Series([], dtype=pl.String),
                "class": pl.Series([], dtype=pl.String),
                "pos": pl.Series([], dtype=pl.Int64),
            }
        )
        loci_error = str(exc)
    return loci, loci_error, read_loci


@app.cell
def _(cnv, mo):
    @mo.cache
    def read_ccre_reference():
        return cnv.load_ccre_reference()

    try:
        ccre_reference = read_ccre_reference()
        ccre_reference_error = None
    except Exception as exc:  # keep the raster dashboard usable if the split reference is unavailable
        ccre_reference = None
        ccre_reference_error = str(exc)
    return ccre_reference, ccre_reference_error, read_ccre_reference


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
def _(mo):
    hide_sentinel = mo.ui.switch(value=False, label="Hide no ATAC-seq signal")
    return (hide_sentinel,)


@app.cell
def _(biosample, read_row):
    meta = read_row(biosample, "header", "shared")
    cn_order = [str(level) for level in meta["cn_order"]]
    return cn_order, meta


@app.cell
def _(biosample, ccre_reference, ccre_reference_error, cn_order, cnv, loci, loci_error, mo):
    @mo.lru_cache(maxsize=4)
    def read_membership(sample: str, expected_n_ccre: int, order: tuple[str, ...]):
        return cnv.load_membership(
            sample,
            reference=ccre_reference,
            expected_n_ccre=expected_n_ccre,
            expected_cn_order=order,
        )

    membership = None
    membership_error = loci_error or ccre_reference_error
    if membership_error is None:
        try:
            membership = read_membership(biosample, loci.height, tuple(cn_order))
            membership_error = None
        except Exception as exc:  # keep the raster dashboard usable if the split sidecar is unavailable
            membership_error = str(exc)
    return membership, membership_error, read_membership


@app.cell
def _(cnv, mo):
    get_window, set_window = mo.state(cnv.FULL_GENOME, allow_self_loops=True)
    # A panel-2 box also carries a z-score band for the ATAC tab's table.
    get_zrange, set_zrange = mo.state(None, allow_self_loops=True)
    # A panel-2 lasso keeps its outline; the table lists the cCREs inside it.
    get_lasso, set_lasso = mo.state(None, allow_self_loops=True)
    # The panel whose selection was made last; its table shows on every tab.
    get_table_source, set_table_source = mo.state(None)
    return (
        get_lasso,
        get_table_source,
        get_window,
        get_zrange,
        set_lasso,
        set_table_source,
        set_window,
        set_zrange,
    )


@app.cell
def _(cnv, mo, set_lasso, set_table_source, set_window, set_zrange):
    def _clear(_value):
        set_window(cnv.FULL_GENOME)
        set_zrange(None)
        set_lasso(None)
        set_table_source(None)

    clear_selection = mo.ui.button(label="Clear selection", on_click=_clear)
    return (clear_selection,)


@app.cell
def _(cnv, json):
    def clamp_window(region):
        # H8: return None so the caller skips set_window (no snap-back)
        if region is None:
            return None
        lo = max(0.0, region["x"][0])
        hi = min(float(cnv.GENOME_LEN), region["x"][1])
        return (lo, hi) if hi > lo else None

    # A plotly value also changes on a tool switch, zoom or pan, carrying the
    # old selection along. Only a new selection may act: re-applying a stale
    # one rebuilt the panel mid-zoom and reset its tool to box select.
    _seen = {}
    # Each panel's last chosen tool (select, lasso, zoom, pan), kept when the
    # panel is rebuilt.
    drag_modes = {}

    def new_selection(name, plot):
        """(changed, cnv.selection_region) for a plot's current selection."""
        # mo.ui.plotly exposes the box as .ranges but keeps a lasso's outline
        # only in its raw selection payload.
        data = getattr(plot, "_selection_data", None) or {}
        drag_modes[name] = data.get("dragmode", drag_modes.get(name))
        key = json.dumps(
            [id(plot), data.get("range"), data.get("selections")], sort_keys=True, default=str
        )
        if _seen.get(name) == key:
            return False, None
        _seen[name] = key
        return True, cnv.selection_region(data)

    return clamp_window, drag_modes, new_selection


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
def _(aliquot_dd, cancer_dd, hide_sentinel, mo, patient_dd):
    mo.hstack([cancer_dd, patient_dd, aliquot_dd, hide_sentinel], justify="start", gap=1.5)
    return


@app.cell
def _(mo):
    # The tab strip only; each tab's panels draw from their own cell below.
    # marimo re-runs every cell that references a UI element whenever its
    # value changes (a brush, a zoom, a pan), except the cell defining it, so
    # panels 1 and 2 are drawn by their defining cells: a panel 1/2
    # interaction then redraws just that panel, never every tab.
    sections = mo.ui.tabs(
        {
            name: ""
            for name in (
                "Copy number",
                "ATAC z-score",
                "CN counts & distributions",
                "Cancer vs healthy",
            )
        }
    )
    sections
    return (sections,)


@app.cell
def _(
    biosample,
    clamp_window,
    cn_order,
    cnv,
    drag_modes,
    get_window,
    mo,
    new_selection,
    read_row,
    sections,
    set_lasso,
    set_table_source,
    set_window,
    set_zrange,
):
    p1_row = read_row(biosample, "1")

    # H4: read the selection off the UI element, not the callback arg (the
    # value is a list[dict] of points). A box or a lasso both set the window.
    def sync1(_value):
        changed, region = new_selection("panel 1", panel1)
        moved = clamp_window(region) if changed else None
        if moved:
            set_window(moved)
            # A new genomic brush drops the previous panel-2 z band and lasso.
            set_zrange(None)
            set_lasso(None)
            set_table_source("panel 1")

    panel1 = None
    if sections.value == "Copy number":
        _fig = cnv.panel_1(p1_row, get_window(), biosample, cn_order)
        _fig.update_layout(dragmode=drag_modes.get("panel 1") or "select")
        panel1 = mo.ui.plotly(_fig, on_change=sync1, config={"responsive": True})
    panel1
    return panel1, p1_row, sync1


@app.cell
def _(
    biosample,
    clamp_window,
    clear_selection,
    cnv,
    detail_error,
    detail_index,
    drag_modes,
    get_lasso,
    get_window,
    get_zrange,
    hide_sentinel,
    loci,
    mo,
    new_selection,
    pl,
    read_ccre_frame,
    read_zscores,
    read_row,
    sections,
    set_lasso,
    set_table_source,
    set_window,
    set_zrange,
):
    p2_row = read_row(biosample, "2")
    _window = get_window()
    _shown = sections.value == "ATAC z-score"
    # Raw per-cCRE values replace the binned overview once the window holds
    # at most DETAIL_MAX_ROWS cCREs. The locus table counts the window, so a
    # wide window never waits on the detail read.
    p2_detail = None
    _n_window = loci.filter(pl.col("pos").is_between(*_window)).height if loci.height else 0
    if _shown and detail_index is not None and _n_window <= cnv.DETAIL_MAX_ROWS:
        _frame = read_ccre_frame()
        if _frame is None or "cancer_z" not in _frame.columns:
            _frame = read_zscores(biosample)
        _in_window = _frame.filter(pl.col("pos").is_between(*_window))
        if _in_window.height <= cnv.DETAIL_MAX_ROWS:
            p2_detail = _in_window

    def sync2(_value):
        changed, region = new_selection("panel 2", panel2)
        moved = clamp_window(region) if changed else None
        if moved:
            set_window(moved)
            set_zrange(region["y"])
            set_lasso(region["lasso"])
            set_table_source("panel 2")

    panel2 = None
    if _shown:
        _fig = cnv.panel_2(
            p2_row,
            _window,
            biosample,
            hide_sentinel=hide_sentinel.value,
            detail=p2_detail,
            zrange=get_zrange(),
            lasso=get_lasso(),
        )
        _fig.update_layout(dragmode=drag_modes.get("panel 2") or "select")
        panel2 = mo.ui.plotly(_fig, on_change=sync2, config={"responsive": True})
    (
        mo.vstack(
            [
                mo.hstack(
                    [clear_selection]
                    + (
                        [mo.md(f"*Panel 2 detail tier unavailable: {detail_error}*")]
                        if detail_error
                        else []
                    ),
                    justify="start",
                    gap=1.5,
                ),
                panel2,
            ],
            gap=1.5,
        )
        if _shown
        else None
    )
    return p2_detail, panel2, p2_row, sync2


@app.cell
def _(biosample, cn_order, cnv, hide_sentinel, mo, new_selection, read_row, set_table_source):
    # H1: read the correct row for panel 3 (tag "3", not "1")
    p3_row = read_row(biosample, "3")
    # Hiding -10 needs only the published per-level sentinel counts.
    _sentinel_n = (
        read_row(biosample, "4")["violin_cancer_sentinel_n"] if hide_sentinel.value else None
    )
    def pick3(_points):
        # A bar pick by box or lasso; a tool switch or pan leaves the table alone.
        if new_selection("panel 3", panel3)[0]:
            set_table_source("panel 3" if panel3.points else None)

    panel3 = mo.ui.plotly(
        cnv.panel_3(p3_row, cn_order, sentinel_n=_sentinel_n), on_change=pick3
    )
    return p3_row, panel3, pick3


@app.cell
def _(cn_order, cnv, panel3):
    # panel3.points MUST be read outside panel3's defining cell: marimo subtracts
    # get_defining_cells() from the cells it reruns on a UI value change
    # (runtime.py:1943), so an in-cell read stays frozen at the initial value.
    picked_cn = cnv.bar_levels(panel3.points, cnv.mapped_levels(cn_order))
    active_cn = picked_cn or cn_order
    return active_cn, picked_cn


@app.cell
def _(active_cn, biosample, cn_order, cnv, hide_sentinel, mo, read_ccre_frame, read_row):
    p4_row = read_row(biosample, "4")
    # With -10 hidden the violins are rebuilt from per-cCRE z-scores, so only
    # then does this cell wait on the detail read.
    _row = p4_row
    if hide_sentinel.value:
        _frame = read_ccre_frame()
        if _frame is not None and {"cancer_z", "healthy_z"} <= set(_frame.columns):
            _row = cnv.drop_sentinel_violins(p4_row, _frame, cn_order)
    panel4 = mo.ui.plotly(cnv.panel_4(_row, active_cn, cn_order, biosample))
    return p4_row, panel4


@app.cell
def _(biosample, cnv, hide_sentinel, read_ccre_frame, read_row):
    p6_row = read_row(biosample, "6")
    # Panels 5/6 draw the published cells, or with -10 hidden recount them
    # from per-cCRE bin_id, leaving out cCREs at -10 on either axis.
    scatter_source = p6_row
    if hide_sentinel.value:
        _frame = read_ccre_frame()
        if _frame is not None and "bin_id" in _frame.columns:
            scatter_source = cnv.scatter_row(cnv.drop_sentinel(_frame, ("cancer_z", "healthy_z")))
    return p6_row, scatter_source


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
    # Membership selection is kept separately from the coordinate selection
    # used for panel 5/6 cross-highlighting.  Panel 6 needs exact
    # (bin_id, copy_number) keys; panel 5 needs bin_id keys.
    get_membership_selection, set_membership_selection = mo.state(None)
    # Programmatic top-window highlighting must not be mistaken for a user
    # lasso event by the widget observers below.
    selection_sync = {"panel5": None, "panel6": None}
    return (
        get_membership_selection,
        get_sel_bins,
        selection_sync,
        set_membership_selection,
        set_sel_bins,
    )


@app.cell
def _(active_cn, cn_order, cnv, jscatter, scatter_source):
    # D6/M3a: density_table_with_index does the same np.unique aggregation as
    # density_table, but also returns inv — the panel-6-row -> panel-5-bin map
    # that used to be discarded (cnvpostpr.md §3).
    density = cnv.density_table_with_index(scatter_source, active_cn, cn_order)
    dens_tbl, dens_inv = density.table, density.inv

    # jscatter cannot draw an empty table: no healthy reference, or a
    # panel-3 pick with no cCREs.
    scatter5 = None
    if dens_tbl.num_rows:
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
    panel5_widget = None if scatter5 is None else scatter5.widget

    return dens_inv, dens_tbl, panel5_widget, scatter5


@app.cell
def _(active_cn, cn_order, cnv, jscatter, scatter_source):
    # One fixed colour per copy-number level (copy_number_color_map.json).
    palette6 = cnv.cn_colors(cn_order)

    cn_tbl6 = cnv.cn_table(scatter_source, active_cn, cn_order)
    # jscatter codes only the levels present, in sorted order, and misaligns a
    # dict map against those codes, so colours shifted from sample to sample.
    # A list in that same order pins every level to its own colour.
    color_map6 = [palette6[level] for level in sorted(set(cn_tbl6["copy_number"].to_pylist()))]

    # Same empty-table guard as panel 5.
    scatter6 = None
    if cn_tbl6.num_rows:
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
    panel6_widget = None if scatter6 is None else scatter6.widget

    return cn_tbl6, color_map6, palette6, panel6_widget, scatter6


@app.cell
def _(loci, membership):
    # The split membership artifact resolves cCRE identities and copy number
    # (joined on rDHS).
    ccre_meta = None
    if membership is not None and loci.height:
        ccre_meta = loci.join(
            membership,
            left_on="rDHS",
            right_on="cCRE_name",
            how="inner",
        )
    return (ccre_meta,)


@app.cell
def _(biosample, ccre_meta, functools, read_zscores):
    # ccre_meta plus the detail tier's cancer and healthy z-scores and panel
    # 5/6 grid cell (joined on reference_index). Cells call this rather than
    # depend on a frame, so only the ones that need per-cCRE values wait on
    # the detail read.
    @functools.cache
    def read_ccre_frame():
        if ccre_meta is None:
            return None
        zscores = read_zscores(biosample)
        if zscores is None:
            return ccre_meta
        return ccre_meta.join(
            zscores.select("reference_index", "cancer_z", "healthy_z", "bin_id"),
            on="reference_index",
            how="left",
        )

    return (read_ccre_frame,)


@app.cell
def _(ccre_frame, cnv, get_lasso, get_window, get_zrange, pl):
    # The tab 1/2 genomic selection: the brushed window, narrowed by the
    # panel-2 z-score band, and to the cCREs inside a panel-2 lasso. None
    # while nothing is brushed.
    _window, _zrange, _lasso = get_window(), get_zrange(), get_lasso()
    genomic_rows = None
    if ccre_frame is not None and (_window != cnv.FULL_GENOME or _zrange is not None):
        genomic_rows = ccre_frame.filter(pl.col("pos").is_between(*_window))
        if _zrange is not None and "cancer_z" in genomic_rows.columns:
            genomic_rows = genomic_rows.filter(pl.col("cancer_z").is_between(*_zrange))
        if _lasso is not None and "cancer_z" in genomic_rows.columns:
            genomic_rows = genomic_rows.filter(
                cnv.in_polygon(
                    genomic_rows["pos"].to_numpy(),
                    genomic_rows["cancer_z"].fill_null(float("nan")).to_numpy(),
                    *_lasso,
                )
            )
    return (genomic_rows,)


@app.cell
def _(
    dens_tbl,
    lasso_indices,
    panel5_widget,
    selection_sync,
    set_membership_selection,
    set_sel_bins,
    set_table_source,
):
    # A panel-5 lasso sets the shared selection directly, as bin coordinates.
    def sync5(change):
        idx = lasso_indices(change.get("new"))
        selected = tuple(int(value) for value in idx.tolist())
        if selection_sync["panel5"] == selected:
            selection_sync["panel5"] = None
            return
        rows = dens_tbl.take(idx) if len(idx) else dens_tbl.slice(0, 0)
        set_sel_bins(frozenset(zip(rows["cancer_z"].to_pylist(), rows["healthy_z"].to_pylist())))
        keys = tuple(sorted({int(value) for value in rows["bin_id"].to_pylist()}))
        set_membership_selection(("bins", keys) if keys else None)
        set_table_source("panel 5/6" if keys else None)

    if panel5_widget is not None:
        panel5_widget.observe(sync5, names="selection")
    return (sync5,)


@app.cell
def _(
    cn_tbl6,
    dens_inv,
    dens_tbl,
    lasso_indices,
    np,
    panel6_widget,
    selection_sync,
    set_membership_selection,
    set_sel_bins,
    set_table_source,
):
    # M3b: a panel-6 lasso maps to panel-5 bins through dens_inv before
    # joining the shared selection.
    def sync6(change):
        idx = lasso_indices(change.get("new"))
        selected = tuple(int(value) for value in idx.tolist())
        if selection_sync["panel6"] == selected:
            selection_sync["panel6"] = None
            return
        bin_pos = np.unique(dens_inv[idx]) if len(idx) else np.array([], dtype=np.int64)
        rows = dens_tbl.take(bin_pos) if len(bin_pos) else dens_tbl.slice(0, 0)
        set_sel_bins(frozenset(zip(rows["cancer_z"].to_pylist(), rows["healthy_z"].to_pylist())))
        selected_rows = cn_tbl6.take(idx) if len(idx) else cn_tbl6.slice(0, 0)
        keys = tuple(
            sorted(
                {
                    (int(bin_id), str(copy_number))
                    for bin_id, copy_number in zip(
                        selected_rows["bin_id"].to_pylist(),
                        selected_rows["copy_number"].to_pylist(),
                        strict=True,
                    )
                }
            )
        )
        set_membership_selection(("points", keys) if keys else None)
        set_table_source("panel 5/6" if keys else None)

    if panel6_widget is not None:
        panel6_widget.observe(sync6, names="selection")
    return (sync6,)


@app.cell
def _(
    active_cn,
    cn_tbl6,
    cnv,
    dens_inv,
    dens_tbl,
    genomic_rows,
    get_membership_selection,
    get_sel_bins,
    np,
    pl,
):
    # Resolve Panel 5/6 selections against the current tables. A lasso always
    # wins; without one, a tab 1/2 genomic selection is shown on panels 5/6
    # through each cCRE's bin_id, never written back as a lasso.
    sel_bins = get_sel_bins()
    membership_selection = get_membership_selection()
    lasso_active = membership_selection is not None or bool(sel_bins)
    if sel_bins:
        pairs5 = list(zip(dens_tbl["cancer_z"].to_pylist(), dens_tbl["healthy_z"].to_pylist()))
        lasso_bin_pos = np.array([i for i, pair in enumerate(pairs5) if pair in sel_bins], dtype=np.int64)
    else:
        lasso_bin_pos = np.array([], dtype=np.int64)

    if membership_selection and membership_selection[0] == "points":
        point_keys = set(membership_selection[1])
        lasso_cn6_pos = np.asarray(
            [
                i
                for i, (bin_id, copy_number) in enumerate(
                    zip(
                        cn_tbl6["bin_id"].to_pylist(),
                        cn_tbl6["copy_number"].to_pylist(),
                        strict=True,
                    )
                )
                if (int(bin_id), str(copy_number)) in point_keys
            ],
            dtype=np.int64,
        )
    else:
        lasso_cn6_pos = (
            np.flatnonzero(np.isin(dens_inv, lasso_bin_pos))
            if len(lasso_bin_pos)
            else np.array([], dtype=np.int64)
        )

    if lasso_active:
        bin_pos, cn6_pos = lasso_bin_pos, lasso_cn6_pos
    elif genomic_rows is not None and genomic_rows["bin_id"].null_count() < genomic_rows.height:
        bin_pos, cn6_pos = cnv.scatter_positions(
            genomic_rows.filter(
                pl.col("copy_number").is_in([cnv.cn_label(level) for level in active_cn])
            ),
            dens_tbl,
            cn_tbl6,
        )
    else:
        bin_pos = cn6_pos = np.array([], dtype=np.int64)
    selection_active = lasso_active or len(bin_pos) > 0

    # D25: show whole-sample stats until a top-window or lasso selection is active.
    selected_dens = dens_tbl.take(bin_pos) if selection_active else dens_tbl
    selected_cn6 = cn_tbl6.take(cn6_pos) if selection_active else cn_tbl6
    return (
        bin_pos,
        cn6_pos,
        membership_selection,
        selected_cn6,
        selected_dens,
        sel_bins,
        selection_active,
    )


@app.cell
def _(bin_pos, panel5_widget, selection_active, selection_sync):
    # Apply the resolved selection back onto panel 5 — first render, a
    # CN-filter rebuild, or a panel-6-driven cross-highlight (M3b) all land
    # here. Guarded so a no-op assignment can't retrigger sync5 in a loop
    # (matching nmf_sample_explorer.py:389-392).
    _target = bin_pos.tolist() if selection_active else []
    _current = None if panel5_widget is None else panel5_widget.selection
    if panel5_widget is not None and list([] if _current is None else _current) != _target:
        selection_sync["panel5"] = tuple(_target)
        panel5_widget.selection = _target
    return


@app.cell
def _(cn6_pos, panel6_widget, selection_active, selection_sync):
    # M3c: apply the resolved selection back onto panel 6, same guard as panel 5.
    _target = cn6_pos.tolist() if selection_active else []
    _current = None if panel6_widget is None else panel6_widget.selection
    if panel6_widget is not None and list([] if _current is None else _current) != _target:
        selection_sync["panel6"] = tuple(_target)
        panel6_widget.selection = _target
    return


@app.cell
def _(cnv, membership_error, mo):
    def ccre_view(frame, message):
        # One rendering for every tab's table: a status line, then up to
        # 2,000 rows of cnv.TABLE_COLUMNS (rDHS is never shown).
        if frame is None:
            return mo.md(message)
        columns = [name for name in cnv.TABLE_COLUMNS if name in frame.columns]
        return mo.vstack(
            [
                mo.md(f"{message}: **{frame.height:,} cCREs** (showing up to 2,000 rows)"),
                mo.ui.table(frame.select(columns).head(2_000).to_pandas(), selection=None),
            ],
            gap=1,
        )

    ccre_unavailable = (
        f"**cCRE table unavailable:** {membership_error}. "
        "Add `ccre_reference.parquet` and `summary_ccre_membership.parquet`."
    )
    return ccre_unavailable, ccre_view


@app.cell
def _(ccre_frame, ccre_unavailable, ccre_view, cnv, get_window, hide_sentinel, pl):
    _window = get_window()
    if ccre_frame is None:
        table1 = ccre_view(None, ccre_unavailable)
    elif _window == cnv.FULL_GENOME:
        table1 = ccre_view(None, "Brush panel 1 to list the cCREs in a genomic window.")
    else:
        _rows = ccre_frame.filter(pl.col("pos").is_between(*_window))
        if hide_sentinel.value:
            _rows = cnv.drop_sentinel(_rows)
        table1 = ccre_view(_rows, "cCREs in the brushed window")
    return (table1,)


@app.cell
def _(
    ccre_frame,
    ccre_unavailable,
    ccre_view,
    cnv,
    genomic_rows,
    get_zrange,
    hide_sentinel,
):
    if ccre_frame is None:
        table2 = ccre_view(None, ccre_unavailable)
    elif genomic_rows is None:
        table2 = ccre_view(
            None, "Box-select on panel 2 to list cCREs by genomic window and z-score."
        )
    elif get_zrange() is not None and "cancer_z" not in ccre_frame.columns:
        table2 = ccre_view(None, "z-score selection needs the panel 2 detail tier.")
    else:
        _rows = cnv.drop_sentinel(genomic_rows) if hide_sentinel.value else genomic_rows
        table2 = ccre_view(_rows, "cCREs in the panel 2 selection")
    return (table2,)


@app.cell
def _(ccre_frame, ccre_unavailable, ccre_view, cnv, hide_sentinel, picked_cn, pl):
    if ccre_frame is None:
        table3 = ccre_view(None, ccre_unavailable)
    elif not picked_cn:
        table3 = ccre_view(None, "Click bars in panel 3 to list the cCREs at those copy numbers.")
    else:
        _rows = ccre_frame.filter(
            pl.col("copy_number").is_in([cnv.cn_label(level) for level in picked_cn])
        )
        if hide_sentinel.value:
            _rows = cnv.drop_sentinel(_rows)
        table3 = ccre_view(
            _rows,
            f"cCREs at copy number {', '.join(cnv.cn_label(level) for level in picked_cn)}",
        )
    return (table3,)


@app.cell
def _(
    active_cn,
    ccre_frame,
    ccre_unavailable,
    ccre_view,
    cnv,
    hide_sentinel,
    membership_selection,
    meta,
):
    # A panel 5/6 lasso resolves to cCREs through each cCRE's bin_id, which
    # load_zscores computes with the writer's own grid binning.
    if ccre_frame is None:
        table4 = ccre_view(None, ccre_unavailable)
    elif not meta.get("healthy_available") or "bin_id" not in ccre_frame.columns:
        table4 = ccre_view(
            None,
            "No per-cCRE healthy z-scores for this sample "
            f"({meta.get('healthy_status')}), so panel 5/6 cells cannot list cCREs.",
        )
    elif membership_selection is None:
        table4 = ccre_view(None, "Lasso points in panel 5 or 6 to list their cCREs.")
    else:
        _rows = cnv.selection_ccres(ccre_frame, membership_selection, active_cn)
        if hide_sentinel.value:
            _rows = cnv.drop_sentinel(_rows, ("cancer_z", "healthy_z"))
        table4 = ccre_view(_rows, "cCREs in the panel 5/6 lasso")
    return (table4,)


@app.cell
def _(meta, mo):
    # Drawn in place of an empty panel 5/6.
    if meta.get("healthy_available"):
        scatter_note = mo.md("*No cCREs at the picked copy numbers.*")
    else:
        scatter_note = mo.md(
            f"*No healthy ATAC reference for this sample ({meta.get('healthy_status')}): "
            "panels 5–6 and the healthy violins are empty.*"
        )
    return (scatter_note,)


@app.cell
def _(mo, panel3, panel4, sections):
    (
        mo.hstack([panel3, panel4], widths="equal", gap=1.5)
        if sections.value == "CN counts & distributions"
        else None
    )
    return


@app.cell
def _(biosample, mo, panel5_widget, panel6_widget, scatter_note, sections):
    (
        mo.hstack(
            [
                mo.vstack(
                    [
                        mo.md(f"### 5 · Aggregated cCRE density — {biosample}"),
                        scatter_note if panel5_widget is None else panel5_widget,
                    ]
                ),
                mo.vstack(
                    [
                        mo.md(f"### 6 · Copy-number-stratified z-score scatter — {biosample}"),
                        scatter_note if panel6_widget is None else panel6_widget,
                    ]
                ),
            ],
            widths="equal",
            gap=1.5,
        )
        if sections.value == "Cancer vs healthy"
        else None
    )
    return


@app.cell
def _(read_ccre_frame):
    # Registered after the panel cells: marimo runs ready cells in file order,
    # so the panels render before this waits on the detail read.
    ccre_frame = read_ccre_frame()
    return (ccre_frame,)


@app.cell
def _(get_table_source, table1, table2, table3, table4):
    # The last selection's table, under the tab strip and kept across tabs;
    # nothing until a first selection. Kept out of the tabs cell so a
    # selection redraws only this table, never the panels.
    {
        "panel 1": table1,
        "panel 2": table2,
        "panel 3": table3,
        "panel 5/6": table4,
    }.get(get_table_source())
    return


if __name__ == "__main__":
    app.run()
