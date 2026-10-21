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
    import polars as pl

    import epifolio.cnv as cnv

    return cnv, jscatter, mo, np, pc, pl


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

    @mo.cache
    def read_cancer_z(biosample: str):
        if detail_index is None:
            return None
        return cnv.load_cancer_z(detail_index, biosample)

    return detail_error, detail_index, read_cancer_z


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
    hide_sentinel = mo.ui.switch(value=False, label="Hide −10 (no ATAC signal)")
    return (hide_sentinel,)


@app.cell
def _(biosample, read_row):
    meta = read_row(biosample, "header", "shared")
    cn_order = [str(level) for level in meta["cn_order"]]
    return cn_order, meta


@app.cell
def _(biosample, ccre_reference, ccre_reference_error, cn_order, cnv, loci, loci_error, mo):
    @mo.cache
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
def _(
    biosample,
    clamp_window,
    cnv,
    get_window,
    hide_sentinel,
    mo,
    pl,
    read_cancer_z,
    read_row,
    set_window,
):
    p2_row = read_row(biosample, "2")
    _window = get_window()
    # Raw per-cCRE values replace the binned overview once the window holds
    # at most DETAIL_MAX_ROWS cCREs.
    p2_detail = None
    _cancer_z = read_cancer_z(biosample)
    if _cancer_z is not None:
        _in_window = _cancer_z.filter(pl.col("pos").is_between(*_window))
        if _in_window.height <= cnv.DETAIL_MAX_ROWS:
            p2_detail = _in_window

    def sync2(_value):
        moved = clamp_window(panel2.ranges)
        if moved:
            set_window(moved)

    panel2 = mo.ui.plotly(
        cnv.panel_2(
            p2_row,
            _window,
            biosample,
            hide_sentinel=hide_sentinel.value,
            detail=p2_detail,
        ),
        on_change=sync2,
        config={"responsive": True},
    )
    return p2_detail, panel2, p2_row, sync2


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
def _(cnv, get_window, loci, membership):
    # A top-panel brush is a genomic-window selection. Resolve it through the
    # split membership artifact into cCRE identities and copy-number labels.
    # This artifact has no per-cCRE Panel 5/6 z-score bin, so it does not
    # manufacture scatter-panel highlights from genomic selection.
    window = get_window()
    top_selection_active = window != cnv.FULL_GENOME
    if membership is not None and loci.height:
        window_loci = loci.filter(
            (loci["pos"] >= window[0]) & (loci["pos"] <= window[1])
        )
        top_ccres = window_loci.join(
            membership,
            left_on="rDHS",
            right_on="cCRE_name",
            how="inner",
        )
    else:
        top_ccres = loci.head(0)
    return top_ccres, top_selection_active


@app.cell
def _(dens_tbl, lasso_indices, panel5_widget, selection_sync, set_membership_selection, set_sel_bins):
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

    panel6_widget.observe(sync6, names="selection")
    return (sync6,)


@app.cell
def _(
    cn_tbl6,
    dens_inv,
    dens_tbl,
    get_membership_selection,
    get_sel_bins,
    np,
):
    # Resolve Panel 5/6 selections against the current tables. The split cCRE
    # membership artifact has no scatter-bin membership, so genomic selection
    # remains a separate cCRE-table filter; it must not clear or remap lasso
    # selections in the raster panels.
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

    bin_pos = lasso_bin_pos if lasso_active else np.array([], dtype=np.int64)
    cn6_pos = lasso_cn6_pos if lasso_active else np.array([], dtype=np.int64)
    selection_active = lasso_active

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
    _current = panel5_widget.selection
    if list([] if _current is None else _current) != _target:
        selection_sync["panel5"] = tuple(_target)
        panel5_widget.selection = _target
    return


@app.cell
def _(cn6_pos, panel6_widget, selection_active, selection_sync):
    # M3c: apply the resolved selection back onto panel 6, same guard as panel 5.
    _target = cn6_pos.tolist() if selection_active else []
    _current = panel6_widget.selection
    if list([] if _current is None else _current) != _target:
        selection_sync["panel6"] = tuple(_target)
        panel6_widget.selection = _target
    return


@app.cell
def _(
    active_cn,
    cnv,
    membership,
    pl,
    top_ccres,
    top_selection_active,
):
    selected_ccres = None
    selected_ccre_preview = None
    selected_ccre_count = 0
    selected_ccre_status = "Make a genomic brush or Panel 5/6 lasso to list cCREs"

    if membership is None:
        selected_ccre_status = "split cCRE reference/membership artifacts unavailable"
    elif not top_selection_active:
        selected_ccre_status = (
            "Make a Panel 1/2 genomic brush; Panel 5/6 raster selections do not "
            "carry per-cCRE scatter membership"
        )
    else:
        selected_ccres = top_ccres
        selected_ccres = selected_ccres.filter(
            pl.col("copy_number").is_in([cnv.cn_label(level) for level in active_cn])
        )

        selected_ccre_count = selected_ccres.height
        selected_ccre_preview = (
            selected_ccres.select(
                [
                    "cCRE",
                    "rDHS",
                    "chrom",
                    "start",
                    "end",
                    "class",
                    "copy_number",
                    "copy_number_code",
                    "x_bin",
                ]
            )
            .head(2_000)
            .to_pandas()
        )
        selected_ccre_status = (
            f"{selected_ccre_count:,} selected cCREs; showing up to 2,000 rows"
        )

    return selected_ccre_count, selected_ccre_preview, selected_ccre_status


@app.cell
def _(
    mo,
    selected_ccre_count,
    selected_ccre_preview,
    selected_ccre_status,
):
    ccre_table = (
        mo.ui.table(selected_ccre_preview, selection=None)
        if selected_ccre_preview is not None
        else mo.md(
            f"**Selected cCRE table unavailable:** {selected_ccre_status}. "
            "Add `ccre_reference.parquet` and `summary_ccre_membership.parquet`."
        )
    )

    # The selected cCRE table is driven by the genomic window and split
    # membership artifacts. Keep the stale aggregate Panel 5/6 readout out of
    # this section; Panel 5/6 retain their own visual lasso linking above.
    readout56 = mo.vstack(
        [
            mo.md(f"{selected_ccre_status} (total: {selected_ccre_count:,})"),
            ccre_table,
        ],
        gap=1,
    )
    return (readout56,)


@app.cell
def _(mo):
    # Remembers the open tab: the tabs cell re-runs (and would reset) whenever a panel changes.
    get_tab, set_tab = mo.state("Copy number")
    return get_tab, set_tab


@app.cell
def _(
    aliquot_dd,
    biosample,
    cancer_dd,
    detail_error,
    get_tab,
    hide_sentinel,
    mo,
    panel1,
    panel2,
    panel3,
    panel4,
    panel5_widget,
    panel6_widget,
    patient_dd,
    readout56,
    set_tab,
):
    sections = mo.ui.tabs(
        {
            "Copy number": panel1,
            "ATAC z-score": mo.vstack(
                [
                    mo.md(f"*Panel 2 detail tier unavailable: {detail_error}*")
                    if detail_error
                    else hide_sentinel,
                    panel2,
                ]
            ),
            "CN counts & distributions": mo.hstack(
                [panel3, panel4], widths="equal", gap=1.5
            ),
            "Cancer vs healthy": mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(f"### 5 · Aggregated cCRE density — {biosample}"),
                            panel5_widget,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(
                                f"### 6 · Copy-number-stratified z-score scatter — {biosample}"
                            ),
                            panel6_widget,
                        ]
                    ),
                ],
                widths="equal",
                gap=1.5,
            ),
        },
        value=get_tab(),
        on_change=set_tab,
        lazy=True,
    )
    mo.vstack(
        [
            mo.hstack([cancer_dd, patient_dd, aliquot_dd], justify="start", gap=1.5),
            sections,
            readout56,
        ],
        gap=2,
    )
    return (sections,)


if __name__ == "__main__":
    app.run()
