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
    import pandas as pd

    import epifolio.cnv as cnv

    return cnv, jscatter, mo, np, pd


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
def _(active_cn, cn_order, cnv, jscatter, mo, p6_row):
    # D6: density_table now does np.unique aggregation, producing fewer rows
    dens_tbl = cnv.density_table(p6_row, active_cn, cn_order)

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

    # H7: reactive selection — mo.state + widget.observe + mo.ui.anywidget
    get_sel5, set_sel5 = mo.state(None)
    scatter5.widget.observe(lambda change: set_sel5(change["new"]), names="selection")
    panel5_widget = mo.ui.anywidget(scatter5.widget)

    return dens_tbl, get_sel5, panel5_widget, scatter5, set_sel5


@app.cell
def _(dens_tbl, get_sel5, lasso_indices, mo, panel5_widget):
    sel5_raw = get_sel5()
    sel5 = lasso_indices(sel5_raw)

    # D25: show whole-sample stats until user lassoes
    selected_dens = dens_tbl.take(sel5) if len(sel5) else dens_tbl

    # cCRE-weighted gain fraction
    cz5 = selected_dens["cancer_z"].to_pylist()
    hz5 = selected_dens["healthy_z"].to_pylist()
    w5 = selected_dens["cCREs"].to_pylist()
    total_w5 = sum(w5) if w5 else 1
    gain_w5 = sum(wi for ci, hi, wi in zip(cz5, hz5, w5) if ci > hi)
    gain5 = float(gain_w5) / max(total_w5, 1)

    readout5 = mo.hstack(
        [
            mo.stat(value=f"{sum(w5):,}", label="cCREs in selection"),
            mo.stat(value=f"{gain5:.1%}", label="gain fraction (cCRE-weighted)"),
        ]
    )
    panel5 = mo.vstack([panel5_widget, readout5], gap=1)
    return cz5, gain5, gain_w5, hz5, panel5, readout5, sel5, sel5_raw, selected_dens, total_w5, w5


@app.cell
def _(active_cn, cn_order, cnv, jscatter, mo, p6_row):
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

    # H7: reactive selection for panel 6
    get_sel6, set_sel6 = mo.state(None)
    scatter6.widget.observe(lambda change: set_sel6(change["new"]), names="selection")
    panel6_widget = mo.ui.anywidget(scatter6.widget)

    return cn_tbl6, color_map6, get_sel6, palette6, panel6_widget, scatter6, set_sel6


@app.cell
def _(cn_tbl6, get_sel6, lasso_indices, mo, panel6_widget, pd):
    sel6_raw = get_sel6()
    sel6 = lasso_indices(sel6_raw)

    # D25: show whole-sample stats until user lassoes
    selected_cn6 = cn_tbl6.take(sel6) if len(sel6) else cn_tbl6

    # cCRE-weighted counts6: aggregate cCREs per level
    levels6 = selected_cn6["copy_number"].to_pylist()
    ccres6 = selected_cn6["cCREs"].to_pylist()
    agg6: dict[str, int] = {}
    for lv6, cv6 in zip(levels6, ccres6):
        agg6[lv6] = agg6.get(lv6, 0) + int(cv6)
    counts6 = pd.DataFrame(
        [{"copy_number": k, "cCREs": v} for k, v in sorted(agg6.items())]
    )
    total6 = int(sum(agg6.values()))

    readout6 = mo.vstack(
        [
            mo.stat(value=f"{total6:,}", label="cCREs in selection"),
            mo.ui.table(counts6, selection=None),
        ],
        gap=1,
    )
    panel6 = mo.vstack([panel6_widget, readout6], gap=1)
    return (
        agg6,
        counts6,
        cv6,
        levels6,
        lv6,
        panel6,
        readout6,
        sel6,
        sel6_raw,
        selected_cn6,
        total6,
    )


@app.cell
def _(aliquot_dd, cancer_dd, mo, panel1, panel2, panel3, panel4, panel5, panel6, patient_dd):
    mo.vstack(
        [
            mo.hstack([cancer_dd, patient_dd, aliquot_dd], justify="start", gap=1.5),
            panel1,
            panel2,
            mo.hstack([panel3, panel4], widths="equal", gap=1.5),
            mo.hstack([panel5, panel6], widths="equal", gap=1.5),
        ],
        gap=2,
    )
    return


if __name__ == "__main__":
    app.run()
