# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "marimo>=0.23.11",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "polars==1.42.0",
#     "dtour>=0.4.4",
# ]
# ///

import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import polars as pl
    import dtour

    return dtour, pl


@app.cell
def _():
    DOWNSAMPLE = True
    N_PCs = 14
    return DOWNSAMPLE, N_PCs


@app.cell
def _(pl):
    df = pl.scan_parquet("https://projects.abdenlab.org/itcr/epifolio/embeddings/tcga.atac.pca.ccre.pq")
    return (df,)


@app.cell
def _(df, pl):
    n_rows = df.select(pl.len()).collect().item()
    n_rows
    return (n_rows,)


@app.cell
def _(DOWNSAMPLE, N_PCs, df, n_rows, pl):
    if DOWNSAMPLE:
        stride = max(1, n_rows // 100_000)
        sample = df.gather_every(stride).collect()
        sample = sample.with_columns(
            (pl.col(f"PC{i+1}") - pl.median(f"PC{i+1}")) for i in range(N_PCs)
        )
    else:
        sample = df.collect()

    sample
    return (sample,)


@app.cell
def _(N_PCs, dtour, sample):
    widget = dtour.Widget(
        data=sample.select([*[f"PC{i+1}" for i in range(N_PCs)], "class"]),
        # preprocess="mean_center"
    )
    widget
    return


@app.cell
def _():
    # sample[widget.selected_indices]
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
