# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "higlass-python>=1.4.0",
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

    from epifolio.higlass_utils import (
        INDEX_VIEWCONF_PATH,
        inject_plugin_urls,
        load_index_viewconf,
    )

    return INDEX_VIEWCONF_PATH, inject_plugin_urls, load_index_viewconf, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    # HiGlass index explorer

    Renders the bundled `higlass_index_viewconf.json` (TCGA bulk multivec
    tracks hosted on resgen.io) with `higlass-python`. The view config ships
    as package data inside `epifolio`, so no network fetch is needed beyond
    the HiGlass tile servers referenced by the tracks.
    """
    )
    return


@app.cell
def _(load_index_viewconf):
    viewconfig = load_index_viewconf()
    return (viewconfig,)


@app.cell
def _(inject_plugin_urls, viewconfig):
    normalized_viewconfig = inject_plugin_urls(viewconfig)
    return (normalized_viewconfig,)


@app.cell
def _(normalized_viewconfig):
    import higlass as hg

    viewconf = hg.Viewconf(**normalized_viewconfig)
    widget = viewconf.widget()

    return viewconf, widget


@app.cell(hide_code=True)
def _(widget):
    widget
    return


if __name__ == "__main__":
    app.run()
