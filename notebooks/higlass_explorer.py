# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.3",
#     "higlass-python>=1.4.0",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import json

    import marimo as mo

    from epifolio.higlass import INDEX_VIEWCONF_PATH, inject_plugin_urls

    return INDEX_VIEWCONF_PATH, inject_plugin_urls, json, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # HiGlass index explorer

    Renders the bundled `higlass_index_viewconf.json` (TCGA bulk multivec
    tracks hosted on resgen.io) with `higlass-python`. The view config ships
    as package data inside `epifolio`, so no network fetch is needed beyond
    the HiGlass tile servers referenced by the tracks.
    """)
    return


@app.cell
def _(INDEX_VIEWCONF_PATH, json):
    # Explicit read of the bundled view config — visible data flow.
    viewconfig = json.loads(INDEX_VIEWCONF_PATH.read_text(encoding="utf-8"))
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
    return (widget,)


@app.cell(hide_code=True)
def _(widget):
    widget
    return


if __name__ == "__main__":
    app.run()
