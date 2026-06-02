# epifolio-notebooks

This is a standalone GitHub repo for developing and deploying Marimo notebooks that explore cancer regulatory genomics data.

Three use cases to support:
1. **Development**: Edit notebooks + shared modules locally
2. **Molab deployment**: Single `.py` file installs `epifolio` and reads packaged assets
3. **Local user launch**: `uvx marimo run <notebook>` with zero repo setup

### Repo Structure

```
epifolio-notebooks/
├── notebooks/                      # The Marimo notebooks
│   ├── nmf_explorer.py            # PEP 723 metadata (no bootstrap cell needed)
│   ├── higlass_browser.py
│   ├── pca_hiplot.py
│   └── ccre_scatter.py
├── lib/                            # The epifolio package (installed via PEP 723)
│   ├── pyproject.toml             # Package metadata for epifolio
│   └── src/
│       └── epifolio/
│           ├── __init__.py
│           ├── color_utils.py
│           ├── data_utils.py
│           ├── heatmap.py
│   ├── itcr_embedding_utils.py
│           ├── sort_utils.py
│   ├── upt_heatmap.py
│   └── assets/            # Bundled as package data
│               ├── conf/          # JSON configs, color maps, vocabularies
│               └── data/          # Small datasets (<~20MB total)
├── pyproject.toml                  # Dev dependencies + editable install of lib/
├── .python-version
└── README.md
```

Each notebook is self-contained. Two mechanisms make this work:

### 1. PEP 723 inline script metadata (all dependencies including epifolio)

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "jupyter-scatter[all]>=0.22.0",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
#     ...
# ]
# ///
```

When run via `uvx marimo run` or on Molab, `uv` reads this and installs all deps — including `epifolio` itself — automatically. **No custom bootstrap cell or source-repo asset downloader is needed.**

> **Migration to PyPI**: When ready for primetime, replace the git URL with just `epifolio` (or `epifolio>=x.y`) and publish to PyPI. One-line change per notebook.

### 2. Assets as package data

Small assets (configs, color maps, small datasets) are bundled inside the `epifolio` package under `epifolio/assets/`. Notebooks access them via `importlib.resources`:

```python
from importlib.resources import files

ASSETS = files("epifolio") / "assets"
config = json.loads((ASSETS / "conf" / "config.json").read_text())
```

This works identically in local dev (editable install) and standalone (pip-installed from git).

### Large datasets

Anything too large for git or the package data bundle gets hosted separately:
```
http://projects.abdenlab.org/itcr/epifolio/...
```
Some things can be fetched lazily by `epifolio.data_utils` on first access, cached locally.

In other cases, we may access large assets like Parquet files using appropriate tools like Polars that can handle streaming and random access as needed directly from URLs.

## Notebook surface

- `notebooks/nmf_explorer.py` — multi-analysis TCGA NMF explorer with linked UMAP, heatmap, grandscatter, and metadata views
- `notebooks/pca_hiplot.py` — PCA result explorer with HiPlot and linked metadata views
- `notebooks/higlass_browser.py` — packaged HiGlass browser for the published multivec viewconfig

## Developer Workflow

```bash
git clone https://github.com/abdenlab/epifolio-notebooks
cd epifolio-notebooks
uv sync                                          # Install deps (epifolio as editable)
uv run marimo edit notebooks/nmf_explorer.py     # Dev with hot reload
uv run marimo edit notebooks/pca_hiplot.py
uv run marimo edit notebooks/higlass_browser.py
```

The root `pyproject.toml` includes `epifolio` as an editable dependency (`lib/`), so changes to shared modules are reflected immediately without reinstalling.

## End-User Workflow

```bash
# Option A: Run directly from URL (no clone needed)
uvx marimo run https://raw.githubusercontent.com/abdenlab/epifolio-notebooks/main/notebooks/nmf_explorer.py
uvx marimo run https://raw.githubusercontent.com/abdenlab/epifolio-notebooks/main/notebooks/pca_hiplot.py
uvx marimo run https://raw.githubusercontent.com/abdenlab/epifolio-notebooks/main/notebooks/higlass_browser.py

# Option B: Clone and run
git clone https://github.com/abdenlab/epifolio-notebooks && cd epifolio-notebooks
uvx marimo run notebooks/nmf_explorer.py
uvx marimo run notebooks/pca_hiplot.py
uvx marimo run notebooks/higlass_browser.py
```

In both cases, `uv` reads PEP 723 metadata and installs `epifolio` from the git repo automatically.

## Molab Workflow

1. Upload `notebooks/nmf_explorer.py`, `notebooks/pca_hiplot.py`, or `notebooks/higlass_browser.py` to Molab
2. PEP 723 handles dependency installation, including `epifolio` from git
3. Shared configs and small datasets are loaded from `epifolio/assets`

## Agents

We include useful agent skills and a lock file following the open agents skills (skills.sh) framework. You may need to perform the appropriate symlinking for your vendor.