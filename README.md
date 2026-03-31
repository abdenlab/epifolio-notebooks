# epifolio-notebooks

This is a standalone GitHub repo for developing and deploying Marimo notebooks that explore cancer regulatory genomics data.

Three use cases to support:
1. **Development**: Edit notebooks + shared modules locally
2. **Molab deployment**: Single `.py` file bootstraps its own modules/data
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
│           ├── scatter.py
│           ├── sort_utils.py
│           └── assets/            # Bundled as package data
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

When run via `uvx marimo run` or on Molab, `uv` reads this and installs all deps — including `epifolio` itself — automatically. **No custom bootstrap cell needed.**

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

## Developer Workflow

```bash
git clone https://github.com/abdenlab/epifolio-notebooks
cd epifolio-notebooks
uv sync                                          # Install deps (epifolio as editable)
uv run marimo edit notebooks/nmf_explorer.py     # Dev with hot reload
```

The root `pyproject.toml` includes `epifolio` as an editable dependency (`lib/`), so changes to shared modules are reflected immediately without reinstalling.

## End-User Workflow

```bash
# Option A: Run directly from URL (no clone needed)
uvx marimo run https://raw.githubusercontent.com/abdenlab/epifolio-notebooks/main/notebooks/nmf_explorer.py

# Option B: Clone and run
git clone https://github.com/abdenlab/epifolio-notebooks && cd epifolio-notebooks
uvx marimo run notebooks/nmf_explorer.py
```

In both cases, `uv` reads PEP 723 metadata and installs `epifolio` from the git repo automatically.

## Molab Workflow

1. Upload `notebooks/nmf_explorer.py` to Molab
2. PEP 723 handles all dependency installation, including `epifolio` from git

## Agents

We include useful agent skills and a lock file following the open agents skills (skills.sh) framework. You may need to perform the appropriate symlinking for your vendor.