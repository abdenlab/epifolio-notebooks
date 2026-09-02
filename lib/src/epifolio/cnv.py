"""CNV/cCRE dashboard: payload contract, data access, genome coordinates, colors, panels.

Companion module to ``notebooks/cnv_ccre_explorer.py``. Sections in dependency
order: schema constants, hg38 coordinates, copy-number colors, payload reads,
then the pure plotly/Arrow panel builders.
"""

from __future__ import annotations

import base64
import json
import urllib.request
from dataclasses import dataclass
from importlib.metadata import files
from itertools import accumulate
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import pyarrow as pa

from epifolio import ASSETS
from epifolio.colors import load_color_map

# ---------------------------------------------------------------------------
# Payload schema and metadata contract
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "3"
N_VIOLIN_BINS = 256
N_GRID = 192
Z_LIM = (-10.5, 5.5)

VIOLIN_EDGES = np.linspace(Z_LIM[0], Z_LIM[1], N_VIOLIN_BINS + 1).tolist()
GRID_EDGES = np.linspace(Z_LIM[0], Z_LIM[1], N_GRID + 1).tolist()
HIST2D_SHAPE = [N_GRID, N_GRID]

TRACKS = "summary_genome_tracks.parquet"
DISTRIBUTIONS = "summary_signal_distributions.parquet"


def cols_for(tags: Mapping[str, str], *panels: str) -> list[str]:
    """Return columns tagged as index/shared/header or any requested panel."""
    wanted = {"index", "shared", *panels}
    return [name for name, panel in tags.items() if panel in wanted]


def file_constants(metadata: Mapping[str, str | bytes]) -> dict[str, object]:
    """Parse payload constants from parquet file-level metadata."""
    parsed: dict[str, object] = {}
    for key in ("schema_version", "violin_edges", "grid_edges", "hist2d_shape"):
        value = metadata[key]
        if isinstance(value, bytes):
            value = value.decode()
        if key == "schema_version":
            parsed[key] = value
        else:
            parsed[key] = json.loads(value)
    return parsed


def assert_edges_match(metadata: Mapping[str, str | bytes]) -> None:
    """Fail loudly if the published payload constants drift from the code."""
    constants = file_constants(metadata)
    if constants["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"CNV payload schema_version={constants['schema_version']!r}; "
            f"expected {SCHEMA_VERSION!r}"
        )
    checks = {
        "violin_edges": VIOLIN_EDGES,
        "grid_edges": GRID_EDGES,
        "hist2d_shape": HIST2D_SHAPE,
    }
    for key, expected in checks.items():
        got = constants[key]
        if not np.allclose(got, expected):
            raise ValueError(f"CNV payload {key} does not match the notebook contract")


# ---------------------------------------------------------------------------
# hg38 chromosome coordinates for whole-genome linear plots
# ---------------------------------------------------------------------------
_BIOFRAME_FILES = files("bioframe")
if _BIOFRAME_FILES is None:
    raise RuntimeError("bioframe package metadata is unavailable")

_SEQINFO_PATH = next(
    file for file in _BIOFRAME_FILES if file.parts[-3:] == ("io", "data", "hg38.seqinfo.tsv")
).locate()
_SEQINFO = pd.read_table(_SEQINFO_PATH)
_PRIMARY = _SEQINFO[
    (_SEQINFO["role"] == "assembled")
    & (_SEQINFO["unit"] == "primary")
    & (_SEQINFO["name"] != "chrM")
]
CHROMS = _PRIMARY["name"].astype(str).tolist()

if len(CHROMS) != 24 or CHROMS[0] != "chr1":
    raise RuntimeError("Unexpected hg38 chromosome order from bioframe packaged data")

CHROM_SIZES = dict(
    zip(_PRIMARY["name"].astype(str), _PRIMARY["length"].astype(int), strict=True)
)
_starts = [0, *accumulate(CHROM_SIZES[chrom] for chrom in CHROMS[:-1])]
CHROM_OFFSET = dict(zip(CHROMS, _starts, strict=True))
GENOME_LEN = sum(CHROM_SIZES.values())
CHROM_MID = {
    chrom: CHROM_OFFSET[chrom] + CHROM_SIZES[chrom] // 2 for chrom in CHROMS
}
FULL_GENOME = (0.0, float(GENOME_LEN))


# ---------------------------------------------------------------------------
# Copy-number color utilities
# ---------------------------------------------------------------------------
MISSING_GREY = "#9e9e9e"
CANCER_BLUE = "#4c78a8"
HEALTHY_ORANGE = "#f58518"


def cn_label(level: object) -> str:
    """Normalize copy-number level labels across payload panels."""
    text = str(level)
    if text in {"4+", "Missing"}:
        return text
    try:
        return f"{float(text):.1f}"
    except ValueError:
        return text


def cn_colors(levels: Iterable[object]) -> dict[str, str]:
    """Return a {label: hex} map for copy-number levels using the packaged JSON map.

    Matching src/constants.py:169-170 from the reference toolkit.
    """
    color_map = load_color_map(ASSETS / "conf" / "copy_number_color_map.json")
    return {str(level): color_map.get(str(level), MISSING_GREY) for level in levels}


# ---------------------------------------------------------------------------
# Payload discovery and panel reads
# ---------------------------------------------------------------------------
DATA_DIR = ASSETS / "data" / "cnv_ccre"
# Canonical mirror. Not yet serving the payload, so source() falls through to FALLBACK.
REMOTE = "https://projects.abdenlab.org/itcr/epifolio/notebooks/cnv_ccre"
# Interim host. Delete this tier once REMOTE serves the payload; no other change needed.
FALLBACK = "https://users.abdenlab.org/voy/itcr/epifolio/notebooks/cnv_ccre"


def _reachable(url: str, timeout: float = 5.0) -> bool:
    """Return whether the canonical mirror is reachable."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def source(name: str) -> str:
    """Use the payload bundled with the installed package, else canonical mirror, else fallback."""
    local = DATA_DIR / name
    if local.exists():
        return str(local)
    remote = f"{REMOTE}/{name}"
    return remote if _reachable(remote) else f"{FALLBACK}/{name}"


def read_metadata(src: str) -> dict[str, str]:
    """Read parquet key-value metadata without scanning row data."""
    return pl.read_parquet_metadata(src)


def panel_tags(src: str) -> dict[str, str]:
    """Read {column: panel tag} from the payload's embedded Arrow schema."""
    blob = read_metadata(src)["ARROW:schema"]
    schema = pa.ipc.read_schema(pa.py_buffer(base64.b64decode(blob)))
    return {
        field.name: (field.metadata or {})[b"panel"].decode()
        for field in schema
        if field.metadata and b"panel" in field.metadata
    }


@dataclass(frozen=True)
class Dataset:
    """Panel tag to file to columns, with ownership read from file metadata."""

    sources: dict[str, str]
    tags: dict[str, dict[str, str]]
    metadata: dict[str, dict[str, str]]

    def source_for(self, panel: str) -> str:
        for src, tags in self.tags.items():
            if panel in set(tags.values()):
                return src
        raise KeyError(f"No CNV payload file owns panel {panel!r}")

    def tags_for(self, panel: str) -> dict[str, str]:
        return self.tags[self.source_for(panel)]

    def schema_for(self, panel: str) -> dict[str, str]:
        return self.metadata[self.source_for(panel)]


def open_dataset(sources: list[str]) -> Dataset:
    """Open the two-file payload and cache metadata needed for panel reads."""
    if len(sources) != 2:
        raise ValueError("CNV dataset expects exactly two parquet sources")
    return Dataset(
        sources={Path(src).name: src for src in sources},
        tags={src: panel_tags(src) for src in sources},
        metadata={src: read_metadata(src) for src in sources},
    )


def load_cohort(dataset: Dataset) -> dict[str, dict[str, list[str]]]:
    """Return cancer -> patient -> biosamples for dropdown construction."""
    src = dataset.source_for("index")
    rows = (
        pl.scan_parquet(src)
        .select(["cancer", "patient_id", "biosample"])
        .unique()
        .sort(["cancer", "patient_id", "biosample"])
        .collect()
        .to_dicts()
    )
    cohort: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        cancer = str(row["cancer"])
        patient = str(row["patient_id"])
        biosample = str(row["biosample"])
        cohort.setdefault(cancer, {}).setdefault(patient, []).append(biosample)
    return cohort


def load_panel(dataset: Dataset, biosample: str, *panels: str) -> dict[str, object]:
    """Load one biosample and the columns tagged for the requested panels."""
    out: dict[str, object] = {}
    for src in {dataset.source_for(panel) for panel in panels}:
        tags = dataset.tags[src]
        cols = cols_for(tags, *panels)
        row = (
            pl.scan_parquet(src)
            .filter(pl.col("biosample") == biosample)
            .select(cols)
            .collect()
            .to_arrow()
            .to_pylist()
        )
        if not row:
            raise KeyError(f"Biosample {biosample!r} not found in {src}")
        out.update(row[0])
    return out


# ---------------------------------------------------------------------------
# Chromosome tick positions and dashed divider shapes (src/panels.py:11-17)
# ---------------------------------------------------------------------------
CHROM_TICKVALS = [CHROM_OFFSET[c] + CHROM_SIZES[c] // 2 for c in CHROMS]
CHROM_TICKTEXT = [c.removeprefix("chr") for c in CHROMS]

# 24 dashed vertical lines at chromosome starts (src/panels.py:13-17)
CHROM_BOUNDS = [
    dict(
        type="line",
        xref="x",
        yref="paper",
        x0=CHROM_OFFSET[chrom],
        x1=CHROM_OFFSET[chrom],
        y0=0,
        y1=1,
        line=dict(color="rgba(128,128,128,0.55)", width=0.8, dash="dash"),
    )
    for chrom in CHROMS
]

BASE_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=70, r=20, t=54, b=48),
    font=dict(size=12),
    autosize=True,
    hovermode="closest",
)

_DARK24 = px.colors.qualitative.Dark24

# Chromosome colours as a discrete colorscale rather than a per-point hex string
# array. Plotly's validator spends ~10 s on a 258k-element string array and ~0.1 s
# on the equivalent float array, so panels 1 and 2 send integer chromosome indices
# plus this scale. Paired stops keep each band flat instead of interpolating, and
# cmin/cmax must be pinned on every trace so masked traces agree on the mapping.
# This is the one deliberate departure from src/panels.py:46, :92.
_N_CHROM = len(CHROMS)
_CHROM_SCALE = [
    stop
    for i in range(_N_CHROM)
    for colour in (_DARK24[i % len(_DARK24)],)
    for stop in ([i / _N_CHROM, colour], [(i + 1) / _N_CHROM, colour])
]


def _chrom_marker(chrom_idx, mask, size: int, opacity: float) -> dict:
    """Marker spec colouring points by chromosome through _CHROM_SCALE."""
    return dict(
        size=size,
        color=chrom_idx[mask],
        colorscale=_CHROM_SCALE,
        cmin=0,
        cmax=_N_CHROM,
        showscale=False,
        opacity=opacity,
    )


def _chrom_index(x: np.ndarray) -> np.ndarray:
    """Chromosome ordinal for each genomic coordinate (src/panels.py:44-45)."""
    boundaries = [CHROM_OFFSET[c] for c in CHROMS] + [GENOME_LEN]
    idx = np.searchsorted(boundaries, x, side="right") - 1
    return np.clip(idx, 0, len(CHROMS) - 1).astype(np.float32)


def genome_layout(title: str, y_title: str, window: tuple[float, float] = FULL_GENOME) -> dict:
    """Return a layout dict for whole-genome x-axis panels (src/panels.py:19-35).

    Caller may append to layout["shapes"] before applying to fig.update_layout(**layout).
    """
    return dict(
        **BASE_LAYOUT,
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=14)),
        xaxis=dict(
            range=list(window),
            tickmode="array",
            tickvals=CHROM_TICKVALS,
            ticktext=CHROM_TICKTEXT,
            title="Genomic position",
            showgrid=False,
            zeroline=False,
            tickangle=0,
        ),
        yaxis=dict(title=y_title, zeroline=False),
        shapes=list(CHROM_BOUNDS),
        showlegend=False,
    )


def rgba(hex_color: str, alpha: float) -> str:
    """Convert #rrggbb to an rgba() CSS color."""
    color = hex_color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _window_rect(window: tuple[float, float]) -> dict:
    """Return a paper-coordinate rectangle shape highlighting the visible window."""
    return dict(
        type="rect",
        xref="x",
        yref="paper",
        x0=window[0],
        x1=window[1],
        y0=0,
        y1=1,
        fillcolor="rgba(76,120,168,0.13)",
        line=dict(width=0),
        layer="below",
    )


# ---------------------------------------------------------------------------
# Panel 1 — Copy-number state across genome (src/panels.py:37-80)
# ---------------------------------------------------------------------------
def panel_1(row: dict[str, object], window: tuple[float, float], biosample: str) -> go.Figure:
    """Copy-number state at each cCRE across genomic coordinates."""
    x = np.asarray(row["p1_x"], dtype=float)
    y = np.asarray(row["p1_y"], dtype=float)
    p1_n = np.asarray(row["p1_n"], dtype=float)

    chrom_idx = _chrom_index(x)

    win_mask = (x >= window[0]) & (x <= window[1])
    out_mask = ~win_mask

    fig = go.Figure()

    # Out-of-window trace: dimmed, no hover. Added unconditionally so trace
    # indices (curveNumber in selection payloads) do not shift with the window.
    fig.add_trace(
        go.Scattergl(
            x=x[out_mask],
            y=y[out_mask],
            mode="markers",
            marker=_chrom_marker(chrom_idx, out_mask, size=3, opacity=0.15),
            hoverinfo="skip",
        )
    )

    # In-window trace: full opacity with hover
    fig.add_trace(
        go.Scattergl(
            x=x[win_mask],
            y=y[win_mask],
            mode="markers",
            marker=_chrom_marker(chrom_idx, win_mask, size=3, opacity=0.85),
            customdata=p1_n[win_mask],
            hovertemplate="position %{x}<br>copy number %{y}<br>cCREs %{customdata}<extra></extra>",
        )
    )

    # Panel 1 is the fixed whole-genome overview: the window is drawn as a
    # highlight rect, never as an axis range (src/panels.py:64-78). Keeping the
    # axis full-width is also what lets a full-span brush restore the window.
    layout = genome_layout(
        f"1 · Copy number at each cCRE across genomic coordinates — {biosample}",
        "Copy number",
        FULL_GENOME,
    )
    # Append window highlight rect when zoomed in
    if window != FULL_GENOME:
        layout["shapes"].append(_window_rect(window))

    layout["dragmode"] = "select"
    layout["selectdirection"] = "h"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Panel 2 — ATAC z-score track (src/panels.py:82-111)
# ---------------------------------------------------------------------------
def panel_2(row: dict[str, object], window: tuple[float, float], biosample: str) -> go.Figure:
    """ATAC signal track across genome position."""
    x = np.asarray(row["p2_x"], dtype=float)
    y = np.asarray(row["p2_y"], dtype=float)
    p2_n = np.asarray(row["p2_n"], dtype=float)

    # Same chromosome cycle as panel 1
    chrom_idx = _chrom_index(x)

    win_mask = (x >= window[0]) & (x <= window[1])

    fig = go.Figure(
        go.Scattergl(
            x=x[win_mask],
            y=y[win_mask],
            mode="markers",
            marker=_chrom_marker(chrom_idx, win_mask, size=3, opacity=0.8),
            customdata=p2_n[win_mask],
            hovertemplate="position %{x}<br>z-score %{y:.2f}<br>cCREs %{customdata}<extra></extra>",
        )
    )
    layout = genome_layout(
        f"2 · ATAC signal at each cCRE across genomic coordinates — {biosample}",
        "ATAC z-score (500 kb median)",
        window,
    )
    layout["dragmode"] = "select"
    layout["selectdirection"] = "h"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Panel 3 — cCRE count per copy-number state (src/panels.py:113-141)
# ---------------------------------------------------------------------------
def panel_3(row: dict[str, object], cn_order: list[str]) -> go.Figure:
    """cCRE count per copy-number state (bar chart)."""
    # Read counts and zip positionally against the passed cn_order (E6 fix)
    cn_counts = list(row["cn_counts"])
    palette = cn_colors(cn_order)

    # x-values are the labelled form; customdata carries raw cn_order back
    labels3 = [cn_label(c) for c in cn_order]
    y = cn_counts[: len(cn_order)]  # positional, no Counter indirection

    fig = go.Figure(
        go.Bar(
            x=labels3,
            y=y,
            marker_color=[palette[c] for c in cn_order],
            customdata=list(cn_order),
            text=[f"{v:,}" for v in y],
            textposition="outside",
            textfont=dict(size=10),
            cliponaxis=False,
            hovertemplate="copy number %{x}<br>%{y:,} cCREs<extra></extra>",
        )
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(
            text="3 · Number of cCREs per copy number",
            x=0.0,
            xanchor="left",
            font=dict(size=14),
        ),
        showlegend=False,
        dragmode="select",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=labels3,
            title="Copy number",
        ),
        yaxis=dict(title="Number of cCREs"),
    )
    return fig


# ---------------------------------------------------------------------------
# bar_levels — hybrid pointIndex / x-label matching (§2.3)
# ---------------------------------------------------------------------------
def bar_levels(points: list[dict], cn_order: list[str]) -> list[str]:
    """Extract CN levels from panel-3 click/box-select payloads.

    Supports both the frontend click payload (pointIndex/pointNumber) and
    marimo 0.21 box-select payload (x key with label string). This is the
    one place that is intentionally a superset of the reference.
    """
    if not points:
        return []
    labels = [cn_label(c) for c in cn_order]
    picked: list[str] = []
    for point in points:
        idx = point.get("pointIndex", point.get("pointNumber"))  # frontend click
        if not isinstance(idx, int) or not 0 <= idx < len(cn_order):
            x = point.get("x")  # marimo 0.21 box-select payload
            idx = labels.index(x) if x in labels else None
        if idx is not None and cn_order[idx] not in picked:
            picked.append(cn_order[idx])
    return picked


# ---------------------------------------------------------------------------
# Panel 4 — Signal distributions (src/panels.py:148-214)
# ---------------------------------------------------------------------------
def panel_4(
    row: dict[str, object],
    active_cn: list[str],
    cn_order: list[str],
    biosample: str,
) -> go.Figure:
    """Cancer and healthy z-score distributions split by copy-number state.

    Draws filled polygon traces from the pre-computed violin outlines
    (violin_*_x / violin_*_y), matching src/panels.py:148-214 verbatim.
    """
    vc_x = list(row["violin_cancer_x"])
    vc_y = list(row["violin_cancer_y"])
    vh_x = list(row["violin_healthy_x"])
    vh_y = list(row["violin_healthy_y"])
    qc = list(row["violin_q_cancer"])
    qh = list(row["violin_q_healthy"])
    vn = list(row["violin_n"])

    fig = go.Figure()
    seen4: set[str] = set()

    for idx, level in enumerate(cn_order):
        if level not in active_cn:
            continue
        label = cn_label(level)
        i = float(idx)
        n_val = vn[idx] if idx < len(vn) else 0

        # One pass per tissue emitting polygon -> IQR bar -> median marker, so
        # trace order matches src/panels.py:155-200 exactly.
        for vx, vy, quart, tissue, colour, side in (
            (vc_x[idx], vc_y[idx], qc[idx] if idx < len(qc) else None,
             "Cancer tissue", CANCER_BLUE, -1.0),
            (vh_x[idx], vh_y[idx], qh[idx] if idx < len(qh) else None,
             "Healthy tissue", HEALTHY_ORANGE, 1.0),
        ):
            # Violin polygon (src/panels.py:161-176)
            fig.add_trace(
                go.Scatter(
                    x=vx,
                    y=vy,
                    mode="lines",
                    fill="toself",
                    fillcolor=rgba(colour, 0.82),
                    line=dict(color="rgba(0,0,0,0.45)", width=0.5),
                    name=tissue,
                    legendgroup=tissue,
                    showlegend=tissue not in seen4,
                    hoveron="fills",
                    text=f"{tissue} · copy number {label} · n = {n_val:,}",
                    hoverinfo="text",
                )
            )
            seen4.add(tissue)

            if quart is None or len(quart) < 3:
                continue
            q25, median, q75 = float(quart[0]), float(quart[1]), float(quart[2])
            x_off = i + side * 0.07

            # IQR bar — no hover; the median marker carries it
            # (src/panels.py:181-187)
            fig.add_trace(
                go.Scatter(
                    x=[x_off, x_off],
                    y=[q25, q75],
                    mode="lines",
                    line=dict(color="#222222", width=3),
                    legendgroup=tissue,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Median marker carries the quartile hover (src/panels.py:188-199)
            fig.add_trace(
                go.Scatter(
                    x=[x_off],
                    y=[median],
                    mode="markers",
                    marker=dict(size=7, color="white", line=dict(color="#222222", width=1.2)),
                    legendgroup=tissue,
                    showlegend=False,
                    hovertemplate=(
                        f"{tissue} · copy number {label}<br>"
                        f"q25 {q25:.2f} · median {median:.2f} · q75 {q75:.2f}<br>"
                        f"n = {n_val:,}<extra></extra>"
                    ),
                )
            )

    # Red dashed significance line at y=1.64 (D18)
    fig.add_hline(y=1.64, line_dash="dash", line_color="red", line_width=1.5)

    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(
            text=f"4 · Distribution of z-scores by copy number — {biosample}",
            x=0.0,
            xanchor="left",
            font=dict(size=14),
        ),
        dragmode="pan",
        legend=dict(title="zscore_type", orientation="h", y=1.02, yanchor="bottom"),
        xaxis=dict(
            range=[-0.6, len(cn_order) - 0.4],
            tickvals=list(range(len(cn_order))),
            ticktext=[cn_label(c) for c in cn_order],
            title="Copy number",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(title="Score", range=list(Z_LIM)),
    )
    return fig


# ---------------------------------------------------------------------------
# Panel 5 helper — aggregated density (src/panels.py:216-260)
# ---------------------------------------------------------------------------
def density_table(row: dict[str, object], active_cn: list[str], cn_order: list[str] | None = None) -> pa.Table:
    """Panel 5 table: aggregated cancer-vs-healthy density for selected CN states.

    Transcribed from src/panels.py:216-260. Groups on the raw float pair (x, y)
    rather than on rounded/binned values. Both axes are drawn from the same
    GRID_CENTERS, so grouping on the raw float pair is bit-exact.
    """
    x = np.asarray(list(row["scatter_cn_x"]), dtype=float)
    y = np.asarray(list(row["scatter_cn_y"]), dtype=float)
    lvl = np.asarray([cn_label(lv) for lv in row["scatter_cn_level"]])
    v = np.asarray(list(row["scatter_cn_v"]), dtype=np.int64)

    keep = np.isin(lvl, [cn_label(c) for c in active_cn])

    if not keep.any():
        return pa.table({
            "cancer_z": pa.array([], type=pa.float64()),
            "healthy_z": pa.array([], type=pa.float64()),
            "cCREs": pa.array([], type=pa.int64()),
            "log_cCREs": pa.array([], type=pa.float64()),
        })

    x, y, v = x[keep], y[keep], v[keep]

    # Group on the (x, y) float pair — bit-exact because both axes come from GRID_CENTERS
    pairs = np.stack([x, y], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    agg_v = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(agg_v, inv, v)

    log_cCREs = np.log1p(agg_v.astype(float))
    return pa.table({
        "cancer_z": uniq[:, 0],
        "healthy_z": uniq[:, 1],
        "cCREs": agg_v,
        "log_cCREs": log_cCREs,
    })


# ---------------------------------------------------------------------------
# Panel 6 helper — per-copy-number scatter (D23, E5)
# ---------------------------------------------------------------------------
def cn_table(row: dict[str, object], active_cn: list[str], cn_order: list[str]) -> pa.Table:
    """Panel 6 table: per-copy-number scatter points for selected CN states."""
    x = np.asarray(list(row["scatter_cn_x"]), dtype=float)
    y = np.asarray(list(row["scatter_cn_y"]), dtype=float)
    lvl_raw = list(row["scatter_cn_level"])
    v = np.asarray(list(row["scatter_cn_v"]), dtype=np.int64)
    lvl = np.asarray([cn_label(lv) for lv in lvl_raw])

    keep = np.isin(lvl, [cn_label(c) for c in active_cn])
    return pa.table({
        "cancer_z": x[keep],
        "healthy_z": y[keep],
        "copy_number": lvl[keep],
        "cCREs": v[keep],
    })


__all__ = [
    "CANCER_BLUE",
    "CHROMS",
    "CHROM_MID",
    "CHROM_OFFSET",
    "CHROM_SIZES",
    "DATA_DIR",
    "DISTRIBUTIONS",
    "Dataset",
    "FALLBACK",
    "FULL_GENOME",
    "GENOME_LEN",
    "GRID_EDGES",
    "HEALTHY_ORANGE",
    "HIST2D_SHAPE",
    "MISSING_GREY",
    "N_GRID",
    "N_VIOLIN_BINS",
    "REMOTE",
    "SCHEMA_VERSION",
    "TRACKS",
    "VIOLIN_EDGES",
    "Z_LIM",
    "assert_edges_match",
    "bar_levels",
    "cn_colors",
    "cn_label",
    "cn_table",
    "cols_for",
    "density_table",
    "file_constants",
    "genome_layout",
    "load_cohort",
    "load_panel",
    "open_dataset",
    "panel_1",
    "panel_2",
    "panel_3",
    "panel_4",
    "panel_tags",
    "read_metadata",
    "rgba",
    "source",
]
