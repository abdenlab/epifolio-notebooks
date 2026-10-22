"""CNV/cCRE dashboard: payload contract, data access, genome coordinates, colors, panels.

Companion module to ``notebooks/cnv_ccre_explorer.py``. Sections in dependency
order: schema constants, hg38 coordinates, copy-number colors, payload reads,
then the pure plotly/Arrow panel builders.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from importlib.metadata import files
from itertools import accumulate
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
SCHEMA_VERSION = "5"
N_VIOLIN_BINS = 464
N_GRID = 348
Z_LIM = (-18.0, 11.0)
# Missing-ATAC marker kept in every v4 histogram (file metadata sentinel_value).
SENTINEL = -10.0

VIOLIN_EDGES = np.linspace(Z_LIM[0], Z_LIM[1], N_VIOLIN_BINS + 1).tolist()
GRID_EDGES = np.linspace(Z_LIM[0], Z_LIM[1], N_GRID + 1).tolist()
# Panel 5/6 cell centres exactly as the writer publishes them: float32 bin
# midpoints, stored as float64.
GRID_CENTERS = (
    ((np.asarray(GRID_EDGES[:-1]) + np.asarray(GRID_EDGES[1:])) / 2).astype(np.float32).astype(np.float64)
)
VIOLIN_CENTERS = (np.asarray(VIOLIN_EDGES[:-1]) + np.asarray(VIOLIN_EDGES[1:])) / 2
# Half-width of the widest bin in each published violin outline.
VIOLIN_HALF_WIDTH = 0.44
HIST2D_SHAPE = [N_GRID, N_GRID]

TRACKS = "summary_genome_tracks.parquet"
DISTRIBUTIONS = "summary_signal_distributions.parquet"
CCRE_REFERENCE = "ccre_reference.parquet"
MEMBERSHIP = "summary_ccre_membership.parquet"
MEMBERSHIP_SCHEMA_VERSION = "2"
CCRE_REFERENCE_SCHEMA_VERSION = "1"
# cn_order level for cCREs outside every ASCAT segment; panels 3-4 leave it out.
UNMAPPED = "Unmapped"
# Every ENCODE cCRE accession in the locus table: this prefix and 7 digits.
CCRE_PREFIX = "EH38E"
DETAIL = "atac_zscore_detail.parquet"
DETAIL_MANIFEST = "detail_manifest.json"
DETAIL_SCHEMA_VERSION = "2"
# Panel 2 switches from the binned overview to raw per-cCRE z-scores once the
# window covers at most this many cCREs (chr1, the largest, is ~204k).
DETAIL_MAX_ROWS = 250_000
# Adjacent overview x bins merged into one point when panel 2 shows the whole
# genome (~4.7k bins on a plot ~1.5k px wide).
OVERVIEW_MERGE = 3
# Columns shown in every per-tab cCRE table. rDHS is the join key between
# the locus table and the membership artifact and is never displayed.
TABLE_COLUMNS = [
    "cCRE",
    "chrom",
    "start",
    "end",
    "class",
    "copy_number",
    "copy_number_code",
    "x_bin",
    "cancer_z",
    "healthy_z",
]


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
            f"CNV data is schema {constants['schema_version']}, this notebook needs "
            f"schema {SCHEMA_VERSION}. Set EPIFOLIO_CNV_REMOTE=dev until prod is updated."
        )
    checks = {
        "violin_edges": VIOLIN_EDGES,
        "grid_edges": GRID_EDGES,
        "hist2d_shape": HIST2D_SHAPE,
    }
    for key, expected in checks.items():
        got = constants[key]
        if np.shape(got) != np.shape(expected) or not np.allclose(got, expected):
            raise ValueError(f"CNV payload {key} does not match the notebook contract")
    sentinel = metadata.get("sentinel_value")
    if isinstance(sentinel, bytes):
        sentinel = sentinel.decode()
    if sentinel is None or float(sentinel) != SENTINEL:
        raise ValueError(f"CNV payload sentinel_value={sentinel!r}; expected {SENTINEL!r}")


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
    """Copy-number label; schema 5 publishes cn_order strings exactly everywhere."""
    return str(level)


def mapped_levels(cn_order: Iterable[object]) -> list[str]:
    """cn_order without UNMAPPED: the levels panels 1, 3 and 4 draw."""
    return [cn_label(level) for level in cn_order if cn_label(level) != UNMAPPED]


def grid_bin_id(x: Iterable[float], y: Iterable[float]) -> np.ndarray:
    """Return stable row/bin IDs for the published N_GRID x N_GRID grid centers."""
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    x_idx = np.searchsorted(GRID_EDGES, x_array, side="right") - 1
    y_idx = np.searchsorted(GRID_EDGES, y_array, side="right") - 1
    if np.any((x_idx < 0) | (x_idx >= N_GRID) | (y_idx < 0) | (y_idx >= N_GRID)):
        raise ValueError("z-score coordinates fall outside the published grid")
    return (y_idx * N_GRID + x_idx).astype(np.int64)


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
# Prod mirror (default) and dev mirror (new data before prod gets it).
# EPIFOLIO_CNV_REMOTE: "prod", "dev" or a mirror URL.
PROD_REMOTE = "https://projects.abdenlab.org/itcr/epifolio/notebooks/cnv_ccre"
DEV_REMOTE = "https://users.abdenlab.org/voy/itcr/epifolio/notebooks/cnv_ccre"
_REMOTE_CHOICE = os.environ.get("EPIFOLIO_CNV_REMOTE", "prod")
REMOTE = {"prod": PROD_REMOTE, "dev": DEV_REMOTE}.get(_REMOTE_CHOICE, _REMOTE_CHOICE)

# cCRE locus table (chrom/start/end/rDHS/cCRE/class), shared across every
# *.ccre.pq embedding file and row-count-identical to the CNV payload's
# n_ccre (cnvpostpr.md §4). Same local-else-canonical discovery as source().
CCRE_INDEX_NAME = "tcga.atac.pca.ccre.pq"
CCRE_INDEX_DIR = ASSETS / "data" / "ccre_index"
CCRE_INDEX = "https://projects.abdenlab.org/itcr/epifolio/embeddings"


def source(name: str) -> str:
    """Use the payload bundled with the installed package, else the canonical mirror."""
    local = DATA_DIR / name
    if local.exists():
        return str(local)
    return f"{REMOTE}/{name}"


def membership_source() -> str:
    """Resolve the per-biosample copy-number membership sidecar.

    The sidecar is intentionally separate from the cCRE reference table and
    published raster payloads.
    """
    return source(MEMBERSHIP)


def ccre_reference_source() -> str:
    """Resolve the stable cCRE reference table used by membership arrays."""
    return source(CCRE_REFERENCE)


def ccre_index_source() -> str:
    """Use the packaged cCRE locus table if present, else the canonical embeddings host."""
    local = CCRE_INDEX_DIR / CCRE_INDEX_NAME
    if local.exists():
        return str(local)
    return f"{CCRE_INDEX}/{CCRE_INDEX_NAME}"


_CCRE_META_COLUMNS = ["chrom", "start", "end", "rDHS", "cCRE", "class"]


def load_loci(columns: Iterable[str] | None = None) -> pl.DataFrame:
    """Load the per-cCRE locus table, resorted into genome-linear order.

    Reads ``tcga.atac.pca.ccre.pq`` as delivered, selecting only the six
    metadata columns shared by every ``*.ccre.pq`` embedding file
    (chrom/start/end/rDHS/cCRE/class). The parquet's own chromosome order is
    lexicographic (chr1, chr10, chr11, ...), not the karyotypic CHROMS order
    the rest of this module uses — joining by row position would silently
    mis-assign ~2.3M cCREs (cnvpostpr.md §4.3). The re-sort and the added
    genome-linear ``pos`` column (comparable to panel 1/2 x-coordinates) are
    computed here, once, at load time — never by re-sorting the source file.
    """
    df = pl.scan_parquet(ccre_index_source()).select(_CCRE_META_COLUMNS).collect()
    # Preserve the delivered source identity before sorting into genome order.
    # The split membership artifact is joined by its cCRE reference identifier,
    # not by this post-sort row position.
    df = df.with_row_index("cCRE_index")

    chrom_rank = {chrom: i for i, chrom in enumerate(CHROMS)}
    df = (
        df.with_columns(pl.col("chrom").replace_strict(chrom_rank, return_dtype=pl.Int64).alias("_chrom_rank"))
        .sort(["_chrom_rank", "start"])
        .with_columns(
            (
                pl.col("chrom").replace_strict(CHROM_OFFSET, return_dtype=pl.Int64) + pl.col("start")
            ).alias("pos")
        )
        .drop("_chrom_rank")
    )

    if columns is None:
        return df
    wanted = list(columns)
    available = set(_CCRE_META_COLUMNS) | {"cCRE_index", "pos"}
    missing = set(wanted) - available
    if missing:
        raise ValueError(f"load_loci columns must be a subset of {sorted(available)}; got extra {sorted(missing)}")
    return df.select(wanted)


def load_ccre_reference(src: str | None = None) -> pl.DataFrame:
    """Load the stable row-indexed cCRE reference for membership arrays."""
    src = src or ccre_reference_source()
    try:
        metadata = read_metadata(src)
        if metadata.get("schema_version") != CCRE_REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"cCRE reference schema_version={metadata.get('schema_version')!r}; "
                f"expected {CCRE_REFERENCE_SCHEMA_VERSION!r}"
            )
        if str(metadata.get("n_genome_bins")) != "5000":
            raise ValueError("cCRE reference n_genome_bins must be 5000")
        if metadata.get("row_order") != (
            "canonical -- row position is the join key "
            "summary_ccre_membership.parquet's copy_number_code arrays are aligned to"
        ):
            raise ValueError("cCRE reference row-order metadata does not match membership contract")
        reference = (
            pl.scan_parquet(src)
            .select(["cCRE_name", "x_bin"])
            .collect()
            .with_row_index("reference_index")
        )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"invalid cCRE reference artifact at {src!r}") from exc

    if reference["cCRE_name"].n_unique() != reference.height:
        raise ValueError("cCRE reference cCRE_name values must be unique")
    x_bin = np.asarray(reference["x_bin"].to_numpy(), dtype=np.int64)
    if np.any((x_bin < 0) | (x_bin >= 5000)):
        raise ValueError("cCRE reference x_bin contains a value outside 0..4999")
    return reference


def load_membership(
    biosample: str,
    src: str | None = None,
    *,
    reference: pl.DataFrame | None = None,
    expected_n_ccre: int | None = None,
    expected_cn_order: Iterable[object] | None = None,
) -> pl.DataFrame:
    """Load one biosample's CN codes and attach them to the cCRE reference.

    ``copy_number_code[i]`` is aligned to row ``i`` of
    ``ccre_reference.parquet``. The returned frame contains the reference
    row, ``cCRE_name``, ``x_bin``, ``copy_number_code`` and normalized
    ``copy_number``. It must be joined to the locus table by
    ``cCRE_name == rDHS`` rather than by row position.
    """
    src = src or membership_source()
    reference = reference if reference is not None else load_ccre_reference()
    try:
        metadata = read_metadata(src)
    except Exception as exc:
        raise FileNotFoundError(
            f"CNV cCRE membership artifact is unavailable at {src!r}; "
            "provide the split cCRE membership artifacts before enabling identity resolution"
        ) from exc

    schema_version = metadata.get("schema_version")
    if schema_version != MEMBERSHIP_SCHEMA_VERSION:
        raise ValueError(
            f"membership schema_version={schema_version!r}; "
            f"expected {MEMBERSHIP_SCHEMA_VERSION!r}"
        )
    if metadata.get("ccre_reference_file") != CCRE_REFERENCE:
        raise ValueError("membership cCRE reference file does not match the notebook contract")
    if metadata.get("join_key") != "copy_number_code[i] maps to ccre_reference.parquet row i":
        raise ValueError("membership join-key metadata does not match the notebook contract")

    expected_order = [cn_label(level) for level in (expected_cn_order or [])]
    declared_order = metadata.get("cn_order")
    if not declared_order:
        raise ValueError("membership is missing cn_order metadata")
    try:
        raw_declared = (
            declared_order
            if isinstance(declared_order, (list, tuple))
            else json.loads(declared_order)
        )
        declared = [cn_label(level) for level in raw_declared]
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("membership cn_order is not valid JSON") from exc
    if expected_order and declared != expected_order:
        raise ValueError("membership cn_order does not match the CNV payload")

    frame = (
        pl.scan_parquet(src)
        .filter(pl.col("biosample") == biosample)
        .select(["biosample", "copy_number_code"])
        .collect()
    )
    if frame.height != 1:
        raise KeyError(f"membership artifact has {frame.height} rows for biosample {biosample!r}; expected 1")

    codes = np.asarray(frame["copy_number_code"][0].to_numpy(), dtype=np.int64)
    n_ccre = reference.height
    if expected_n_ccre is not None and n_ccre != expected_n_ccre:
        raise ValueError(f"cCRE reference n_ccre={n_ccre}; expected {expected_n_ccre}")
    if len(codes) != n_ccre:
        raise ValueError(
            f"membership copy_number_code length={len(codes)}; expected {n_ccre}"
        )

    if np.any((codes < 0) | (codes >= len(declared))):
        raise ValueError("membership copy_number_code contains an unknown code")

    return reference.with_columns(
        pl.Series("copy_number_code", codes, dtype=pl.UInt8),
        pl.Series("copy_number", declared, dtype=pl.String).gather(codes),
    )


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
# Detail tier: raw per-cCRE cancer and healthy z-scores
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DetailIndex:
    """The detail manifest reduced to the biosample -> column maps.

    The detail parquet is wide: one float32 cancer column per biosample
    (``sNNNN``) and one healthy column per ENCODE reference (``h_ENCSR…``),
    with rows in genome order and ``reference_index`` pointing at the
    ccre_reference.parquet row each value belongs to. ``healthy`` maps a
    biosample to None when it has no healthy reference.
    """

    src: str
    columns: dict[str, str]
    healthy: dict[str, str | None]


def _read_text(src: str) -> str:
    local = Path(src)
    if local.exists():
        return local.read_text()
    with urllib.request.urlopen(src) as response:
        return response.read().decode()


def load_detail_index(
    overview_metadata: Mapping[str, str | bytes],
    src: str | None = None,
    manifest_src: str | None = None,
) -> DetailIndex:
    """Load the detail manifest and check it was built from this overview payload."""
    src = src or source(DETAIL)
    manifest = json.loads(_read_text(manifest_src or source(DETAIL_MANIFEST)))
    contract = manifest["contract"]
    if contract["detail_schema_version"] != DETAIL_SCHEMA_VERSION:
        raise ValueError(
            f"detail_schema_version={contract['detail_schema_version']!r}; "
            f"expected {DETAIL_SCHEMA_VERSION!r}"
        )
    if contract["chromosomes"] != CHROMS:
        raise ValueError("detail chromosome order does not match the notebook contract")
    overview_build = overview_metadata.get("build_id")
    if isinstance(overview_build, bytes):
        overview_build = overview_build.decode()
    if contract["overview_build_id"] != overview_build:
        raise ValueError("detail tier was built against a different overview payload")
    if read_metadata(src).get("detail_build_id") != manifest["build_id"]:
        raise ValueError("detail parquet does not match its manifest")

    samples = contract["samples"]
    return DetailIndex(
        src=src,
        columns={sample["biosample"]: sample["column"] for sample in samples},
        healthy={sample["biosample"]: sample["healthy_column"] for sample in samples},
    )


def _grid_index(z: np.ndarray) -> np.ndarray:
    """Grid bin of each raw z-score: left-closed bins, the last one right-closed."""
    idx = np.searchsorted(GRID_EDGES, z, side="right") - 1
    return np.where(z == GRID_EDGES[-1], N_GRID - 1, idx)


def load_coords(index: DetailIndex) -> pl.DataFrame:
    """The detail rows' ``reference_index`` and genome-linear ``pos``.

    The same for every biosample, so read once per session. ``pos`` anchors
    each point at the cCRE start, as in load_loci.
    """
    return (
        pl.scan_parquet(index.src)
        .select(
            "reference_index",
            (
                pl.col("chrom").replace_strict(CHROM_OFFSET, return_dtype=pl.Int64)
                + pl.col("start")
            ).alias("pos"),
        )
        .collect()
    )


def load_detail_column(index: DetailIndex, column: str) -> pl.Series:
    """One float32 detail column, cancer (``sNNNN``) or healthy (``h_ENCSR…``)."""
    return pl.scan_parquet(index.src).select(column).collect().to_series()


def zscore_frame(
    coords: pl.DataFrame, cancer: pl.Series, healthy: pl.Series | None
) -> pl.DataFrame:
    """Whole-genome cancer and healthy z-scores for one biosample, in genome order.

    Columns from one detail file share its row order, so ``coords`` and the
    value columns line up by position. Returns ``reference_index``, ``pos``,
    ``cancer_z``, ``healthy_z`` and ``bin_id``, the cCRE's panel 5/6 grid cell
    in the grid_bin_id convention. The healthy columns and ``bin_id`` are null
    without a healthy reference. Windows are filtered from this frame, never
    re-read.
    """
    frame = coords.with_columns(
        cancer.alias("cancer_z"),
        (pl.lit(None, dtype=pl.Float32) if healthy is None else healthy).alias("healthy_z"),
    )
    if healthy is None:
        return frame.with_columns(bin_id=pl.lit(None, dtype=pl.Int64))
    # Same binning the writer used for scatter_cn_*; float64 edges on the
    # float32 values reproduce the published panel 6 counts exactly.
    cancer_idx = _grid_index(frame["cancer_z"].to_numpy().astype(np.float64))
    healthy_idx = _grid_index(frame["healthy_z"].to_numpy().astype(np.float64))
    return frame.with_columns(
        bin_id=pl.when(pl.col("healthy_z").is_not_null()).then(
            pl.Series(healthy_idx * N_GRID + cancer_idx, dtype=pl.Int64)
        )
    )


def load_zscores(index: DetailIndex, biosample: str) -> pl.DataFrame:
    """zscore_frame for one biosample in a single call (coordinates re-read)."""
    healthy = index.healthy[biosample]
    return zscore_frame(
        load_coords(index),
        load_detail_column(index, index.columns[biosample]),
        None if healthy is None else load_detail_column(index, healthy),
    )


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
_CHROM_STARTS = np.asarray([CHROM_OFFSET[chrom] for chrom in CHROMS], dtype=float)

# Chromosome colours as a discrete colorscale rather than a per-point hex string
# array. Plotly's validator spends ~10 s on a 258k-element string array and ~0.1 s
# on the equivalent float array, so panel 1 sends integer chromosome indices
# plus this scale (panel 2 draws one single-colour trace per chromosome).
# Paired stops keep each band flat instead of interpolating, and cmin/cmax must
# be pinned on every trace so masked traces agree on the mapping.
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


def _chrom_traces(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    customdata: np.ndarray,
    hover: str,
    *,
    size: int,
    opacity: float,
    x_dtype: type = np.float64,
) -> list[go.Scattergl]:
    """One Scattergl per chromosome under ``mask``, coloured as in panel 1.

    A trace per chromosome lets the hover name its chromosome without a
    string per point: ``{chrom}`` in ``hover`` becomes the chromosome name.
    ``y`` is sent as float32 and ``x`` as ``x_dtype``: float32 is exact
    enough for binned centres, not for per-cCRE positions.
    """
    chrom_idx = _chrom_index(x).astype(int)
    traces = []
    for i in np.unique(chrom_idx[mask]):
        keep = mask & (chrom_idx == i)
        traces.append(
            go.Scattergl(
                x=x[keep].astype(x_dtype),
                y=y[keep].astype(np.float32),
                mode="markers",
                marker=dict(size=size, color=_DARK24[i % len(_DARK24)], opacity=opacity),
                customdata=customdata[keep],
                hovertemplate=hover.replace("{chrom}", CHROMS[i]),
                name=CHROMS[i],
            )
        )
    return traces


def _coarsen_overview(
    x: np.ndarray, y: np.ndarray, n: np.ndarray, factor: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge every ``factor`` adjacent overview x bins within a chromosome.

    Counts add up per (merged bin, z bin), so the total is unchanged; a
    merged bin sits at the mean of its bins' centres.
    """
    ux, xi = np.unique(x, return_inverse=True)
    chrom = _chrom_index(ux).astype(np.int64)
    rank = np.arange(len(ux)) - np.searchsorted(chrom, chrom)
    group = np.unique(chrom * len(ux) + rank // factor, return_inverse=True)[1]
    centre = np.bincount(group, weights=ux) / np.bincount(group)
    uy, yi = np.unique(y, return_inverse=True)
    key, inv = np.unique(group[xi] * len(uy) + yi, return_inverse=True)
    return centre[key // len(uy)], uy[key % len(uy)], np.bincount(inv, weights=n)


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
            title="Chromosome",
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
def panel_1(
    row: dict[str, object],
    window: tuple[float, float],
    biosample: str,
    cn_order: list[str],
) -> go.Figure:
    """Copy-number state at each cCRE across genomic coordinates.

    ``p1_y`` is the position in cn_order (6 stands for "6+"); Unmapped cCREs
    have no y and are only counted in ``p1_unmapped_n``.
    """
    x = np.asarray(row["p1_x"], dtype=float)
    y = np.asarray(row["p1_y"], dtype=float)
    p1_n = np.asarray(row["p1_n"], dtype=float)
    levels = mapped_levels(cn_order)
    y_label = np.asarray(levels, dtype=object)[y.astype(int)]

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
            customdata=np.column_stack([y_label[win_mask], p1_n[win_mask]]),
            hovertemplate=(
                "position %{x}<br>copy number %{customdata[0]}<br>"
                "cCREs %{customdata[1]}<extra></extra>"
            ),
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

    layout["yaxis"].update(tickvals=list(range(len(levels))), ticktext=levels)
    layout["dragmode"] = "select"
    layout["selectdirection"] = "h"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Panel 2 — ATAC z-score track (src/panels.py:82-111)
# ---------------------------------------------------------------------------
def sentinel_mask(y: np.ndarray, binned: bool) -> np.ndarray:
    """True where a panel-2 value is the -10 sentinel.

    Binned overview values are grid-bin centres, so the whole bin holding
    SENTINEL is masked; that bin also holds the few genuine values in
    [-10, -10 + bin width), at most 6 per biosample in the v4 build.
    """
    if not binned:
        return y == SENTINEL
    bin_of = np.searchsorted(GRID_EDGES, y, side="right") - 1
    return bin_of == np.searchsorted(GRID_EDGES, SENTINEL, side="right") - 1


_PATH_POINT = re.compile(r"[ML]\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)")


def selection_region(selection: Mapping[str, object]) -> dict[str, object] | None:
    """Extent and lasso outline of a plotly selection, or None for no selection.

    A box arrives as ``range``; a lasso has no range, only its outline as the
    last ``selections`` entry (an SVG path in data coordinates). Returns
    ``x`` and ``y`` as (lo, hi) and ``lasso`` as (xs, ys), None for a box.
    """
    box = selection.get("range")
    if isinstance(box, Mapping) and box.get("x"):
        y = box.get("y")
        return {
            "x": (float(min(box["x"])), float(max(box["x"]))),
            "y": (float(min(y)), float(max(y))) if y else None,
            "lasso": None,
        }
    shapes = selection.get("selections") or []
    if not shapes:
        return None
    shape = shapes[-1]
    if shape.get("type") == "rect":
        xs = [float(shape["x0"]), float(shape["x1"])]
        ys = [float(shape["y0"]), float(shape["y1"])]
        lasso = None
    else:
        pairs = _PATH_POINT.findall(str(shape.get("path", "")))
        if len(pairs) < 3:
            return None
        xs = [float(x) for x, _ in pairs]
        ys = [float(y) for _, y in pairs]
        lasso = (xs, ys)
    return {"x": (min(xs), max(xs)), "y": (min(ys), max(ys)), "lasso": lasso}


def in_polygon(
    x: np.ndarray, y: np.ndarray, polygon_x: Sequence[float], polygon_y: Sequence[float]
) -> np.ndarray:
    """True where (x, y) falls inside the polygon (even-odd ray casting)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = np.zeros(len(x), dtype=bool)
    j = len(polygon_x) - 1
    for i in range(len(polygon_x)):
        xi, yi, xj, yj = polygon_x[i], polygon_y[i], polygon_x[j], polygon_y[j]
        if yi != yj:
            crosses = (yi > y) != (yj > y)
            inside ^= crosses & (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        j = i
    return inside


def drop_sentinel(frame: pl.DataFrame, columns: Iterable[str] = ("cancer_z",)) -> pl.DataFrame:
    """Rows where none of ``columns`` holds the -10 sentinel; nulls are kept.

    Columns the frame lacks (no detail tier) are skipped.
    """
    present = [name for name in columns if name in frame.columns]
    if not present:
        return frame
    return frame.filter(pl.all_horizontal(pl.col(name).ne_missing(SENTINEL) for name in present))


def panel_2(
    row: dict[str, object],
    window: tuple[float, float],
    biosample: str,
    *,
    hide_sentinel: bool = False,
    detail: pl.DataFrame | None = None,
    zrange: tuple[float, float] | None = None,
    lasso: tuple[Sequence[float], Sequence[float]] | None = None,
) -> go.Figure:
    """ATAC signal track across genome position.

    ``detail`` (per-cCRE rows inside the window, with ``pos`` and
    ``cancer_z``) replaces the binned overview with raw values when the
    window is narrow enough; with the locus columns ``cCRE``/``start``/``end``
    the hover names each cCRE and its interval. ``zrange`` is the z-score
    band of the last box selection, drawn as a highlight; ``lasso`` the
    outline of the last lasso, drawn in its place.
    """
    if detail is None:
        x = np.asarray(row["p2_x"], dtype=float)
        y = np.asarray(row["p2_y"], dtype=float)
        n = np.asarray(row["p2_n"], dtype=float)
        if window == FULL_GENOME:
            x, y, n = _coarsen_overview(x, y, n, OVERVIEW_MERGE)
        local_mb = (x - _CHROM_STARTS[_chrom_index(x).astype(int)]) / 1e6
        customdata = np.column_stack([n, local_mb]).astype(np.float32)
        hover = (
            "{chrom} ~%{customdata[1]:.1f} Mb<br>z-score %{y:.2f}<br>"
            "cCREs %{customdata[0]:,}<extra></extra>"
        )
        tier = ""
    else:
        x = detail["pos"].to_numpy().astype(float)
        y = detail["cancer_z"].to_numpy().astype(float)
        if {"cCRE", "start", "end"} <= set(detail.columns) and (
            detail["cCRE"].str.contains(f"^{CCRE_PREFIX}[0-9]{{7}}$").all()
        ):
            # Accession digits, not strings: a string per point triples the payload.
            customdata = np.column_stack(
                [
                    detail["cCRE"].str.slice(len(CCRE_PREFIX)).cast(pl.Int32).to_numpy(),
                    detail["start"].to_numpy(),
                    detail["end"].to_numpy(),
                ]
            ).astype(np.int32)
            hover = (
                f"cCRE {CCRE_PREFIX}%{{customdata[0]:07d}}<br>"
                "{chrom}:%{customdata[1]:,}–%{customdata[2]:,}<br>"
                "z-score %{y:.3f}<extra></extra>"
            )
        else:
            customdata = (x - _CHROM_STARTS[_chrom_index(x).astype(int)]).astype(np.int32)[:, None]
            hover = "{chrom}:%{customdata[0]:,}<br>z-score %{y:.3f}<extra></extra>"
        tier = f" · per-cCRE detail ({len(y):,})"

    win_mask = (x >= window[0]) & (x <= window[1])
    if hide_sentinel:
        win_mask &= ~sentinel_mask(y, binned=detail is None)

    fig = go.Figure(
        _chrom_traces(
            x,
            y,
            win_mask,
            customdata,
            hover,
            size=3,
            opacity=0.8,
            x_dtype=np.float32 if detail is None else np.float64,
        )
    )
    layout = genome_layout(
        f"2 · ATAC signal at each cCRE across genomic coordinates — {biosample}{tier}",
        "ATAC z-score (500 kb median)",
        window,
    )
    if lasso is not None:
        layout["shapes"].append(
            dict(
                type="path",
                xref="x",
                yref="y",
                path="M" + "L".join(f"{px},{py}" for px, py in zip(*lasso)) + "Z",
                fillcolor="rgba(76,120,168,0.13)",
                line=dict(width=1, color="rgba(76,120,168,0.6)"),
                layer="below",
            )
        )
    elif zrange is not None:
        layout["shapes"].append(
            dict(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=zrange[0],
                y1=zrange[1],
                fillcolor="rgba(76,120,168,0.13)",
                line=dict(width=0),
                layer="below",
            )
        )
    # A 2-D box: x narrows the genomic window, y picks a z-score band.
    layout["dragmode"] = "select"
    layout["selectdirection"] = "any"
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Panel 3 — cCRE count per copy-number state (src/panels.py:113-141)
# ---------------------------------------------------------------------------
def panel_3(
    row: dict[str, object],
    cn_order: list[str],
    sentinel_n: Iterable[int] | None = None,
) -> go.Figure:
    """cCRE count per copy-number state (bar chart), Unmapped left out.

    ``sentinel_n`` (the published ``violin_cancer_sentinel_n``) is subtracted
    per level to leave out cCREs whose cancer z-score is the -10 sentinel.
    """
    counts = list(row["cn_counts"])
    if sentinel_n is not None:
        counts = [count - n for count, n in zip(counts, sentinel_n, strict=True)]
    # Zip counts positionally against cn_order (E6 fix), then drop Unmapped
    # as a pair so the remaining bars keep their own counts.
    pairs = [
        (cn_label(level), count)
        for level, count in zip(cn_order, counts, strict=True)
        if cn_label(level) != UNMAPPED
    ]
    labels3 = [label for label, _ in pairs]
    y = [count for _, count in pairs]
    palette = cn_colors(labels3)

    fig = go.Figure(
        go.Bar(
            x=labels3,
            y=y,
            marker_color=[palette[label] for label in labels3],
            customdata=labels3,
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
    vn_healthy = list(row.get("violin_healthy_n", vn))

    fig = go.Figure()
    seen4: set[str] = set()
    shown = mapped_levels(cn_order)

    for idx, level in enumerate(cn_order):
        label = cn_label(level)
        if label == UNMAPPED or level not in active_cn:
            continue
        i = float(shown.index(label))
        # One pass per tissue emitting polygon -> IQR bar -> median marker, so
        # trace order matches src/panels.py:155-200 exactly.
        for vx, vy, quart, n_val, tissue, colour, side in (
            (vc_x[idx], vc_y[idx], qc[idx] if idx < len(qc) else None,
             vn[idx] if idx < len(vn) else 0, "Cancer tissue", CANCER_BLUE, -1.0),
            (vh_x[idx], vh_y[idx], qh[idx] if idx < len(qh) else None,
             vn_healthy[idx] if idx < len(vn_healthy) else 0, "Healthy tissue", HEALTHY_ORANGE, 1.0),
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

            # Samples without a healthy reference publish empty outlines and
            # null quartiles.
            if quart is None or len(quart) < 3 or any(q is None for q in quart[:3]):
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
            range=[-0.6, len(shown) - 0.4],
            tickvals=list(range(len(shown))),
            ticktext=shown,
            title="Copy number",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(title="Score", range=list(Z_LIM)),
    )
    return fig


def violin_outline(counts: np.ndarray, position: float, side: float) -> tuple[list[float], list[float]]:
    """The writer's half-violin outline for one histogram over VIOLIN_EDGES.

    Width grows with the bin count up to VIOLIN_HALF_WIDTH at the widest bin,
    on ``side`` (-1 cancer, +1 healthy) of ``position``, closed along it. An
    empty level gets a zero-width outline, as published.
    """
    counts = np.asarray(counts, dtype=float)
    if not len(counts):
        return [], []
    peak = counts.max()
    width = counts / peak if peak > 0 else np.zeros_like(counts)
    x = np.concatenate(
        [position + side * VIOLIN_HALF_WIDTH * width, np.full(len(counts), float(position))]
    )
    y = np.concatenate([VIOLIN_CENTERS, VIOLIN_CENTERS[::-1]])
    return x.tolist(), y.tolist()


def drop_sentinel_violins(
    row: dict[str, object], frame: pl.DataFrame, cn_order: list[str]
) -> dict[str, object]:
    """Panel 4 row with the -10 sentinels taken out of every violin.

    Each level's published sentinel count leaves the histogram bin holding
    SENTINEL, the outline is redrawn by violin_outline, and the quartiles are
    recomputed from ``frame`` (per-cCRE ``copy_number``, ``cancer_z``,
    ``healthy_z``) under the published quantile policy, numpy's linear
    percentile. A tissue with no published values (no healthy reference)
    keeps its published, empty violins.
    """
    sentinel_bin = int(np.searchsorted(VIOLIN_EDGES, SENTINEL, side="right") - 1)
    by_level = frame.partition_by("copy_number", as_dict=True)
    out = dict(row)
    for tissue, side, n_key in (("cancer", -1.0, "violin_n"), ("healthy", 1.0, "violin_healthy_n")):
        if not any(row[n_key]):
            continue
        column = f"{tissue}_z"
        xs, ys, quartiles, ns = [], [], [], []
        for idx, level in enumerate(cn_order):
            counts = np.array(row[f"violin_{tissue}_counts"][idx], dtype=float)
            counts[sentinel_bin] -= row[f"violin_{tissue}_sentinel_n"][idx]
            x, y = violin_outline(counts, idx, side)
            level_frame = by_level.get((cn_label(level),))
            values = (
                np.array([])
                if level_frame is None
                else drop_sentinel(level_frame, (column,))[column].drop_nulls().to_numpy()
            )
            xs.append(x)
            ys.append(y)
            quartiles.append(np.percentile(values, [25, 50, 75]).tolist() if len(values) else None)
            ns.append(int(counts.sum()))
        out[f"violin_{tissue}_x"], out[f"violin_{tissue}_y"] = xs, ys
        out[f"violin_q_{tissue}"], out[n_key] = quartiles, ns
    return out


# ---------------------------------------------------------------------------
# Panel 5 helper — aggregated density (src/panels.py:216-260)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DensityTable:
    """Panel 5 table plus the bin assignment it aggregated from.

    ``inv`` is a total map from panel-6 (filtered, active_cn-masked) row to
    panel-5 bin index — the array ``np.unique(..., return_inverse=True)``
    already computes and previously discarded. See cnvpostpr.md §3, M3a.
    """

    table: pa.Table
    inv: np.ndarray


def _density_table_impl(
    row: dict[str, object], active_cn: list[str], cn_order: list[str] | None = None
) -> tuple[pa.Table, np.ndarray]:
    x = np.asarray(list(row["scatter_cn_x"]), dtype=float)
    y = np.asarray(list(row["scatter_cn_y"]), dtype=float)
    lvl = np.asarray([cn_label(lv) for lv in row["scatter_cn_level"]])
    v = np.asarray(list(row["scatter_cn_v"]), dtype=np.int64)

    keep = np.isin(lvl, [cn_label(c) for c in active_cn])

    if not keep.any():
        empty = pa.table({
            "cancer_z": pa.array([], type=pa.float64()),
            "healthy_z": pa.array([], type=pa.float64()),
            "bin_id": pa.array([], type=pa.int64()),
            "cCREs": pa.array([], type=pa.int64()),
            "log_cCREs": pa.array([], type=pa.float64()),
        })
        return empty, np.array([], dtype=np.int64)

    x, y, v = x[keep], y[keep], v[keep]

    # Group on the (x, y) float pair — bit-exact because both axes come from GRID_CENTERS
    pairs = np.stack([x, y], axis=1)
    uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
    agg_v = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(agg_v, inv, v)

    log_cCREs = np.log1p(agg_v.astype(float))
    table = pa.table({
        "cancer_z": uniq[:, 0],
        "healthy_z": uniq[:, 1],
        "bin_id": grid_bin_id(uniq[:, 0], uniq[:, 1]),
        "cCREs": agg_v,
        "log_cCREs": log_cCREs,
    })
    return table, inv


def density_table(row: dict[str, object], active_cn: list[str], cn_order: list[str] | None = None) -> pa.Table:
    """Panel 5 table: aggregated cancer-vs-healthy density for selected CN states.

    Transcribed from src/panels.py:216-260. Groups on the raw float pair (x, y)
    rather than on rounded/binned values. Both axes are drawn from the same
    GRID_CENTERS, so grouping on the raw float pair is bit-exact.
    """
    table, _ = _density_table_impl(row, active_cn, cn_order)
    return table


def density_table_with_index(
    row: dict[str, object], active_cn: list[str], cn_order: list[str] | None = None
) -> DensityTable:
    """Like density_table, but also returns the panel-6-row -> panel-5-bin map."""
    table, inv = _density_table_impl(row, active_cn, cn_order)
    return DensityTable(table=table, inv=inv)


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
    bin_ids = grid_bin_id(x, y)
    return pa.table({
        "cancer_z": x[keep],
        "healthy_z": y[keep],
        "bin_id": bin_ids[keep],
        "copy_number": lvl[keep],
        "cCREs": v[keep],
    })


def scatter_row(frame: pl.DataFrame) -> dict[str, object]:
    """Panel 5/6 cells (the ``scatter_cn_*`` columns) recounted from per-cCRE bin_id.

    From an unfiltered frame this reproduces the published cells exactly, so
    a drop_sentinel frame redraws panels 5/6 without the -10 cCREs.
    """
    cells = (
        frame.filter(pl.col("bin_id").is_not_null())
        .group_by("bin_id", "copy_number")
        .len()
        .sort("bin_id", "copy_number")
    )
    bin_id = cells["bin_id"].to_numpy()
    return {
        "scatter_cn_x": GRID_CENTERS[bin_id % N_GRID],
        "scatter_cn_y": GRID_CENTERS[bin_id // N_GRID],
        "scatter_cn_level": cells["copy_number"].to_list(),
        "scatter_cn_v": cells["len"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# Panel 5/6 selections <-> cCREs, through the per-cCRE bin_id of load_zscores
# ---------------------------------------------------------------------------
def selection_ccres(
    frame: pl.DataFrame, selection: tuple[str, tuple], active_cn: list[str]
) -> pl.DataFrame:
    """The cCREs behind a panel 5/6 lasso.

    ``("bins", bin_ids)`` comes from panel 5, whose cells aggregate the active
    copy numbers; ``("points", ((bin_id, copy_number), ...))`` comes from
    panel 6, whose cells are already split by copy number.
    """
    kind, keys = selection
    if kind == "bins":
        return frame.filter(
            pl.col("bin_id").is_in(list(keys))
            & pl.col("copy_number").is_in([cn_label(level) for level in active_cn])
        )
    cells = pl.DataFrame(
        {"bin_id": [key[0] for key in keys], "copy_number": [key[1] for key in keys]},
        schema={"bin_id": pl.Int64, "copy_number": pl.String},
    )
    return frame.join(cells, on=["bin_id", "copy_number"], how="semi")


def scatter_positions(
    rows: pl.DataFrame, dens_tbl: pa.Table, cn_tbl6: pa.Table
) -> tuple[np.ndarray, np.ndarray]:
    """Panel 5 and panel 6 row positions of the cells holding any of ``rows``."""
    bins = rows["bin_id"].drop_nulls().unique().to_numpy()
    bin_pos = np.flatnonzero(np.isin(np.asarray(dens_tbl["bin_id"]), bins))
    cells = pl.from_arrow(cn_tbl6.select(["bin_id", "copy_number"])).with_row_index("row")
    hit = cells.join(
        rows.select("bin_id", "copy_number").drop_nulls().unique(),
        on=["bin_id", "copy_number"],
        how="semi",
    )
    return bin_pos.astype(np.int64), hit["row"].to_numpy().astype(np.int64)


__all__ = [
    "CANCER_BLUE",
    "CCRE_REFERENCE",
    "CCRE_REFERENCE_SCHEMA_VERSION",
    "CCRE_INDEX",
    "CCRE_PREFIX",
    "CCRE_INDEX_DIR",
    "CCRE_INDEX_NAME",
    "CHROMS",
    "CHROM_MID",
    "CHROM_OFFSET",
    "CHROM_SIZES",
    "DATA_DIR",
    "DETAIL",
    "DETAIL_MANIFEST",
    "DETAIL_MAX_ROWS",
    "DETAIL_SCHEMA_VERSION",
    "DEV_REMOTE",
    "DISTRIBUTIONS",
    "Dataset",
    "DensityTable",
    "DetailIndex",
    "FULL_GENOME",
    "GENOME_LEN",
    "GRID_CENTERS",
    "GRID_EDGES",
    "HEALTHY_ORANGE",
    "HIST2D_SHAPE",
    "MISSING_GREY",
    "MEMBERSHIP",
    "MEMBERSHIP_SCHEMA_VERSION",
    "N_GRID",
    "N_VIOLIN_BINS",
    "OVERVIEW_MERGE",
    "PROD_REMOTE",
    "REMOTE",
    "SCHEMA_VERSION",
    "SENTINEL",
    "TABLE_COLUMNS",
    "TRACKS",
    "UNMAPPED",
    "VIOLIN_CENTERS",
    "VIOLIN_EDGES",
    "VIOLIN_HALF_WIDTH",
    "Z_LIM",
    "assert_edges_match",
    "bar_levels",
    "cn_colors",
    "cn_label",
    "cn_table",
    "ccre_reference_source",
    "ccre_index_source",
    "cols_for",
    "density_table",
    "density_table_with_index",
    "drop_sentinel",
    "drop_sentinel_violins",
    "file_constants",
    "genome_layout",
    "load_cohort",
    "load_coords",
    "load_ccre_reference",
    "load_detail_column",
    "load_detail_index",
    "load_membership",
    "load_zscores",
    "mapped_levels",
    "load_loci",
    "load_panel",
    "grid_bin_id",
    "in_polygon",
    "membership_source",
    "open_dataset",
    "panel_1",
    "panel_2",
    "panel_3",
    "panel_4",
    "panel_tags",
    "read_metadata",
    "rgba",
    "scatter_positions",
    "scatter_row",
    "selection_ccres",
    "selection_region",
    "sentinel_mask",
    "source",
    "violin_outline",
    "zscore_frame",
]
