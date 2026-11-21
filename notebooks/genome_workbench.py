# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.14",
#     "anywidget>=0.11.0",
#     "genome-spy-python[arrow]>=0.10.0",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "pyarrow>=18.0.0",
#     "plotly>=6.1.2",
#     "polars>=1.42.0",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
#
# [tool.marimo.runtime]
# output_max_bytes = 50_000_000
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full", app_title="Epifolio Genome Workbench")


@app.cell
def _():
    import functools
    import gzip
    import html
    import io
    import json
    import re
    import urllib.request as urlreq

    import genome_spy as gs
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    import polars as pl
    import pyarrow as pa
    import pyarrow.parquet as pq

    import epifolio.cnv as cnv
    from epifolio import asset_path
    from epifolio.colors import load_color_map
    from epifolio.nmf import component_color_sequence

    return (
        asset_path,
        cnv,
        component_color_sequence,
        functools,
        go,
        gs,
        gzip,
        html,
        io,
        json,
        load_color_map,
        mo,
        np,
        pa,
        pl,
        pq,
        re,
        urlreq,
    )


@app.cell
def _(asset_path, component_color_sequence, load_color_map):
    # ---- Hosted Epifolio payloads and streamed public resources (nothing is downloaded to disk) ----
    EMBEDDINGS = "https://projects.abdenlab.org/itcr/epifolio/embeddings"
    MODELS = {
        "CN-aware adjusted": "adjusted",
        "CN-aware unadjusted": "confounded",
        "Canonical": "tcga.atac",
    }
    CYTOBAND_URL = "https://genomespy.app/data/genomes/hg38/cytoBand.hg38.txt"
    REFSEQ_URL = "https://genomespy.app/data/genomes/hg38/refSeq_genes_scored_compressed.hg38.txt"
    FASTA_URL = "https://data.genomespy.app/genomes/hg38/hg38.fa"
    CATV1 = "https://huggingface.co/programmable-genomics/CATv1"
    ENCODE = "https://www.encodeproject.org"

    # ---- Epifolio palettes ----
    CN_LEVELS = ["0", "1", "2", "3", "4", "5", "6+", "Unmapped"]
    _cn_map = load_color_map(asset_path("conf", "copy_number_color_map.json"))
    CN_COLORS = [_cn_map.get(k, "#d4d4d8") for k in CN_LEVELS]
    PROGRAMS = [f"NMF{i}" for i in range(1, 25)]
    NMF_COLORS = component_color_sequence(
        load_color_map(asset_path("conf", "nmf_component_color_map.json")), len(PROGRAMS)
    )
    CCRE_CLASSES = {
        "PLS": "#ff0000",
        "pELS": "#ffa700",
        "dELS": "#ffcd00",
        "CA-H3K4me3": "#ffaaaa",
        "CA-CTCF": "#00b0f0",
        "CA-TF": "#be28e5",
        "CA": "#808080",
        "TF": "#d876ec",
    }
    GIEMSA = {
        "gneg": "#f5f5f4",
        "gpos25": "#d6d3d1",
        "gpos50": "#a8a29e",
        "gpos75": "#78716c",
        "gpos100": "#44403c",
        "acen": "#dc2626",
        "gvar": "#cbd5e1",
        "stalk": "#94a3b8",
    }
    BASE_COLORS = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "T": "#D62839"}
    NO_READ = -9.99  # cnv.SENTINEL (-10) marks cCREs with no ATAC-seq reads

    # ---- Streaming budgets ----
    DETAIL_MAX = 60_000  # focus-tumor cCREs streamed once the view holds at most this many
    COHORT_MAX = 3_000  # cohort ATAC heatmap streams once the view holds at most this many cCREs
    BLOCK = 16_384  # detail-parquet row block read (and cached) at a time for the cohort heatmap
    TABLE_MAX = 5_000

    # Navigation shortcuts (coordinates only; every value shown comes from the hosted payloads).
    PRESETS = {
        "MYC / PVT1 (8q24.21)": ("chr8", 127_200_000, 129_000_000),
        "ERBB2 / GRB7 (17q12)": ("chr17", 39_100_000, 40_400_000),
        "CCND1 / FGF19 (11q13.3)": ("chr11", 69_100_000, 70_500_000),
        "EGFR (7p11.2)": ("chr7", 54_700_000, 55_900_000),
        "KRAS (12p12.1)": ("chr12", 24_900_000, 25_900_000),
        "ESR1 (6q25.1)": ("chr6", 151_500_000, 152_600_000),
        "GATA3 (10p14)": ("chr10", 7_750_000, 8_650_000),
        "NKX2-1 / FOXA1 (14q13.3)": ("chr14", 36_300_000, 38_000_000),
        "SOX10 (22q13.1)": ("chr22", 37_800_000, 38_600_000),
        "HNF4A (20q13.12)": ("chr20", 44_100_000, 44_900_000),
        "EPAS1 (2p21)": ("chr2", 46_150_000, 47_050_000),
        "PTK2 / LY6E (8q24.3)": ("chr8", 141_600_000, 143_800_000),
    }
    # Cancer cell-line CATv1 experiments used as the default "compare" sequence model.
    CELL_LINE_CATV1 = {
        "BRCA": "ENCSR422SUG",  # MCF-7
        "COAD": "ENCSR872WGW",  # HCT116
        "LUAD": "ENCSR220ASC",  # A549
        "LUSC": "ENCSR220ASC",
        "LIHC": "ENCSR291GJU",  # HepG2
        "CHOL": "ENCSR291GJU",
        "PRAD": "ENCSR499ASS",  # PC-3
        "CESC": "ENCSR959ZXU",  # HeLa-S3
    }
    FALLBACK_COMPARE = "ENCSR859USB"  # K562
    return (
        BASE_COLORS,
        BLOCK,
        CATV1,
        CCRE_CLASSES,
        CELL_LINE_CATV1,
        CN_COLORS,
        CN_LEVELS,
        COHORT_MAX,
        CYTOBAND_URL,
        DETAIL_MAX,
        EMBEDDINGS,
        ENCODE,
        FALLBACK_COMPARE,
        FASTA_URL,
        GIEMSA,
        MODELS,
        NMF_COLORS,
        NO_READ,
        PRESETS,
        PROGRAMS,
        REFSEQ_URL,
        TABLE_MAX,
    )


@app.cell
def _(CATV1, ENCODE, cnv, functools, gzip, io, json, mo, pl, urlreq):
    # ---- External atlases: CATv1 sequence models and ENCODE-rE2G links ----
    def _get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
        req = urlreq.Request(url, headers={"User-Agent": "epifolio-notebook", **(headers or {})})
        with urlreq.urlopen(req, timeout=timeout) as r:
            return r.read()

    def _encode_json(path: str) -> dict:
        return json.loads(_get(f"{ENCODE}{path}", {"Accept": "application/json"}))

    @mo.cache
    def read_catv1_meta() -> pl.DataFrame:
        return pl.read_csv(
            io.BytesIO(_get(f"{CATV1}/resolve/main/CATv1-metadata.tsv")),
            separator="\t",
            infer_schema_length=0,
        )

    @functools.lru_cache(maxsize=32)
    def resolve_model(experiment: str) -> dict | None:
        """CATv1 experiment -> its ENCODE ChromBPNet annotation -> streamable bigWig/bigBed URLs."""
        meta = read_catv1_meta().filter(pl.col("experiment_accession") == experiment)
        if meta.height == 0:
            return None
        row = meta.row(0, named=True)
        ann = _encode_json(f"/annotations/{row['annotation_accession']}/?format=json&frame=embedded")
        wanted = {
            "counts sequence contribution scores": "contrib",
            "normalized bias-corrected predicted signal profile": "pred",
            "sequence motifs instances": "motifs",
        }
        out = {
            "experiment": experiment,
            "annotation": row["annotation_accession"],
            "biosample": row["experiment_biosample_term_name"],
            "assay": row["assay_term_name"],
            "catv1": f"{CATV1}/tree/main/models/{experiment}",
        }
        for f in ann.get("files", []):
            key = wanted.get(f.get("output_type"))
            if key and f.get("file_format") in ("bigWig", "bigBed") and f.get("status") == "released":
                out[key] = (f.get("cloud_metadata") or {}).get("url")
        return out

    @functools.lru_cache(maxsize=16)
    def resolve_e2g(tissue: str) -> tuple[dict | None, pl.DataFrame | None]:
        """ENCODE-rE2G thresholded enhancer->gene links for a tissue (smallest released bedpe)."""
        q = (
            "/search/?type=Annotation&annotation_type=element+gene+regulatory+interaction+predictions"
            f"&biosample_ontology.term_name={urlreq.quote(tissue)}&status=released&format=json&limit=50"
            "&field=accession&field=description&field=files.accession&field=files.file_format"
            "&field=files.output_type&field=files.file_size&field=files.cloud_metadata"
        )
        hits = _encode_json(q).get("@graph", [])
        hits = sorted(hits, key=lambda a: not str(a.get("description", "")).startswith("ENCODE-rE2G"))
        for ann in hits:
            beds = [
                f
                for f in ann.get("files", [])
                if f.get("output_type") == "thresholded links" and f.get("file_format") == "bedpe"
            ]
            if not beds:
                continue
            f = min(beds, key=lambda x: x.get("file_size") or 1e18)
            raw = gzip.decompress(_get((f.get("cloud_metadata") or {}).get("url"), timeout=120))
            links = (
                pl.read_csv(
                    io.BytesIO(raw),
                    separator="\t",
                    has_header=False,
                    new_columns=["chrom", "start", "end", "chrom2", "tss", "tss_end", "name", "score", "s1", "s2"],
                    schema_overrides={"start": pl.Int64, "end": pl.Int64, "tss": pl.Int64, "score": pl.Float32},
                )
                .filter(pl.col("chrom").is_in(cnv.CHROMS))
                .with_columns(pl.col("name").str.split("|").list.first().alias("gene"))
                .select("chrom", "start", "end", "tss", "gene", "score")
            )
            return {"annotation": ann["accession"], "description": ann.get("description", "")}, links
        return None, None

    return read_catv1_meta, resolve_e2g, resolve_model


@app.cell
def _(
    BLOCK,
    CYTOBAND_URL,
    EMBEDDINGS,
    PROGRAMS,
    REFSEQ_URL,
    cnv,
    functools,
    io,
    mo,
    np,
    pl,
    pq,
    urlreq,
):
    # ---- Session-level reads of the hosted Epifolio CNV x ATAC payload ----
    with mo.status.spinner(title="Opening the Epifolio CNV × ATAC payload…"):
        DATA = cnv.open_dataset([cnv.source(n) for n in (cnv.TRACKS, cnv.DISTRIBUTIONS)])
        cnv.assert_edges_match(DATA.schema_for("index"))
        DETAIL = cnv.load_detail_index(DATA.schema_for("2"))

    @mo.cache
    def read_samples() -> pl.DataFrame:
        """One row per tumor aliquot, with a short unique label for heatmap rows."""
        df = (
            pl.scan_parquet(DATA.source_for("index"))
            .select(
                "biosample", "patient_id", "cancer", "sex", "age", "diagnosis",
                "encode_tissue", "healthy_status", "healthy_column",
            )
            .unique("biosample")
            .collect()
            .with_columns(
                pl.col("cancer").str.strip_chars_end("x").alias("cohort"),
                pl.col("biosample").str.split("-").list.slice(1, 4).list.join("-").alias("sample16"),
                pl.col("biosample").map_elements(lambda b: DETAIL.columns.get(b), return_dtype=pl.String).alias("column"),
            )
            .filter(pl.col("column").is_not_null())
            .sort("cohort", "biosample")
        )
        # e.g. "BRCA A0SH-01A"; aliquots of the same sample get a numeric suffix.
        short = df["cohort"] + " " + df["sample16"].str.split("-").list.slice(2, 2).list.join("-")
        df = df.with_columns(short.alias("_s"))
        df = df.with_columns(
            pl.when(pl.len().over("_s") > 1)
            .then(pl.col("_s") + " #" + (pl.int_range(pl.len()).over("_s") + 1).cast(pl.String))
            .otherwise(pl.col("_s"))
            .alias("label")
        ).drop("_s")
        return df

    @mo.cache
    def read_ccre_base() -> pl.DataFrame:
        """Every cCRE in detail-tier row order (genome order), with class and genome-linear start."""
        coords = (
            pl.scan_parquet(DETAIL.src)
            .select("reference_index", "chrom", "start", "end", pl.col("ccre_id").alias("rDHS"))
            .collect()
        )
        meta = pl.scan_parquet(f"{EMBEDDINGS}/adjusted.nmf.ccre.pq").select("rDHS", "cCRE", "class").collect()
        ref = cnv.load_ccre_reference().select("reference_index", "x_bin")
        return (
            coords.join(meta, on="rDHS", how="left", maintain_order="left")
            .join(ref, on="reference_index", how="left", maintain_order="left")
            .with_columns(
                (pl.col("chrom").replace_strict(cnv.CHROM_OFFSET, default=None, return_dtype=pl.Int64) + pl.col("start")).alias("lin")
            )
        )

    @mo.cache
    def read_cohort_cn() -> pl.DataFrame:
        """All tumors' ASCAT copy number as run-length segments in genome order.

        The membership parquet stores one 2.35M-long code array per tumor; it is decoded a few
        rows at a time and reduced to segments in memory, so the full matrix is never held.
        """
        base = read_ccre_base()
        ref_idx = base["reference_index"].to_numpy()
        chrom = base["chrom"].to_numpy()
        starts = base["start"].to_numpy()
        ends = base["end"].to_numpy()
        chrom_breaks = np.flatnonzero(chrom[1:] != chrom[:-1]) + 1
        src = cnv.membership_source()
        if src.startswith("http"):
            with urlreq.urlopen(src) as r:
                src = io.BytesIO(r.read())
        label = dict(read_samples().select("biosample", "label").iter_rows())
        parts = []
        for batch in pq.ParquetFile(src).iter_batches(batch_size=4, columns=["biosample", "copy_number_code"]):
            names = batch.column(0).to_pylist()
            col = batch.column(1)
            vals = col.values.to_numpy(zero_copy_only=False)
            off = col.offsets.to_numpy()
            for i, b in enumerate(names):
                if b not in label:
                    continue
                codes = vals[off[i] : off[i + 1]][ref_idx]
                i0 = np.union1d(np.concatenate(([0], np.flatnonzero(codes[1:] != codes[:-1]) + 1)), chrom_breaks)
                i1 = np.concatenate((i0[1:], [len(codes)])) - 1
                parts.append(
                    pl.DataFrame(
                        {
                            "chrom": chrom[i0],
                            "start": starts[i0].astype(np.int32),
                            "end": ends[i1].astype(np.int32),
                            "code": codes[i0].astype(np.int8),
                            "label": [label[b]] * len(i0),
                            "biosample": [b] * len(i0),
                        }
                    )
                )
        lv = ["0", "1", "2", "3", "4", "5", "6+", "Unmapped"]
        return pl.concat(parts).with_columns(
            pl.col("code").cast(pl.Int64).replace_strict(dict(enumerate(lv)), return_dtype=pl.String).alias("copy_number")
        )

    @functools.lru_cache(maxsize=6)
    def read_z(column: str) -> pl.Series:
        return cnv.load_detail_column(DETAIL, column)

    @functools.lru_cache(maxsize=4)
    def read_cn(biosample: str) -> pl.DataFrame:
        return cnv.load_membership(biosample, reference=cnv.load_ccre_reference()).select(
            "reference_index", "copy_number_code", "copy_number"
        )

    @functools.lru_cache(maxsize=1)
    def read_w(model_file: str) -> np.ndarray:
        """All 24 cCRE loadings (float32), rows aligned to read_ccre_base()."""
        w = (
            pl.scan_parquet(f"{EMBEDDINGS}/{model_file}.nmf.ccre.pq")
            .select(["rDHS"] + [pl.col(p).cast(pl.Float32) for p in PROGRAMS])
            .collect()
            .with_row_index("_wi")
        )
        idx = (
            read_ccre_base()
            .select("rDHS")
            .join(w.select("rDHS", "_wi"), on="rDHS", how="left", maintain_order="left")["_wi"]
            .to_numpy()
        )
        mat = w.select(PROGRAMS).to_numpy()
        out = np.zeros((len(idx), len(PROGRAMS)), dtype=np.float32)
        ok = ~np.isnan(idx.astype(float))
        out[ok] = mat[idx[ok].astype(np.int64)]
        return out

    @functools.lru_cache(maxsize=3)
    def read_h(model_file: str) -> pl.DataFrame:
        return pl.read_parquet(f"{EMBEDDINGS}/{model_file}.nmf.sample.pq")

    @functools.lru_cache(maxsize=6)
    def read_block(block: int) -> pl.DataFrame:
        """Every tumor and healthy z column for one block of detail rows (cohort heatmap)."""
        cols = sorted(set(DETAIL.columns.values()) | {h for h in DETAIL.healthy.values() if h})
        return pl.scan_parquet(DETAIL.src).slice(block * BLOCK, BLOCK).select(cols).collect()

    @mo.cache
    def read_genes() -> pl.DataFrame:
        return (
            pl.read_csv(
                REFSEQ_URL,
                separator="\t",
                has_header=False,
                new_columns=["symbol", "chrom", "start", "length", "strand", "score", "exons"],
                schema_overrides={"start": pl.Int64, "length": pl.Int64, "score": pl.Int64},
            )
            .filter(pl.col("chrom").is_in(cnv.CHROMS))
            .with_columns(
                (pl.col("start") + pl.col("length")).alias("end"),
                (pl.col("chrom").replace_strict(cnv.CHROM_OFFSET, return_dtype=pl.Int64) + pl.col("start")).alias("lin"),
            )
            .sort("lin")
        )

    @mo.cache
    def read_cytobands() -> pl.DataFrame:
        return pl.read_csv(
            CYTOBAND_URL,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "chromStart", "chromEnd", "name", "gieStain"],
        ).filter(pl.col("chrom").is_in(cnv.CHROMS))

    with mo.status.spinner(title="Loading cCRE coordinates and the cohort's copy-number segments…"):
        SAMPLES = read_samples()
        BASE = read_ccre_base()
        LIN = BASE["lin"].to_numpy()
        COHORT_CN = read_cohort_cn()
    return (
        BASE,
        COHORT_CN,
        DETAIL,
        LIN,
        SAMPLES,
        read_block,
        read_cn,
        read_cytobands,
        read_genes,
        read_h,
        read_w,
        read_z,
    )


@app.cell
def _(
    CELL_LINE_CATV1,
    DETAIL,
    FALLBACK_COMPARE,
    SAMPLES,
    mo,
    pl,
    read_catv1_meta,
    resolve_model,
):
    # ---- Initial focus tumor and its tissue-matched sequence models (spec defaults only) ----
    _first = SAMPLES.filter((pl.col("cohort") == "BRCA") & pl.col("biosample").map_elements(lambda b: bool(DETAIL.healthy.get(b)), return_dtype=pl.Boolean))
    INITIAL_FOCUS = (_first if _first.height else SAMPLES).row(0, named=True)["biosample"]
    _catv1_ids = set(read_catv1_meta()["experiment_accession"].to_list())

    def default_models(biosample: str) -> tuple[str, str]:
        """Matched-normal CATv1 experiment (the Epifolio ENCODE normal) and a cancer cell-line comparison."""
        h = DETAIL.healthy.get(biosample)
        cohort = biosample.split("-", 1)[0].rstrip("x")
        primary = h[2:] if h and h[2:] in _catv1_ids else CELL_LINE_CATV1.get(cohort, FALLBACK_COMPARE)
        compare = CELL_LINE_CATV1.get(cohort, FALLBACK_COMPARE)
        if compare == primary:
            compare = FALLBACK_COMPARE
        return primary, compare

    with mo.status.spinner(title="Resolving CATv1 → ENCODE ChromBPNet tracks…"):
        _p, _c = default_models(INITIAL_FOCUS)
        INITIAL_URLS = {}
        for _key, _exp in (("p", _p), ("c", _c)):
            try:
                _m = resolve_model(_exp) or {}
            except Exception:
                _m = {}
            INITIAL_URLS[_key] = _m
    return INITIAL_FOCUS, INITIAL_URLS, default_models


@app.cell
def _(INITIAL_FOCUS, mo):
    # Cross-cell state. The browser writes these from its callbacks; it never reads them,
    # so the browser cell itself is never re-run by them.
    get_focus, set_focus = mo.state(INITIAL_FOCUS)
    get_sel, set_sel = mo.state(None)
    get_stream, set_stream = mo.state("")
    # Plain mutable store shared by the browser callbacks and the push cells below.
    STATE: dict = {"view": None, "focus_loaded": None, "cohort_key": None, "rows": [], "atac_rows": []}
    return (
        STATE,
        get_focus,
        get_sel,
        get_stream,
        set_focus,
        set_sel,
        set_stream,
    )


@app.cell
def _(
    BASE_COLORS,
    CCRE_CLASSES,
    CN_COLORS,
    CN_LEVELS,
    COHORT_MAX,
    FASTA_URL,
    GIEMSA,
    INITIAL_URLS,
    NMF_COLORS,
    PRESETS,
    PROGRAMS,
):
    # ---- The GenomeSpy specification: built once per session ----
    # Every control below drives this view through parameters or dataset swaps, never by rebuilding it.
    _c, _s, _e = PRESETS["MYC / PVT1 (8q24.21)"]
    INITIAL_DOMAIN = [{"chrom": _c, "pos": _s}, {"chrom": _c, "pos": _e}]
    _link = {"domain": {"param": "nav", "initial": INITIAL_DOMAIN}}

    def X(pos="start", chrom="chrom", axis=None):
        return {"chrom": chrom, "pos": pos, "type": "locus", "scale": _link, "axis": axis}

    def X2(pos="end", chrom="chrom"):
        return {"chrom": chrom, "pos": pos}

    def title(text):
        return {"text": text, "style": "overlay-title"}

    _cn_scale = {"domain": CN_LEVELS, "range": CN_COLORS}
    _fade_out = {"unitsPerPixel": [60_000, 20_000], "values": [1, 0]}
    _fade_in = {"unitsPerPixel": [60_000, 20_000], "values": [0, 1]}
    _noread = {"type": "filter", "expr": "!hideNoRead || datum.tumor_z > -9.99"}
    _prog = {"type": "filter", "expr": "progFocus == 'all' || datum.nmf == progFocus"}
    _ccre_tip = [
        {"field": "cCRE", "title": "cCRE"},
        {"field": "class", "title": "ENCODE class"},
        {"field": "copy_number", "title": "ASCAT CN"},
        {"field": "tumor_z", "title": "Tumor ATAC z", "format": ".2f"},
        {"field": "healthy_z", "title": "Normal z", "format": ".2f"},
        {"field": "nmf", "title": "Dominant program"},
        {"field": "w", "title": "Loading", "format": ".3f"},
    ]
    _bin_tip = [
        {"field": "start", "title": "Bin start"},
        {"field": "end", "title": "Bin end"},
        {"field": "n_ccres", "title": "cCREs"},
        {"field": "copy_number", "title": "Median CN"},
        {"field": "tumor_z", "title": "Mean tumor z", "format": ".2f"},
        {"field": "healthy_z", "title": "Mean normal z", "format": ".2f"},
        {"field": "nmf", "title": "Most common program"},
    ]

    # -- Whole-genome overview: focus tumor's copy number; drag here to navigate --
    overview = {
        "name": "overview",
        "height": 26,
        "resolve": {"scale": {"x": "excluded"}},
        "params": [
            {
                "name": "nav",
                "push": "outer",
                "persist": False,
                "select": {
                    "type": "interval",
                    "encodings": ["x"],
                    "mark": {"fill": "#2563eb", "fillOpacity": 0.15, "stroke": "#1d4ed8", "strokeWidth": 1.2},
                },
            }
        ],
        "layer": [
            {
                "data": {"name": "overview_cn"},
                "mark": {"type": "rect", "minWidth": 0.5},
                "encoding": {
                    "x": {
                        "chrom": "chrom",
                        "pos": "start",
                        "type": "locus",
                        "scale": {"zoom": False},
                        "axis": {"orient": "top", "title": None, "chromTicks": True, "chromLabels": True},
                    },
                    "x2": X2(),
                    "color": {"field": "copy_number", "type": "nominal", "scale": _cn_scale, "legend": None},
                    "tooltip": [
                        {"field": "chrom", "title": "Chrom"},
                        {"field": "start", "title": "Start"},
                        {"field": "end", "title": "End"},
                        {"field": "copy_number", "title": "Focus tumor CN"},
                    ],
                },
            }
        ],
    }

    cyto = {
        "name": "cytobands",
        "height": 14,
        "data": {"name": "cyto"},
        "encoding": {"x": X("chromStart", axis={"orient": "top", "title": None}), "x2": X2("chromEnd")},
        "layer": [
            {
                "mark": {"type": "rect", "stroke": "#a8a29e", "strokeWidth": 0.3},
                "encoding": {
                    "color": {
                        "field": "gieStain",
                        "type": "nominal",
                        "scale": {"domain": list(GIEMSA), "range": list(GIEMSA.values())},
                        "legend": None,
                    },
                    "tooltip": [{"field": "chrom", "title": "Chrom"}, {"field": "name", "title": "Band"}],
                },
            },
            {
                "transform": [{"type": "filter", "expr": "datum.gieStain == 'gneg' || datum.gieStain == 'gpos25'"}],
                "mark": {"type": "text", "size": 9, "color": "#1f2937", "tooltip": None},
                "encoding": {"text": {"field": "name"}},
            },
        ],
    }

    genes = {
        "name": "genes",
        "height": 38,
        "data": {"name": "genes"},
        "transform": [
            {"type": "linearizeGenomicCoordinate", "chrom": "chrom", "pos": ["start", "end"], "as": ["_start", "_end"]},
            {"type": "collect", "sort": {"field": ["_start"]}},
            {"type": "pileup", "start": "_start", "end": "_end", "as": "_lane", "spacing": 5000},
            {"type": "filter", "expr": "datum._lane < 3"},
        ],
        "encoding": {
            "y": {
                "field": "_lane",
                "type": "index",
                "axis": None,
                "scale": {"domain": [0, 3], "zoom": False, "paddingInner": 0.45, "paddingOuter": 0.25},
            }
        },
        "layer": [
            {
                "mark": {"type": "rule", "color": "#64748b", "size": 1},
                "encoding": {
                    "x": {"field": "_start", "type": "locus", "scale": _link, "axis": None},
                    "x2": {"field": "_end"},
                    "tooltip": [
                        {"field": "symbol", "title": "Gene"},
                        {"field": "strand", "title": "Strand"},
                        {"field": "start", "title": "Start"},
                        {"field": "end", "title": "End"},
                    ],
                },
            },
            {
                "transform": [{"type": "flattenCompressedExons", "start": "_start"}],
                "mark": {"type": "rect", "color": "#1e3a8a", "minWidth": 0.5, "minOpacity": 0.5, "tooltip": None},
                "encoding": {"x": {"field": "exonStart", "type": "locus", "axis": None}, "x2": {"field": "exonEnd"}},
            },
            {
                "transform": [
                    {"type": "measureText", "field": "symbol", "fontSize": 10, "as": "_tw"},
                    {
                        "type": "filterScoredLabels",
                        "lane": "_lane",
                        "score": "score",
                        "pos": "_start",
                        "pos2": "_end",
                        "asMidpoint": "_mid",
                        "width": "_tw",
                        "padding": 6,
                    },
                ],
                "mark": {"type": "text", "size": 10, "dy": -7, "color": "#0f172a", "tooltip": None},
                "encoding": {"x": {"field": "_mid", "type": "locus", "axis": None}, "text": {"field": "symbol"}},
            },
        ],
    }

    # -- Focus tumor --
    ascat = {
        "name": "ascat",
        "height": 10,
        "data": {"name": "segments"},
        "mark": {"type": "rect", "minWidth": 0.5},
        "encoding": {
            "x": X(),
            "x2": X2(),
            "color": {"field": "copy_number", "type": "nominal", "scale": _cn_scale, "legend": None},
            "tooltip": [
                {"field": "start", "title": "Segment start"},
                {"field": "end", "title": "Segment end"},
                {"field": "copy_number", "title": "ASCAT CN"},
                {"field": "n_ccres", "title": "cCREs"},
            ],
        },
    }

    def z_track(name, field, label, height, coloured):
        fill = {"value": "#94a3b8"}
        line = "#1e293b" if coloured else "#475569"

        def profile(data, filters, tooltip, max_gap):
            # Continuous z profile: a 3-point running mean, joined to the next point by a line segment and
            # shaded down to zero. GenomeSpy has no line/area mark, so both come from a window transform
            # over neighbours, applied after the filters so hidden cCREs never bridge the line.
            steps = filters + [
                {"type": "formula", "expr": "(datum.start + datum.end) / 2", "as": "_x"},
                {"type": "window", "groupby": ["chrom"], "sort": {"field": "_x"}, "ops": ["mean"], "fields": [field], "frame": [-1, 1], "as": ["_zs"]},
                {"type": "window", "groupby": ["chrom"], "sort": {"field": "_x"}, "ops": ["lead", "lead"], "fields": ["_x", "_zs"], "as": ["_x2", "_zs2"]},
                {"type": "filter", "expr": f"isValid(datum._x2) && datum._x2 - datum._x <= {max_gap}"},
                {"type": "formula", "expr": "(datum._zs + datum._zs2) / 2", "as": "_zm"},
            ]
            return [
                {
                    "data": data,
                    "transform": steps,
                    "mark": {"type": "rect", "minWidth": 0.5, "opacity": 0.3},
                    "encoding": {"x": X("_x"), "x2": X2("_x2"), "y": {"field": "_zm", "type": "quantitative"}, "y2": {"datum": 0}, "color": fill, "tooltip": tooltip},
                },
                {
                    "data": data,
                    "transform": steps,
                    "mark": {"type": "rule", "size": 1.4, "color": line, "tooltip": None},
                    "encoding": {"x": X("_x"), "x2": X2("_x2"), "y": {"field": "_zs", "type": "quantitative"}, "y2": {"field": "_zs2"}},
                },
            ]

        _ccre_filters = [_noread, _prog] if coloured else [{"type": "filter", "expr": f"datum.{field} > -9.99"}, _prog]
        return {
            "name": name,
            "height": height,
            "title": title(label),
            "encoding": {
                "y": {
                    "field": field,
                    "type": "quantitative",
                    "title": None,
                    "scale": {"domain": [-3, 5], "clamp": True},
                    "axis": {"tickCount": 3},
                }
            },
            "layer": [
                {
                    "data": {"values": [{}]},
                    "mark": {"type": "rule", "color": "#94a3b8", "strokeDash": [3, 3], "tooltip": None},
                    "encoding": {"y": {"datum": 0, "type": "quantitative"}},
                },
                {"opacity": _fade_out, "layer": profile({"name": "bins"}, [{"type": "filter", "expr": f"isValid(datum.{field})"}], _bin_tip, 1e12)},
                {"opacity": _fade_in, "layer": profile({"name": "ccres"}, _ccre_filters, _ccre_tip, 50_000)},
            ],
        }

    tumor = z_track("tumor", "tumor_z", "Focus tumor - ATAC z (shift+drag to select)", 96, True)
    tumor["params"] = [{"name": "sel", "select": {"type": "interval", "encodings": ["x"]}}]
    normal = z_track("normal", "healthy_z", "Matched ENCODE normal - ATAC z", 44, False)

    nmf = {
        "name": "nmf",
        "height": 56,
        "title": title("NMF cCRE loadings (W) - stacked top programs"),
        "encoding": {
            "y": {"field": "_y1", "type": "quantitative", "title": None, "scale": {"zero": True}, "axis": {"tickCount": 2}},
            # Program colours straight from the Epifolio NMF colour map (NMF{i+1} <- Comp_{i}).
            "color": {
                "field": "nmf",
                "type": "nominal",
                "scale": {"domain": PROGRAMS + ["none"], "range": NMF_COLORS + ["#cbd5e1"]},
                "legend": None,
            },
        },
        "layer": [
            {
                "data": {"name": "bins"},
                "opacity": _fade_out,
                "transform": [{"type": "formula", "expr": "datum.w", "as": "_y1"}],
                "mark": {"type": "rect", "minWidth": 0.5},
                "encoding": {"x": X(), "x2": X2(), "y2": {"datum": 0}, "tooltip": _bin_tip},
            },
            {
                "data": {"name": "nmf_stack"},
                "opacity": _fade_in,
                "transform": [_prog, {"type": "formula", "expr": "datum.y1", "as": "_y1"}],
                "mark": {"type": "rect", "minWidth": 1, "minOpacity": 0.8},
                "encoding": {
                    "x": X(),
                    "x2": X2(),
                    "y2": {"field": "y0"},
                    "tooltip": [
                        {"field": "cCRE", "title": "cCRE"},
                        {"field": "nmf", "title": "Program"},
                        {"field": "weight", "title": "W", "format": ".3f"},
                        {"field": "rank", "title": "Rank in cCRE"},
                    ],
                },
            },
        ],
    }

    # -- Cohort heatmaps; row order and membership are parameters set from Python --
    def heat_y(rows_param):
        return {
            "field": "label",
            "type": "nominal",
            "scale": {"domain": {"expr": rows_param}, "paddingInner": 0, "paddingOuter": 0},
            "axis": {"title": None, "labelFontSize": 8, "labelOverlap": "greedy", "labelSeparation": 2, "ticks": False},
        }

    def focus_outline(source_name, rows_param):
        return {
            "data": {"name": source_name},
            "transform": [{"type": "filter", "expr": f"datum.label == focusLab && indexof({rows_param}, datum.label) >= 0"}],
            "mark": {"type": "rect", "fillOpacity": 0, "stroke": "#111827", "strokeWidth": 1.2, "minWidth": 1, "tooltip": None},
            "encoding": {"x": X(), "x2": X2(), "y": heat_y(rows_param)},
        }

    cohort_cn = {
        "name": "cohort_cn",
        "height": 210,
        "title": title("Cohort - ASCAT copy number (click a row to focus that tumor)"),
        "layer": [
            {
                "data": {"name": "cohort_cn"},
                "transform": [{"type": "filter", "expr": "indexof(cohortRows, datum.label) >= 0"}],
                "mark": {"type": "rect", "minWidth": 0.5},
                "encoding": {
                    "x": X(),
                    "x2": X2(),
                    "y": heat_y("cohortRows"),
                    "color": {"field": "copy_number", "type": "nominal", "scale": _cn_scale, "legend": None},
                    "tooltip": [
                        {"field": "label", "title": "Tumor"},
                        {"field": "copy_number", "title": "ASCAT CN"},
                        {"field": "start", "title": "Segment start"},
                        {"field": "end", "title": "Segment end"},
                    ],
                },
            },
            focus_outline("cohort_cn", "cohortRows"),
        ],
    }

    cohort_atac = {
        "name": "cohort_atac",
        "height": 230,
        "title": title(f"Cohort - ATAC z per cCRE, with matched normals (streams below ~{COHORT_MAX:,} cCREs in view)"),
        "layer": [
            {
                "data": {"name": "cohort_atac"},
                "transform": [{"type": "filter", "expr": "indexof(atacRows, datum.label) >= 0"}],
                "mark": {"type": "rect", "minWidth": 1, "minOpacity": 0.9},
                "encoding": {
                    "x": X(),
                    "x2": X2(),
                    "y": heat_y("atacRows"),
                    "color": {
                        "field": "z",
                        "type": "quantitative",
                        "scale": {
                            "domain": {"expr": "[-zMax, zMax]"},
                            "domainMid": 0,
                            "clamp": True,
                            "range": ["#2563eb", "#f8fafc", "#dc2626"],
                        },
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "label", "title": "Sample"},
                        {"field": "z", "title": "ATAC z", "format": ".2f"},
                        {"field": "cCRE", "title": "cCRE"},
                    ],
                },
            },
            focus_outline("cohort_atac", "atacRows"),
        ],
    }

    # Same encoding as the NMF locus browser: blue domes, opacity scaled by the rE2G score.
    links = {
        "name": "links",
        "height": 48,
        "title": title("Enhancer -> gene - ENCODE-rE2G (matched tissue)"),
        "data": {"name": "links"},
        "transform": [
            {"type": "formula", "expr": "0", "as": "_base"},
            {"type": "formula", "expr": "1", "as": "_apex"},
            {"type": "formula", "expr": "max(0.06, min(1, (datum.score - 0.18) / 0.72))", "as": "_op"},
        ],
        "mark": {
            "type": "link",
            "linkShape": "dome",
            "orient": "vertical",
            "clampApex": True,
            "color": "#2563eb",
            "size": 1.5,
            "minPickingSize": 5,
        },
        "encoding": {
            "x": X("mid"),
            "x2": X2("tss"),
            "y": {"field": "_base", "type": "quantitative", "scale": {"domain": [0, 1]}, "axis": None},
            "y2": {"field": "_apex"},
            "opacity": {"field": "_op", "type": "quantitative", "scale": None},
            "tooltip": [
                {"field": "gene", "title": "Target gene"},
                {"field": "score", "title": "rE2G score", "format": ".3f"},
                {"field": "cell", "title": "Tissue"},
                {"field": "start", "title": "Enhancer start"},
                {"field": "end", "title": "Enhancer end"},
                {"field": "tss", "title": "Gene TSS"},
            ],
        },
    }

    ccre_strip = {
        "name": "ccre_strip",
        "height": 24,
        "title": title("ENCODE v4 cCREs"),
        "data": {"name": "ccres"},
        "mark": {"type": "rect", "minWidth": 1.5, "minOpacity": 0.85},
        "encoding": {
            "x": X(),
            "x2": X2(),
            "color": {
                "field": "class",
                "type": "nominal",
                "scale": {"domain": list(CCRE_CLASSES), "range": list(CCRE_CLASSES.values())},
                "legend": None,
            },
            "tooltip": [
                {"field": "cCRE", "title": "cCRE"},
                {"field": "class", "title": "ENCODE class"},
                {"field": "nmf", "title": "Dominant program"},
                {"field": "copy_number", "title": "ASCAT CN"},
            ],
        },
    }

    # -- CATv1 sequence models; their bigWig/bigBed URLs are parameters, so swapping models streams new tracks --
    pred = {
        "name": "pred",
        "height": 30,
        "title": title("Primary sequence model - predicted accessibility"),
        "data": {"lazy": {"type": "bigwig", "url": {"expr": "predUrl"}, "pixelsPerBin": 2}},
        "mark": {"type": "rect", "minWidth": 0.5, "color": "#7c3aed"},
        "encoding": {
            "x": X(),
            "x2": X2(),
            "y": {"field": "score", "type": "quantitative", "title": None, "axis": {"tickCount": 2}},
            "y2": {"datum": 0},
            "tooltip": [{"field": "score", "title": "Predicted accessibility", "format": ".3f"}],
        },
    }

    def dynseq(name, url_param, label, colour):
        # Bars from the bigWig at wide zoom; at base resolution the hg38 letters (indexed FASTA,
        # streamed by the browser) are drawn at the height of their contribution score.
        _bars = {"unitsPerPixel": [0.8, 0.3], "values": [1, 0]}
        _letters = {"unitsPerPixel": [0.8, 0.3], "values": [0, 1]}
        y = {"field": "score", "type": "quantitative", "title": None, "axis": {"tickCount": 2}}
        return {
            "name": name,
            "height": 54,
            "title": title(label),
            "layer": [
                {
                    "data": {"values": [{}]},
                    "mark": {"type": "rule", "color": "#cbd5e1", "tooltip": None},
                    "encoding": {"y": {"datum": 0, "type": "quantitative"}},
                },
                {
                    "opacity": _bars,
                    "data": {"lazy": {"type": "bigwig", "url": {"expr": url_param}, "pixelsPerBin": 1}},
                    "mark": {"type": "rect", "minWidth": 0.5, "color": colour},
                    "encoding": {
                        "x": X(),
                        "x2": X2(),
                        "y": y,
                        "y2": {"datum": 0},
                        "tooltip": [{"field": "score", "title": "Contribution", "format": ".4f"}],
                    },
                },
                {
                    "opacity": _letters,
                    "data": {"lazy": {"type": "indexedFasta", "url": FASTA_URL}},
                    "transform": [
                        {"type": "flattenSequence", "field": "sequence", "as": ["rawPos", "base"]},
                        {"type": "formula", "expr": "upper(datum.base)", "as": "base"},
                        {"type": "formula", "expr": "datum.start + datum.rawPos", "as": "pos"},
                        {
                            "type": "coordinateLookup",
                            "from": {
                                "data": {"lazy": {"type": "bigwig", "url": {"expr": url_param}, "pixelsPerBin": 1}},
                                "transform": [{"type": "formula", "expr": "datum.start", "as": "pos"}],
                            },
                            "key": ["chrom", "pos"],
                            "values": ["score"],
                        },
                        {"type": "filter", "expr": "isValid(datum.score)"},
                    ],
                    "mark": {"type": "text", "logoLetters": True, "squeeze": True, "fitToBand": True, "paddingX": 0, "paddingY": 0},
                    "encoding": {
                        "x": X("pos"),
                        "y": {"datum": 0, "type": "quantitative"},
                        "y2": {"field": "score"},
                        "text": {"field": "base"},
                        "color": {
                            "field": "base",
                            "type": "nominal",
                            "scale": {"domain": list(BASE_COLORS), "range": list(BASE_COLORS.values())},
                            "legend": None,
                        },
                        "tooltip": [{"field": "base", "title": "Base"}, {"field": "score", "title": "Contribution", "format": ".4f"}],
                    },
                },
            ],
        }

    motifs = {
        "name": "motifs",
        "height": 20,
        "title": title("Primary model - TF motif hits"),
        "data": {"lazy": {"type": "bigbed", "url": {"expr": "motifUrl"}}},
        "encoding": {"x": X("chromStart", axis={"title": None}), "x2": X2("chromEnd")},
        "layer": [
            {
                "mark": {"type": "rect", "minWidth": 1, "color": "#0d9488", "opacity": 0.35},
                "encoding": {"tooltip": [{"field": "name", "title": "Motif"}, {"field": "score", "title": "Score"}]},
            },
            {
                "mark": {"type": "text", "size": 9, "squeeze": True, "color": "#134e4a", "tooltip": None},
                "encoding": {"text": {"field": "name"}},
            },
        ],
    }

    _p, _c = INITIAL_URLS["p"], INITIAL_URLS["c"]
    _params = [
        {"name": "nav"},
        # Browser-only display controls (rendered by GenomeSpy; no Python round trip).
        {
            "name": "progFocus",
            "value": "all",
            "bind": {"input": "select", "options": ["all"] + PROGRAMS, "name": "Only program "},
        },
        {"name": "hideNoRead", "value": True, "bind": {"input": "checkbox", "name": "Hide no-read cCREs "}},
        {"name": "zMax", "value": 4, "bind": {"input": "range", "min": 1, "max": 10, "step": 0.5, "name": "Heatmap |z| "}},
        # Set from Python.
        {"name": "cohortRows", "value": []},
        {"name": "atacRows", "value": []},
        {"name": "focusLab", "value": ""},
        {"name": "predUrl", "value": _p.get("pred") or ""},
        {"name": "contribUrl", "value": _p.get("contrib") or ""},
        {"name": "motifUrl", "value": _p.get("motifs") or ""},
        {"name": "cmpUrl", "value": _c.get("contrib") or ""},
    ]
    # Genes -> rE2G links -> cCREs -> focus tumor CN + ATAC vs matched normal -> NMF loadings ->
    # sequence-model tracks -> cohort ATAC heatmap -> cohort copy number.
    TRACK_ORDER = [cyto, genes, links, ccre_strip, ascat, tumor, normal, nmf, pred,
                   dynseq("dynseq", "contribUrl", "Primary model - contribution scores (dynseq letters below ~300 bp)", "#0f766e"),
                   dynseq("dynseq_cmp", "cmpUrl", "Compare model - contribution scores", "#b45309"),
                   motifs, cohort_atac, cohort_cn]
    BROWSER_SPEC = {
        "assembly": "hg38",
        "datasets": {k: [] for k in ("cyto", "genes", "overview_cn", "segments", "bins", "ccres", "nmf_stack", "cohort_cn", "cohort_atac", "links")},
        "params": _params,
        "resolve": {"scale": {"x": "independent"}},
        "spacing": 6,
        "vconcat": [
            overview,
            {
                "spacing": 4,
                "resolve": {"scale": {"x": "shared", "y": "independent", "color": "independent"}},
                "vconcat": TRACK_ORDER,
            },
        ],
    }
    return (BROWSER_SPEC,)


@app.cell
def _(
    BASE,
    BLOCK,
    BROWSER_SPEC,
    COHORT_CN,
    COHORT_MAX,
    DETAIL_MAX,
    LIN,
    PROGRAMS,
    STATE: dict,
    cnv,
    gs,
    np,
    pa,
    pl,
    read_block,
    read_cytobands,
    read_genes,
    set_focus,
    set_sel,
    set_stream,
):
    # ---- Mount GenomeSpy exactly once. Nothing reactive is read here, so no control re-runs this cell. ----
    _DICT_COLS = {"chrom", "class", "copy_number", "nmf", "source", "cell", "label", "biosample", "gene", "kind"}

    def to_arrow(df: pl.DataFrame) -> pa.Table:
        # GenomeSpy's Arrow reader wants utf8 (not string_view); repeated text goes as dictionaries.
        t = df.to_arrow().combine_chunks()
        cols, fields = [], []
        for col, f in zip(t.columns, t.schema):
            if f.name in _DICT_COLS or pa.types.is_dictionary(f.type):
                typ = pa.dictionary(pa.int32(), pa.string())
                col = col.cast(pa.string()).cast(typ)
            elif pa.types.is_string_view(f.type) or pa.types.is_large_string(f.type):
                typ = pa.string()
                col = col.cast(typ)
            else:
                typ = f.type
            cols.append(col)
            fields.append(pa.field(f.name, typ))
        return pa.Table.from_arrays(cols, schema=pa.schema(fields))

    browser = gs.TopLevelSpec.from_dict(BROWSER_SPEC, validate=True).widget(
        inline=True,
        enable_click_events=True,
        parameter_names=["nav", "sel", "cohortRows", "atacRows", "focusLab", "predUrl", "contribUrl", "motifUrl", "cmpUrl"],
    )

    def push(name: str, df: pl.DataFrame) -> None:
        browser.set_dataset(name, to_arrow(df))

    _desired: dict = {}

    def set_params(**values) -> None:
        # Python-owned parameters are kept as one desired state, so values written before the
        # browser mounts are all applied. Navigation ("nav") is sent only when explicitly requested,
        # so these writes never move the view.
        _desired.update({k: v for k, v in values.items() if k != "nav"})
        browser.parameter_values = {**_desired, **({"nav": values["nav"]} if "nav" in values else {})}

    push("cyto", read_cytobands())
    push("cohort_cn", COHORT_CN.select("chrom", "start", "end", "copy_number", "label", "biosample"))

    _genes = read_genes()
    _genes_lin = _genes["lin"].to_numpy()

    def lin_to_pos(x: float) -> tuple[str, int]:
        for c in reversed(cnv.CHROMS):
            if x >= cnv.CHROM_OFFSET[c]:
                return c, int(x - cnv.CHROM_OFFSET[c])
        return "chr1", int(x)

    # ---- Focus tumor: per-cCRE detail, stacked loadings, links and genes for the visible window ----
    def stream_focus(lo: float, hi: float, force: bool = False) -> None:
        frame, W = STATE.get("frame"), STATE.get("W")
        if frame is None or W is None:
            return
        i0, i1 = (int(v) for v in np.searchsorted(LIN, [lo, hi]))
        if i1 - i0 > DETAIL_MAX:
            if STATE.get("focus_loaded") is not None:
                for _n in ("ccres", "nmf_stack", "links"):
                    push(_n, STATE["empty"][_n])
                push("genes", _genes.head(0).drop("lin", "length"))
                STATE["focus_loaded"] = None
            return
        got = STATE.get("focus_loaded")
        if not force and got is not None and lo >= got[0] and hi <= got[1]:
            return
        pad = hi - lo
        j0, j1 = (int(v) for v in np.searchsorted(LIN, [lo - pad, hi + pad]))
        if j1 - j0 > 2 * DETAIL_MAX:
            j0, j1, pad = i0, i1, 0
        sl = frame.slice(j0, j1 - j0)
        dom = STATE["dom_label"].slice(j0, j1 - j0).fill_null("none")
        wd = STATE["wdom"][j0:j1]
        push(
            "ccres",
            sl.select(
                "cCRE",
                "chrom",
                pl.col("start").cast(pl.Int32),
                pl.col("end").cast(pl.Int32),
                ((pl.col("start") + pl.col("end")) // 2).cast(pl.Int32).alias("mid"),
                "class",
                pl.col("tumor_z").cast(pl.Float32),
                pl.col("healthy_z").cast(pl.Float32),
                "copy_number",
            ).with_columns(dom.alias("nmf"), pl.Series("w", wd)),
        )
        # Stacked loadings: each cCRE's top four programs, largest at the bottom.
        Wsub = W[j0:j1]
        k = 4
        order = np.argsort(-Wsub, axis=1)[:, :k]
        vals = np.take_along_axis(Wsub, order, axis=1)
        tops = np.cumsum(vals, axis=1)
        rows = np.repeat(np.arange(j1 - j0), k)
        keep = vals.ravel() > 0
        rows, ordk = rows[keep], order.ravel()[keep]
        push(
            "nmf_stack",
            pl.DataFrame(
                {
                    "cCRE": sl["cCRE"].gather(rows),
                    "chrom": sl["chrom"].gather(rows),
                    "start": sl["start"].gather(rows).cast(pl.Int32),
                    "end": sl["end"].gather(rows).cast(pl.Int32),
                    "nmf": np.asarray(PROGRAMS, dtype=object)[ordk],
                    "weight": vals.ravel()[keep],
                    "y0": (tops - vals).ravel()[keep],
                    "y1": tops.ravel()[keep],
                    "rank": np.tile(np.arange(1, k + 1), j1 - j0)[keep].astype(np.int8),
                }
            ),
        )
        a, b = float(LIN[j0]) if j0 < len(LIN) else lo, float(LIN[j1 - 1]) if j1 > 0 else hi
        links = STATE.get("links_all")
        if links is not None and links.height:
            k0, k1 = np.searchsorted(STATE["links_lin"], [a - 1e6, b + 1e6])
            push("links", links.slice(int(k0), int(k1 - k0)).drop("lin"))
        g0, g1 = np.searchsorted(_genes_lin, [lo - pad - 2_000_000, hi + pad])
        push("genes", _genes.slice(int(g0), int(g1 - g0)).drop("lin", "length"))
        STATE["focus_loaded"] = (a, b)

    # ---- Cohort ATAC heatmap: every selected tumor + matched normals, per cCRE, for the visible window ----
    _stream_msg = {"v": None}

    def _notify(msg: str) -> None:
        # Only wake marimo (the status line) when the message actually changes.
        if msg != _stream_msg["v"]:
            _stream_msg["v"] = msg
            set_stream(msg)

    def stream_cohort(lo: float, hi: float, force: bool = False) -> None:
        rows = STATE.get("atac_cols") or []
        i0, i1 = (int(v) for v in np.searchsorted(LIN, [lo, hi]))
        if i1 - i0 > COHORT_MAX or not rows:
            if STATE.get("cohort_key") is not None:
                push("cohort_atac", STATE["empty"]["cohort_atac"])
                STATE["cohort_key"] = None
            _notify(
                f"Cohort ATAC heatmap: zoom to ≤ {COHORT_MAX:,} cCREs to stream ({i1 - i0:,} in view)."
                if rows
                else "Cohort ATAC heatmap: pick at least one cohort."
            )
            return
        got = STATE.get("cohort_key")
        if not force and got is not None and i0 >= got[0] and i1 <= got[1]:
            return
        pad = (i1 - i0) // 2
        j0, j1 = max(0, i0 - pad), min(len(LIN), i1 + pad)
        _notify(f"Cohort ATAC heatmap: streaming {j1 - j0:,} cCREs × {len(rows)} samples…")
        b0, b1 = j0 // BLOCK, (j1 - 1) // BLOCK
        cols = [c for _, c, _, _ in rows]
        block = pl.concat([read_block(b).select(cols) for b in range(b0, b1 + 1)])
        block = block.slice(j0 - b0 * BLOCK, j1 - j0)
        mat = block.to_numpy().T  # samples x cCREs
        r_idx, c_idx = np.nonzero(mat > -9.99)
        sl = BASE.slice(j0, j1 - j0)
        labels = np.asarray([lab for lab, _, _, _ in rows], dtype=object)
        bios = np.asarray([b for _, _, _, b in rows], dtype=object)
        push(
            "cohort_atac",
            pl.DataFrame(
                {
                    "chrom": sl["chrom"].gather(c_idx),
                    "start": sl["start"].gather(c_idx).cast(pl.Int32),
                    "end": sl["end"].gather(c_idx).cast(pl.Int32),
                    "cCRE": sl["cCRE"].gather(c_idx),
                    "label": labels[r_idx],
                    "biosample": bios[r_idx],
                    "z": mat[r_idx, c_idx].astype(np.float32),
                }
            ),
        )
        STATE["cohort_key"] = (j0, j1)
        _notify(f"Cohort ATAC heatmap: {j1 - j0:,} cCREs × {len(rows)} samples streamed.")

    STATE["empty"] = {
        "ccres": pl.DataFrame(schema={"cCRE": pl.String, "chrom": pl.String, "start": pl.Int32, "end": pl.Int32, "mid": pl.Int32, "class": pl.String, "tumor_z": pl.Float32, "healthy_z": pl.Float32, "copy_number": pl.String, "nmf": pl.String, "w": pl.Float32}),
        "nmf_stack": pl.DataFrame(schema={"cCRE": pl.String, "chrom": pl.String, "start": pl.Int32, "end": pl.Int32, "nmf": pl.String, "weight": pl.Float32, "y0": pl.Float32, "y1": pl.Float32, "rank": pl.Int8}),
        "links": pl.DataFrame(schema={"chrom": pl.String, "start": pl.Int32, "end": pl.Int32, "mid": pl.Int32, "tss": pl.Int32, "gene": pl.String, "score": pl.Float32, "cell": pl.String, "source": pl.String}),
        "cohort_atac": pl.DataFrame(schema={"chrom": pl.String, "start": pl.Int32, "end": pl.Int32, "cCRE": pl.String, "label": pl.String, "biosample": pl.String, "z": pl.Float32}),
    }

    def current_view() -> tuple[float, float]:
        return STATE.get("view") or STATE["initial_view"]

    _c0 = BROWSER_SPEC["vconcat"][1]["vconcat"][0]["encoding"]["x"]["scale"]["domain"]["initial"]
    STATE["initial_view"] = (
        float(cnv.CHROM_OFFSET[_c0[0]["chrom"]] + _c0[0]["pos"]),
        float(cnv.CHROM_OFFSET[_c0[1]["chrom"]] + _c0[1]["pos"]),
    )

    def interval_of(v):
        # Interval selections arrive as {"type": "interval", "intervals": {"x": [lo, hi]}}.
        if isinstance(v, dict):
            v = (v.get("intervals") or {}).get("x")
        if isinstance(v, (list, tuple)) and len(v) == 2 and v[0] is not None:
            a, b = v
            if isinstance(a, dict):
                a = cnv.CHROM_OFFSET[a["chrom"]] + a["pos"]
                b = cnv.CHROM_OFFSET[b["chrom"]] + b["pos"]
            return float(min(a, b)), float(max(a, b))
        return None

    def go_to(chrom: str, start: int, end: int) -> None:
        off = cnv.CHROM_OFFSET[chrom]
        set_params(nav={"type": "interval", "intervals": {"x": [off + int(start), off + int(end)]}})

    _last = {"sel": None, "nav": None}

    def _on_params(change):
        vals = dict(change.get("new") or {})
        if "nav" in vals:
            iv = interval_of(vals["nav"]) or (0.0, float(cnv.GENOME_LEN))
            if iv != _last["nav"]:
                _last["nav"] = iv
                STATE["view"] = iv
                stream_focus(*iv)
                stream_cohort(*iv)
        if "sel" in vals:
            iv = interval_of(vals["sel"])
            if iv and iv != _last["sel"]:
                _last["sel"] = iv
                set_sel((iv[0], iv[1], "brushed interval"))

    def _on_click(_change):
        d = dict(browser.clicked_datum or {})
        if d.get("biosample") and "label" in d and "tumor_z" not in d:
            set_focus(d["biosample"])  # a row of a cohort heatmap
            return
        if not d.get("chrom") or d.get("start") is None:
            return
        off = cnv.CHROM_OFFSET.get(d["chrom"])
        if off is None:
            return
        lo = off + int(d.get("chromStart", d["start"]))
        hi = off + int(d.get("chromEnd", d.get("end", d["start"])))
        if "source" in d:
            set_sel((float(lo), float(hi), f"clicked {d['source']} element → {d.get('gene')}"))
            return
        kind = "cCRE" if "class" in d else "CN segment" if "n_ccres" in d and "tumor_z" not in d else "genome bin" if "n_ccres" in d else "region"
        set_sel((float(lo), float(hi), f"clicked {kind}"))

    browser.observe(_on_params, names=["parameter_values"])
    browser.observe(_on_click, names=["click_revision"])
    return (
        browser,
        current_view,
        go_to,
        push,
        set_params,
        stream_cohort,
        stream_focus,
    )


@app.cell
def _(MODELS, PRESETS, SAMPLES, mo):
    # ---- Controls that do not depend on the focus tumor ----
    model_dd = mo.ui.dropdown(options=list(MODELS), value="CN-aware adjusted", label="NMF model")
    _cohorts = sorted(SAMPLES["cohort"].unique().to_list())
    cohorts_ms = mo.ui.multiselect(options=_cohorts, value=["BRCA"], label="Cohort heatmaps")
    sort_dd = mo.ui.dropdown(
        options=["cancer · CN in view", "cancer · ATAC in view", "cancer · sample"],
        value="cancer · CN in view",
        label="Row order",
    )
    resort_btn = mo.ui.run_button(label="↻ re-sort at view", kind="neutral")
    preset_dd = mo.ui.dropdown(options=["—"] + list(PRESETS), value="—", label="Preset locus")
    goto_text = mo.ui.text(placeholder="ERBB2 · chr17 · chr8:126-130Mb", label="Go to", debounce=True)
    return cohorts_ms, goto_text, model_dd, preset_dd, resort_btn, sort_dd


@app.cell
def _(SAMPLES, get_focus, mo, pl, set_focus):
    # ---- Focus tumor: cancer -> patient -> aliquot. Clicking a heatmap row also sets it. ----
    _focus = get_focus()
    _row = SAMPLES.filter(pl.col("biosample") == _focus).row(0, named=True)
    _by_cancer = SAMPLES.filter(pl.col("cohort") == _row["cohort"])
    _patients = _by_cancer["patient_id"].unique(maintain_order=True).to_list()

    def _first_of(df):
        return df.row(0, named=True)["biosample"]

    focus_cancer_dd = mo.ui.dropdown(
        options=sorted(SAMPLES["cohort"].unique().to_list()),
        value=_row["cohort"],
        label="Focus cancer",
        searchable=True,
        on_change=lambda c: set_focus(_first_of(SAMPLES.filter(pl.col("cohort") == c))),
    )
    focus_patient_dd = mo.ui.dropdown(
        options={p.split("-", 1)[1]: p for p in _patients},
        value=_row["patient_id"].split("-", 1)[1],
        label=f"Patient ({len(_patients)})",
        searchable=True,
        on_change=lambda p: set_focus(_first_of(SAMPLES.filter(pl.col("patient_id") == p))),
    )
    _aliquots = _by_cancer.filter(pl.col("patient_id") == _row["patient_id"])["biosample"].to_list()
    focus_aliquot_dd = mo.ui.dropdown(
        options={a.split("-", 1)[1][13:]: a for a in _aliquots},
        value=_focus.split("-", 1)[1][13:],
        label=f"Aliquot ({len(_aliquots)})",
        on_change=lambda a: set_focus(a),
    )
    return focus_aliquot_dd, focus_cancer_dd, focus_patient_dd


@app.cell
def _(
    BASE,
    DETAIL,
    SAMPLES,
    cnv,
    get_focus,
    mo,
    np,
    pl,
    push,
    read_cn,
    read_z,
    set_params,
):
    # ---- Load the focus tumor genome-wide and push it into the running browser ----
    focus = get_focus()
    header = SAMPLES.filter(pl.col("biosample") == focus).row(0, named=True)
    healthy_col = DETAIL.healthy.get(focus)
    sample16 = header["sample16"]
    with mo.status.spinner(title=f"Loading {sample16} across 2.35M cCREs…"):
        frame = BASE.with_columns(
            read_z(DETAIL.columns[focus]).alias("tumor_z"),
            (read_z(healthy_col) if healthy_col else pl.lit(None, dtype=pl.Float32)).alias("healthy_z"),
        ).join(read_cn(focus), on="reference_index", how="left", maintain_order="left")
    _codes = frame["copy_number_code"].to_numpy()
    _mapped = _codes[_codes < 7]
    modal_cn = int(np.bincount(_mapped).argmax()) if _mapped.size else 2
    cn_segments = (
        frame.select("chrom", "start", "end", "copy_number_code", "copy_number")
        .with_columns(
            (
                (pl.col("copy_number_code") != pl.col("copy_number_code").shift(1))
                | (pl.col("chrom") != pl.col("chrom").shift(1))
            )
            .fill_null(True)
            .cum_sum()
            .alias("_run")
        )
        .group_by("_run", maintain_order=True)
        .agg(
            pl.col("chrom").first(),
            pl.col("start").min().cast(pl.Int32),
            pl.col("end").max().cast(pl.Int32),
            pl.col("copy_number_code").first(),
            pl.col("copy_number").first(),
            pl.len().alias("n_ccres"),
        )
        .drop("_run")
        .filter(pl.col("chrom").is_in(cnv.CHROMS))
    )
    push("segments", cn_segments.select("chrom", "start", "end", "copy_number", "n_ccres"))
    push("overview_cn", cn_segments.select("chrom", "start", "end", "copy_number"))
    set_params(focusLab=header["label"])
    return cn_segments, focus, frame, header, healthy_col, modal_cn, sample16


@app.cell
def _(MODELS, PROGRAMS, mo, model_dd, np, pl, read_w):
    # ---- NMF loadings for the chosen model ----
    model_file = MODELS[model_dd.value]
    with mo.status.spinner(title=f"Loading {model_dd.value} cCRE loadings (24 programs)…"):
        W = read_w(model_file)
    wdom = W.max(1)
    dom_label = (
        pl.Series("nmf", PROGRAMS, dtype=pl.String)
        .gather(W.argmax(1).astype(np.int64))
        .zip_with(pl.Series(wdom > 0), pl.Series("nmf", [None] * len(wdom), dtype=pl.String))
        .alias("nmf")
    )
    return W, dom_label, model_file, wdom


@app.cell
def _(
    NO_READ,
    STATE: dict,
    W,
    cn_segments,
    current_view,
    dom_label,
    frame,
    modal_cn,
    pl,
    push,
    read_genes,
    stream_focus,
    wdom,
):
    # ---- Genome-wide bins (drawn when zoomed out) and per-cCRE detail for the current window ----
    STATE.update(frame=frame, W=W, dom_label=dom_label, wdom=wdom)
    _ok = pl.col("tumor_z") > NO_READ
    overview_bins = (
        frame.select("x_bin", "chrom", "start", "end", "tumor_z", "healthy_z", "copy_number_code")
        .with_columns(dom_label, pl.Series("w", wdom))
        .filter(pl.col("x_bin").is_not_null())
        .group_by("x_bin", "chrom")
        .agg(
            pl.col("start").min().cast(pl.Int32),
            pl.col("end").max().cast(pl.Int32),
            pl.col("tumor_z").filter(_ok).mean().cast(pl.Float32).alias("tumor_z"),
            pl.col("healthy_z").filter(pl.col("healthy_z") > NO_READ).mean().cast(pl.Float32).alias("healthy_z"),
            pl.col("copy_number_code").filter(pl.col("copy_number_code") < 7).median().alias("_cn"),
            pl.col("nmf").drop_nulls().mode().first().alias("nmf"),
            pl.col("w").mean().cast(pl.Float32).alias("w"),
            pl.len().alias("n_ccres"),
        )
        .with_columns(
            pl.col("_cn")
            .round(0)
            .cast(pl.Int64)
            .replace_strict({0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6+"}, default="Unmapped")
            .alias("copy_number"),
            pl.col("nmf").fill_null("none"),
            pl.lit("none").alias("class"),
        )
        .drop("_cn")
        .sort("x_bin")
    )
    push("bins", overview_bins)
    stream_focus(*current_view(), force=True)

    # Focal events relative to this tumor's ploidy (navigation shortcuts).
    def _genes_in(chrom, start, end, k=3):
        g = (
            read_genes()
            .filter((pl.col("chrom") == chrom) & (pl.col("end") >= start) & (pl.col("start") <= end))
            .sort("score", descending=True)["symbol"]
            .unique(maintain_order=True)
            .head(k)
            .to_list()
        )
        return ", ".join(g) if g else "—"

    _seg = cn_segments.with_columns((pl.col("end") - pl.col("start")).alias("span")).filter(
        (pl.col("n_ccres") >= 50) & (pl.col("span") < 15_000_000) & (pl.col("copy_number_code") < 7)
    )
    _amps = _seg.filter(pl.col("copy_number_code") >= modal_cn + 2).sort(["copy_number_code", "n_ccres"], descending=[True, True]).head(6)
    _dels = _seg.filter((pl.col("copy_number_code") == 0) | (pl.col("copy_number_code") <= modal_cn - 2)).sort(
        ["copy_number_code", "n_ccres"], descending=[False, True]
    ).head(4)
    focal_events = {
        f"{'▲' if kind == 'gain' else '▼'} {r['chrom']}:{r['start'] / 1e6:.1f}–{r['end'] / 1e6:.1f}Mb CN {r['copy_number']} · "
        + _genes_in(r["chrom"], r["start"], r["end"]): (r["chrom"], int(r["start"]), int(r["end"]))
        for kind, tbl in (("gain", _amps), ("loss", _dels))
        for r in tbl.iter_rows(named=True)
    }
    return (focal_events,)


@app.cell
def _(
    STATE: dict,
    cnv,
    current_view,
    header,
    mo,
    pl,
    resolve_e2g,
    stream_focus,
):
    # ---- Enhancer -> gene links for the focus tumor's tissue (ENCODE-rE2G) ----
    _tissue = header.get("encode_tissue")
    with mo.status.spinner(title="Resolving enhancer–gene links for this tissue…"):
        try:
            e2g_meta, e2g_links = resolve_e2g(_tissue) if _tissue else (None, None)
        except Exception as _e:
            e2g_meta, e2g_links = {"error": str(_e)}, None
    _cols = ["chrom", "start", "end", "mid", "tss", "gene", "score", "cell", "source"]
    _parts = []
    if e2g_links is not None:
        _parts.append(e2g_links.with_columns(pl.lit(str(_tissue or "")).alias("cell"), pl.lit("ENCODE-rE2G").alias("source")))
    if _parts:
        _all = (
            pl.concat([p.select("chrom", "start", "end", "tss", "gene", pl.col("score").cast(pl.Float32), "cell", "source") for p in _parts])
            .with_columns(((pl.col("start") + pl.col("end")) // 2).alias("mid"))
            .with_columns(
                *[pl.col(c).cast(pl.Int32) for c in ("start", "end", "mid", "tss")],
                (pl.col("chrom").replace_strict(cnv.CHROM_OFFSET, return_dtype=pl.Int64) + pl.col("start")).alias("lin"),
            )
            .sort("lin")
            .select(_cols + ["lin"])
        )
        STATE["links_all"], STATE["links_lin"] = _all, _all["lin"].to_numpy()
    else:
        STATE["links_all"] = None
    stream_focus(*current_view(), force=True)
    return e2g_links, e2g_meta


@app.cell
def _(
    BLOCK,
    COHORT_CN,
    COHORT_MAX,
    DETAIL,
    LIN,
    SAMPLES,
    STATE: dict,
    cnv,
    cohorts_ms,
    current_view,
    mo,
    np,
    pl,
    read_block,
    resort_btn,
    set_params,
    sort_dd,
    stream_cohort,
):
    # ---- Heatmap rows: selected cohorts in the chosen order. Only parameters and data change in the browser. ----
    _ = resort_btn.value
    _lo, _hi = current_view()
    rows_df = SAMPLES.filter(pl.col("cohort").is_in(list(cohorts_ms.value or [])))
    _i0, _i1 = (int(v) for v in np.searchsorted(LIN, [_lo, _hi]))
    row_note = ""
    if sort_dd.value == "cancer · ATAC in view" and 0 < _i1 - _i0 <= COHORT_MAX:
        with mo.status.spinner(title="Reading cohort ATAC in view to order rows…"):
            _blk = pl.concat([read_block(b) for b in range(_i0 // BLOCK, (_i1 - 1) // BLOCK + 1)]).slice(
                _i0 - (_i0 // BLOCK) * BLOCK, _i1 - _i0
            )
        _m = {c: float(np.nanmean(np.where(v > -9.99, v, np.nan))) if np.any(v > -9.99) else -99.0
              for c, v in ((c, _blk[c].to_numpy()) for c in rows_df["column"].to_list())}
        rows_df = rows_df.with_columns(pl.col("column").replace_strict(_m, return_dtype=pl.Float64).alias("_k"))
        rows_df = rows_df.sort(["cohort", "_k"], descending=[False, True])
        row_note = "rows ordered by mean ATAC z in the view (highest first)"
    elif sort_dd.value in ("cancer · CN in view", "cancer · ATAC in view"):
        _off = pl.col("chrom").replace_strict(cnv.CHROM_OFFSET, return_dtype=pl.Int64)
        _seg = COHORT_CN.with_columns((_off + pl.col("start")).alias("_a"), (_off + pl.col("end")).alias("_b")).filter(
            (pl.col("_b") > _lo) & (pl.col("_a") < _hi) & (pl.col("code") < 7)
        )
        _score = (
            _seg.with_columns((pl.min_horizontal("_b", pl.lit(_hi)) - pl.max_horizontal("_a", pl.lit(_lo))).alias("_len"))
            .group_by("biosample")
            .agg(((pl.col("code").cast(pl.Float64) * pl.col("_len")).sum() / pl.col("_len").sum()).alias("_k"))
        )
        rows_df = rows_df.join(_score, on="biosample", how="left").with_columns(pl.col("_k").fill_null(-1.0))
        rows_df = rows_df.sort(["cohort", "_k"], descending=[False, True])
        row_note = "rows ordered by mean copy number in the view (highest first)"
        if sort_dd.value == "cancer · ATAC in view":
            row_note += f"; ATAC ordering needs ≤ {COHORT_MAX:,} cCREs in view"
    else:
        rows_df = rows_df.sort("cohort", "biosample")
        row_note = "rows ordered by sample"
    _atac = []
    for _cohort in rows_df["cohort"].unique(maintain_order=True).to_list():
        _sub = rows_df.filter(pl.col("cohort") == _cohort)
        for _h in sorted({DETAIL.healthy.get(b) for b in _sub["biosample"]} - {None}):
            _tissue = _sub.filter(pl.col("healthy_column") == _h)["encode_tissue"].first()
            _atac.append((f"normal {_cohort}: {_tissue or _h[2:]}", _h, "normal", ""))
        _atac += [(r["label"], r["column"], "tumor", r["biosample"]) for r in _sub.iter_rows(named=True)]
    STATE["atac_cols"] = _atac
    set_params(cohortRows=rows_df["label"].to_list(), atacRows=[a[0] for a in _atac])
    with mo.status.spinner(title="Streaming cohort ATAC for the current view…"):
        stream_cohort(_lo, _hi, force=True)
    return row_note, rows_df


@app.cell
def _(default_models, focus, mo, read_catv1_meta):
    # ---- CATv1 sequence models: defaults follow the focus tumor's matched normal ----
    _meta = read_catv1_meta()
    _opts = {
        f"{r['experiment_biosample_term_name']} · {r['assay_term_name']} · {r['experiment_accession']}": r["experiment_accession"]
        for r in _meta.iter_rows(named=True)
    }
    _inv = {v: k for k, v in _opts.items()}
    _p, _c = default_models(focus)
    primary_dd = mo.ui.dropdown(options=_opts, value=_inv.get(_p), label="Primary sequence model", searchable=True)
    compare_dd = mo.ui.dropdown(options=_opts, value=_inv.get(_c), label="Compare model", searchable=True)
    return compare_dd, primary_dd


@app.cell
def _(compare_dd, mo, primary_dd, resolve_model, set_params):
    # Swapping a model rewrites three URL parameters; GenomeSpy streams the new bigWig/bigBed in place.
    with mo.status.spinner(title="Resolving CATv1 → ENCODE ChromBPNet files…"):
        try:
            primary_model = resolve_model(primary_dd.value) if primary_dd.value else None
        except Exception as _e:
            primary_model = {"error": str(_e)}
        try:
            compare_model = resolve_model(compare_dd.value) if compare_dd.value else None
        except Exception as _e:
            compare_model = {"error": str(_e)}
    _pm, _cm = primary_model or {}, compare_model or {}
    _upd = {k: v for k, v in (("predUrl", _pm.get("pred")), ("contribUrl", _pm.get("contrib")), ("motifUrl", _pm.get("motifs")), ("cmpUrl", _cm.get("contrib"))) if v}
    if _upd:
        set_params(**_upd)
    return compare_model, primary_model


@app.cell
def _(focal_events, mo):
    events_dd = mo.ui.dropdown(options={"—": None} | focal_events, value="—", label="Focus CN events")
    return (events_dd,)


@app.cell
def _(PRESETS, go_to, preset_dd):
    if preset_dd.value in PRESETS:
        go_to(*PRESETS[preset_dd.value])
    return


@app.cell
def _(events_dd, go_to):
    if events_dd.value:
        _c, _s, _e = events_dd.value
        _pad = max(250_000, (_e - _s) // 2)
        go_to(_c, max(0, _s - _pad), _e + _pad)
    return


@app.cell
def _(cnv, go_to, goto_text, html, pl, re, read_genes):
    # ---- Typed navigation: RefSeq gene, chromosome, or interval ----
    def _resolve(raw: str):
        q = raw.replace(",", "").replace(" ", "")
        m = re.fullmatch(r"chr([0-9]{1,2}|X|Y)(?::([\d.]+)(Mb|kb)?-([\d.]+)(Mb|kb)?)?", q, flags=re.I)
        if m:
            c = "chr" + m.group(1).upper()
            if c in cnv.CHROM_SIZES:
                if m.group(2) is None:
                    return c, 0, cnv.CHROM_SIZES[c]
                mult = {"mb": 1_000_000, "kb": 1_000}
                unit = (m.group(5) or m.group(3) or "").lower()
                f = mult.get(unit, 1)
                s, e = int(float(m.group(2)) * f), int(float(m.group(4)) * f)
                if 0 <= s < e <= cnv.CHROM_SIZES[c]:
                    return c, s, e
        hit = read_genes().filter(pl.col("symbol").str.to_uppercase() == q.upper()).sort("score", descending=True)
        if hit.height:
            g = hit.row(0, named=True)
            pad = max(50_000, int(g["length"]))
            return g["chrom"], max(0, g["start"] - pad), min(cnv.CHROM_SIZES[g["chrom"]], g["end"] + pad)
        return None

    goto_warning = None
    _raw = (goto_text.value or "").strip()
    if _raw:
        _hit = _resolve(_raw)
        if _hit:
            go_to(*_hit)
        else:
            goto_warning = f"“{html.escape(_raw[:60])}” is not a RefSeq gene or hg38 interval."
    return (goto_warning,)


@app.cell
def _(
    cohorts_ms,
    compare_dd,
    compare_model,
    e2g_links,
    e2g_meta,
    events_dd,
    focus_aliquot_dd,
    focus_cancer_dd,
    focus_patient_dd,
    get_stream,
    goto_text,
    goto_warning,
    header,
    healthy_col,
    mo,
    modal_cn,
    model_dd,
    preset_dd,
    primary_dd,
    primary_model,
    resort_btn,
    row_note,
    rows_df,
    sample16,
    sort_dd,
):
    def _model_txt(m, tag):
        m = m or {}
        if m.get("annotation"):
            return (
                f"{tag}: <a href='{m['catv1']}' target='_blank'>{m['biosample']} ({m['experiment']})</a> → "
                f"<a href='https://www.encodeproject.org/annotations/{m['annotation']}/' target='_blank'>ChromBPNet</a>"
            )
        return f"{tag}: unavailable" + (f" ({m['error'][:50]})" if m.get("error") else "")

    _e2g = (
        f"<a href='https://www.encodeproject.org/annotations/{e2g_meta['annotation']}/' target='_blank'>ENCODE-rE2G</a> "
        f"{header['encode_tissue']} ({len(e2g_links):,} links)"
        if isinstance(e2g_meta, dict) and e2g_meta.get("annotation")
        else "no ENCODE-rE2G links for this tissue"
    )
    _ref = f"{header['encode_tissue']} ({healthy_col[2:]})" if healthy_col else "no matched ENCODE normal"
    _n = rows_df.height
    mo.vstack(
        [
            mo.md(
                "### Epifolio Genome Workbench "
                "<span style='font-size:0.85rem;font-weight:400;opacity:0.75'>"
                f"— focus <b>{sample16}</b> · {header['sex']}, {header['age']} y · {header['diagnosis']} · "
                f"normal: {_ref} · ploidy {modal_cn} · heatmaps: {_n} tumors in {', '.join(cohorts_ms.value) or 'no cohort'}</span>"
                f"<br><span style='font-size:0.74rem;opacity:0.65'>{_model_txt(primary_model, 'Primary')} · "
                f"{_model_txt(compare_model, 'Compare')} · {_e2g} · {row_note} · {get_stream()}</span>"
            ),
            mo.hstack(
                [focus_cancer_dd, focus_patient_dd, focus_aliquot_dd, model_dd, events_dd, preset_dd, goto_text],
                justify="start",
                gap=0.8,
                wrap=True,
            ),
            mo.hstack([cohorts_ms, sort_dd, resort_btn, primary_dd, compare_dd], justify="start", gap=0.8, wrap=True, align="end"),
        ]
        + ([mo.callout(mo.md(goto_warning), kind="warn")] if goto_warning else []),
        gap=0.3,
    )
    return


@app.cell
def _(browser):
    # The browser is displayed by a cell with no other inputs, so nothing re-renders (re-mounts) it.
    browser
    return


@app.cell
def _(CCRE_CLASSES, CN_COLORS, CN_LEVELS, NMF_COLORS, PROGRAMS, mo):
    def _sw(c, t):
        return (
            "<span style='display:inline-flex;align-items:center;gap:2px;margin-right:6px'>"
            f"<span style='width:9px;height:9px;background:{c};display:inline-block;border-radius:2px'></span>{t}</span>"
        )

    mo.Html(
        "<div style='font-size:0.72rem;opacity:.85;line-height:1.5'>"
        + "<b>ASCAT CN</b> "
        + "".join(_sw(c, t) for t, c in zip(CN_LEVELS, CN_COLORS))
        + " &nbsp;<b>NMF program</b> "
        + "".join(_sw(c, t.removeprefix("NMF")) for t, c in zip(PROGRAMS, NMF_COLORS))
        + " &nbsp;<b>ATAC z</b> " + _sw("#2563eb", "low") + _sw("#f8fafc", "0") + _sw("#dc2626", "high")
        + " &nbsp;<b>cCRE</b> " + "".join(_sw(c, t) for t, c in CCRE_CLASSES.items())
        + " &nbsp;<b>Links</b> " + _sw("#2563eb", "ENCODE-rE2G (opacity = score)")
        + "<br><span style='opacity:.7'>scroll = zoom · drag = pan · drag on the top genome strip = jump · "
        "<b>shift+drag</b> on the tumor track or click = select cCREs · click a heatmap row = focus that tumor · "
        "letters appear below ~300 bp · display options are under the browser</span></div>"
    )
    return


@app.cell
def _(mo):
    # Tab strip only; each panel is its own cell, so switching tabs never touches the browser.
    sections = mo.ui.tabs({k: "" for k in ("Selection", "Cohort at selection", "Dosage", "Programs", "Cohort", "About")})
    sections
    return (sections,)


@app.cell
def _(BLOCK, DETAIL, LIN, TABLE_MAX, np, pl, read_block, rows_df):
    # Shared helper: selected cohorts' ATAC z for detail rows [i0, i1), as samples x cCREs.
    def cohort_matrix(i0: int, i1: int):
        cols = rows_df["column"].to_list()
        if i1 <= i0 or not cols:
            return cols, np.zeros((0, 0), dtype=np.float32), []
        blk = pl.concat([read_block(b).select(cols + sorted({h for h in DETAIL.healthy.values() if h})) for b in range(i0 // BLOCK, (i1 - 1) // BLOCK + 1)])
        blk = blk.slice(i0 - (i0 // BLOCK) * BLOCK, i1 - i0)
        normals = sorted({DETAIL.healthy.get(b) for b in rows_df["biosample"]} - {None})
        return cols, blk.select(cols).to_numpy().T, [(h, blk[h].to_numpy()) for h in normals]

    def sel_rows(sel):
        lo, hi = sel[0], sel[1]
        i0, i1 = (int(x) for x in np.searchsorted(LIN, [lo, hi + 1]))
        return i0, min(i1, i0 + TABLE_MAX), i1 - i0

    return cohort_matrix, sel_rows


@app.cell
def _(
    PROGRAMS,
    W,
    cnv,
    cohort_matrix,
    dom_label,
    e2g_links,
    frame,
    get_sel,
    mo,
    np,
    pl,
    sections,
    sel_rows,
    wdom,
):
    # ---- Selection: every cCRE in the brushed / clicked interval, focus tumor + cohort context ----
    _out = None
    _sel = get_sel()
    if sections.value == "Selection":
        if not _sel:
            _out = mo.md(
                "<span style='opacity:.6;font-size:.85rem'>Shift+drag on the focus-tumor track, or click a cCRE, "
                "a CN segment, a genome bin or an enhancer link, to list its cCREs here.</span>"
            )
        else:
            _lo, _hi, _how = _sel
            _i0, _j1, _n = sel_rows(_sel)
            _rows = frame.slice(_i0, _j1 - _i0)
            _, _mat, _normals = cohort_matrix(_i0, _j1)
            _valid = np.where(_mat > -9.99, _mat, np.nan) if _mat.size else _mat

            def _nanmean(a, axis):
                with np.errstate(all="ignore"):
                    return np.nanmean(a, axis=axis) if a.size else np.full(_j1 - _i0, np.nan)

            _cohort_mean = _nanmean(_valid, 0) if _mat.size else np.full(_j1 - _i0, np.nan)
            _normal_mean = (
                _nanmean(np.vstack([np.where(v > -9.99, v, np.nan) for _, v in _normals]), 0)
                if _normals
                else np.full(_j1 - _i0, np.nan)
            )
            _tbl = _rows.select(
                "cCRE",
                "chrom",
                "start",
                "end",
                "class",
                pl.col("copy_number").alias("ASCAT CN"),
                pl.col("tumor_z").cast(pl.Float64).round(2).alias("tumor z"),
                pl.col("healthy_z").cast(pl.Float64).round(2).alias("normal z"),
                (pl.col("tumor_z") - pl.col("healthy_z")).cast(pl.Float64).round(2).alias("Δz"),
            ).with_columns(
                pl.Series("cohort mean z", np.round(_cohort_mean.astype(np.float64), 2)),
                pl.Series("cohort normals z", np.round(_normal_mean.astype(np.float64), 2)),
                dom_label.slice(_i0, _j1 - _i0).fill_null("—").alias("dominant"),
                pl.Series("W dominant", np.round(wdom[_i0:_j1].astype(np.float64), 3)),
                *[pl.Series(p, np.round(W[_i0:_j1, k].astype(np.float64), 3)) for k, p in enumerate(PROGRAMS)],
            ).with_row_index("_r")

            def _overlap(src, label):
                if src is None or _rows.height == 0:
                    return None
                sub = src.filter(pl.col("chrom") == _rows["chrom"][0]).filter(
                    (pl.col("end") > _rows["start"].min()) & (pl.col("start") < _rows["end"].max())
                ).rename({"chrom": "c2", "start": "s2", "end": "e2"})
                if sub.height == 0:
                    return None
                return (
                    _tbl.select("_r", "chrom", "start", "end")
                    .join_where(sub, pl.col("chrom") == pl.col("c2"), pl.col("start") < pl.col("e2"), pl.col("end") > pl.col("s2"))
                    .sort("_sort", descending=True)
                    .group_by("_r", maintain_order=True)
                    .agg(pl.col("_label").unique(maintain_order=True).head(6).str.join(", ").alias(label))
                )

            _extras = [
                _overlap(
                    e2g_links.with_columns((pl.col("gene") + " " + pl.col("score").round(2).cast(pl.String)).alias("_label"), pl.col("score").alias("_sort"))
                    if e2g_links is not None
                    else None,
                    "rE2G targets",
                ),
            ]
            _front = []
            for _x in _extras:
                if _x is not None:
                    _col = [c for c in _x.columns if c != "_r"][0]
                    _tbl = _tbl.join(_x, on="_r", how="left").with_columns(pl.col(_col).fill_null("—"))
                    _front.append(_col)
            _tbl = _tbl.select(
                ["cCRE", "chrom", "start", "end", "class", "ASCAT CN", "tumor z", "normal z", "Δz", "cohort mean z", "cohort normals z", "dominant", "W dominant"]
                + _front
                + PROGRAMS
            )

            def _pos(x):
                c = next(c for c in reversed(cnv.CHROMS) if x >= cnv.CHROM_OFFSET[c])
                return f"{c}:{int(x - cnv.CHROM_OFFSET[c]):,}"

            _cap = f" (first {_j1 - _i0:,} shown)" if _n > _j1 - _i0 else ""
            _out = mo.vstack(
                [
                    mo.md(f"**{_n:,} cCREs** in {_how} {_pos(_lo)} – {_pos(_hi)}{_cap}").style({"font-size": "0.82rem"}),
                    mo.ui.table(
                        _tbl.to_pandas(),
                        page_size=50,
                        selection=None,
                        show_column_summaries=False,
                        show_data_types=False,
                        freeze_columns_left=["cCRE"],
                        max_height=320,
                    ),
                ],
                gap=0.2,
            )
    _out
    return


@app.cell
def _(
    CN_LEVELS,
    COHORT_CN,
    cnv,
    cohort_matrix,
    frame,
    get_sel,
    go,
    mo,
    np,
    pl,
    rows_df,
    sections,
    sel_rows,
):
    # ---- Cohort at selection: dosage and accessibility of the selected cCREs across every chosen tumor ----
    _out = None
    _sel = get_sel()
    if sections.value == "Cohort at selection":
        if not _sel:
            _out = mo.md("<span style='opacity:.6;font-size:.85rem'>Select an interval in the browser first.</span>")
        else:
            _i0, _j1, _n = sel_rows(_sel)
            _cols, _mat, _ = cohort_matrix(_i0, _j1)
            _rows = frame.slice(_i0, _j1 - _i0)
            _chrom = _rows["chrom"][0] if _rows.height else None
            # Copy number of each tumor at each selected cCRE, from the cohort's segments.
            _starts = _rows["start"].to_numpy()
            _cn = np.full(_mat.shape, 7, dtype=np.int8)
            _segs = COHORT_CN.filter(
                (pl.col("chrom") == _chrom) & (pl.col("end") >= int(_starts.min(initial=0))) & (pl.col("start") <= int(_rows["end"].max() or 0))
            )
            _bidx = {b: i for i, b in enumerate(rows_df["biosample"].to_list())}
            for _r in _segs.iter_rows(named=True):
                _k = _bidx.get(_r["biosample"])
                if _k is not None:
                    _m = (_starts >= _r["start"]) & (_starts <= _r["end"])
                    _cn[_k, _m] = _r["code"]
            _long = pl.DataFrame(
                {
                    "cohort": np.repeat(rows_df["cohort"].to_numpy(), _mat.shape[1]),
                    "cn": _cn.ravel(),
                    "z": _mat.ravel(),
                }
            ).filter((pl.col("z") > -9.99) & (pl.col("cn") < 7))
            _agg = _long.group_by("cohort", "cn").agg(pl.col("z").median().alias("med"), pl.len().alias("n"))
            _cohorts = sorted(_agg["cohort"].unique().to_list())
            _levels = list(range(7))
            _z = np.full((len(_cohorts), 7), np.nan)
            _txt = [["" for _ in _levels] for _ in _cohorts]
            for _r in _agg.iter_rows(named=True):
                _a, _b = _cohorts.index(_r["cohort"]), int(_r["cn"])
                _z[_a, _b] = _r["med"]
                _txt[_a][_b] = f"{_r['med']:.2f}<br><sub>n={_r['n']:,}</sub>"
            fig_dose = go.Figure(
                go.Heatmap(
                    z=_z,
                    x=CN_LEVELS[:7],
                    y=_cohorts,
                    text=_txt,
                    texttemplate="%{text}",
                    colorscale=[[0, "#2563eb"], [0.5, "#f8fafc"], [1, "#dc2626"]],
                    zmid=0,
                    colorbar=dict(title="median z"),
                    hovertemplate="%{y} · CN %{x}: median z %{z:.2f}<extra></extra>",
                )
            )
            fig_dose.update_layout(
                template="plotly_white",
                height=120 + 42 * len(_cohorts),
                margin=dict(l=70, r=10, t=60, b=40),
                title=dict(
                    text="Selected cCREs: ATAC z by each tumor's own copy number<br><sub>"
                    + (f"first {_j1 - _i0:,} of {_n:,}" if _n > _j1 - _i0 else f"{_n:,}")
                    + f" cCREs × {len(_cols)} tumors · sentinels and Unmapped excluded</sub>",
                    font_size=14,
                ),
                xaxis_title="ASCAT copy number at the cCRE",
            )
            # Per-tumor mean over the selection, highest first, coloured by CN at the selection.
            with np.errstate(all="ignore"):
                _per = np.nanmean(np.where(_mat > -9.99, _mat, np.nan), axis=1)
                _per_cn = [int(np.bincount(r[r < 7]).argmax()) if np.any(r < 7) else 7 for r in _cn]
            _order = np.argsort(-np.nan_to_num(_per, nan=-99))
            fig_rank = go.Figure(
                go.Bar(
                    x=np.arange(len(_order)),
                    y=_per[_order],
                    marker_color=[cnv.cn_colors(CN_LEVELS).get(CN_LEVELS[_per_cn[i]], "#9e9e9e") for i in _order],
                    customdata=np.stack([rows_df["label"].to_numpy()[_order], np.asarray(CN_LEVELS)[np.asarray(_per_cn)[_order]]], axis=1),
                    hovertemplate="%{customdata[0]}: mean z %{y:.2f} · modal CN %{customdata[1]}<extra></extra>",
                )
            )
            fig_rank.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=55, r=10, t=50, b=30),
                title=dict(text="Tumors ranked by mean ATAC z over the selection<br><sub>bar colour = modal ASCAT CN at the selection</sub>", font_size=14),
                xaxis=dict(showticklabels=False, title="tumors"),
                yaxis_title="mean z",
            )
            _out = mo.hstack([fig_dose, fig_rank], widths=[0.45, 0.55], gap=0.5)
    _out
    return


@app.cell
def _(
    CN_COLORS,
    CN_LEVELS,
    NMF_COLORS,
    NO_READ,
    PROGRAMS,
    cnv,
    dom_label,
    frame,
    go,
    healthy_col,
    mo,
    modal_cn,
    np,
    pl,
    sections,
):
    # ---- Dosage: is the focus tumor's accessibility explained by its own copy number? ----
    _out = None
    if sections.value == "Dosage":
        _ok = frame.with_columns(dom_label).filter(
            (pl.col("tumor_z") > NO_READ)
            & (pl.col("copy_number_code") < 7)
            & ((pl.col("healthy_z") > NO_READ) if healthy_col else pl.lit(True))
        ).with_columns(((pl.col("tumor_z") - pl.col("healthy_z")) if healthy_col else pl.col("tumor_z")).alias("m"))
        _ylab = "Tumor − normal z" if healthy_col else "Tumor z (no normal reference)"
        _q = (
            _ok.group_by("copy_number", "copy_number_code")
            .agg(
                pl.len().alias("n"),
                pl.col("m").quantile(0.05).alias("lo"),
                pl.col("m").quantile(0.25).alias("q1"),
                pl.col("m").median().alias("med"),
                pl.col("m").quantile(0.75).alias("q3"),
                pl.col("m").quantile(0.95).alias("hi"),
            )
            .sort("copy_number_code")
        )
        _colors = dict(zip(CN_LEVELS, CN_COLORS))
        fig_box = go.Figure()
        for _r in _q.iter_rows(named=True):
            fig_box.add_trace(
                go.Box(
                    x=[f"{_r['copy_number']}<br><sub>{_r['n']:,}</sub>"],
                    q1=[_r["q1"]], median=[_r["med"]], q3=[_r["q3"]], lowerfence=[_r["lo"]], upperfence=[_r["hi"]],
                    marker_color=_colors.get(_r["copy_number"], "#64748b"),
                    name=_r["copy_number"],
                    showlegend=False,
                )
            )
        fig_box.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        fig_box.update_layout(
            template="plotly_white", height=420, margin=dict(l=55, r=10, t=50, b=50),
            title=dict(text="Accessibility by copy-number state<br><sub>all cCREs in the focus tumor · whiskers 5–95%</sub>", font_size=14),
            xaxis_title="ASCAT copy number (n cCREs)", yaxis_title=_ylab,
        )
        _gain = pl.col("copy_number_code") >= modal_cn + 2
        _prog = (
            _ok.filter(pl.col("nmf").is_not_null())
            .group_by("nmf")
            .agg(pl.col("m").filter(_gain).median().alias("gain"), pl.col("m").filter(~_gain).median().alias("rest"), _gain.mean().alias("frac_gain"))
            .with_columns(pl.col("nmf").str.slice(3).cast(pl.Int32).alias("_k"))
            .sort("_k")
        )
        _pc = dict(zip(PROGRAMS, NMF_COLORS))
        fig_prog = go.Figure(
            go.Scatter(
                x=_prog["rest"], y=_prog["gain"], mode="markers+text",
                text=[p.removeprefix("NMF") for p in _prog["nmf"]], textposition="top center", textfont_size=9,
                marker=dict(size=(6 + 30 * _prog["frac_gain"].fill_null(0)).to_list(), color=[_pc[p] for p in _prog["nmf"]], line=dict(width=0.5, color="#334155")),
                customdata=np.stack([_prog["nmf"].to_numpy(), _prog["frac_gain"].to_numpy()], axis=1),
                hovertemplate="%{customdata[0]}: gained %{y:.2f} vs other %{x:.2f}<br>%{customdata[1]:.0%} of its cCREs on gains<extra></extra>",
            )
        )
        _lim = [
            float(min(_prog["rest"].min() or 0, _prog["gain"].min() or 0)) - 0.2,
            float(max(_prog["rest"].max() or 0, _prog["gain"].max() or 0)) + 0.2,
        ]
        fig_prog.add_shape(type="line", x0=_lim[0], y0=_lim[0], x1=_lim[1], y1=_lim[1], line=dict(dash="dot", color="#94a3b8"))
        fig_prog.update_layout(
            template="plotly_white", height=420, margin=dict(l=55, r=10, t=50, b=50),
            title=dict(text=f"Programs on gained vs other segments<br><sub>median {_ylab.lower()} · gains ≥ CN {modal_cn + 2} · size = share on gains</sub>", font_size=14),
            xaxis_title="Other segments", yaxis_title="Gained segments",
        )
        _chr = (
            _ok.filter(pl.col("chrom").is_in(cnv.CHROMS))
            .group_by("chrom")
            .agg(pl.col("copy_number_code").clip(0, 6).mean().alias("cn"), pl.col("m").median().alias("med"))
        )
        _r = float(np.corrcoef(_chr["cn"], _chr["med"])[0, 1]) if _chr.height > 2 else float("nan")
        fig_chr = go.Figure(
            go.Scatter(
                x=_chr["cn"], y=_chr["med"], mode="markers+text",
                text=[c.removeprefix("chr") for c in _chr["chrom"]], textposition="top center", textfont_size=9,
                marker=dict(size=9, color="#334155"),
                hovertemplate="%{text}: mean CN %{x:.2f}, median %{y:.2f}<extra></extra>",
            )
        )
        fig_chr.update_layout(
            template="plotly_white", height=420, margin=dict(l=55, r=10, t=50, b=50),
            title=dict(text=f"Chromosome dosage vs accessibility<br><sub>Pearson r = {_r:.2f} across chromosomes</sub>", font_size=14),
            xaxis_title="Mean copy number", yaxis_title=f"Median {_ylab.lower()}",
        )
        _out = mo.vstack(
            [
                mo.md(
                    f"Focus tumor only. Metric = **{_ylab}**; no-read cCREs (z = −10) and Unmapped CN excluded; "
                    f"gains judged against this tumor's ploidy (modal CN {modal_cn})."
                ).style({"font-size": "0.85rem", "opacity": "0.8"}),
                mo.hstack([fig_box, fig_prog, fig_chr], widths="equal", gap=0.5),
            ],
            gap=0.3,
        )
    _out
    return


@app.cell
def _(
    NMF_COLORS,
    PROGRAMS,
    SAMPLES,
    go,
    header,
    mo,
    model_dd,
    model_file,
    pl,
    read_h,
    rows_df,
    sample16,
    sections,
):
    # ---- Programs: the focus tumor's NMF activity against its cohort, and every cohort's program mix ----
    _out = None
    if sections.value == "Programs":
        _h = read_h(model_file)
        _h = _h.with_columns(pl.sum_horizontal(PROGRAMS).alias("_t")).with_columns([(pl.col(c) / pl.col("_t")).alias(c) for c in PROGRAMS])
        _lab = SAMPLES.select("sample16", "cohort").unique("sample16")
        _hc = _h.join(_lab, left_on="sample", right_on="sample16", how="inner")
        _own = _hc.filter(pl.col("cohort") == header["cohort"])
        _focus = _h.filter(pl.col("sample") == sample16)
        fig_h = go.Figure()
        for _c, _col in zip(PROGRAMS, NMF_COLORS):
            fig_h.add_trace(go.Box(y=_own[_c].to_list(), name=_c, marker_color=_col, boxpoints=False, line_width=1, showlegend=False))
        if _focus.height:
            fig_h.add_trace(
                go.Scatter(
                    x=PROGRAMS, y=[float(_focus[c][0]) for c in PROGRAMS], mode="markers",
                    marker=dict(symbol="diamond", size=10, color="#dc2626", line=dict(width=1, color="white")), name=sample16,
                )
            )
        fig_h.update_layout(
            template="plotly_white", height=420, margin=dict(l=55, r=10, t=50, b=40),
            title=dict(text=f"{model_dd.value} program activity (H, row-normalised) · {sample16} vs {_own.height} {header['cohort']} tumors", font_size=14),
            yaxis_tickformat=".0%", yaxis_title="Share of sample activity",
            legend=dict(orientation="h", y=1.0, x=1, xanchor="right", yanchor="bottom"),
        )
        _mean = _hc.group_by("cohort").agg([pl.col(c).mean() for c in PROGRAMS]).sort("cohort")
        _hl = set(rows_df["cohort"].unique().to_list())
        fig_mix = go.Figure(
            go.Heatmap(
                z=_mean.select(PROGRAMS).to_numpy(),
                x=[p.removeprefix("NMF") for p in PROGRAMS],
                y=[f"<b>{c}</b>" if c in _hl else c for c in _mean["cohort"]],
                colorscale="Viridis",
                colorbar=dict(title="mean share", tickformat=".0%"),
                hovertemplate="%{y} · NMF%{x}: %{z:.1%}<extra></extra>",
            )
        )
        fig_mix.update_layout(
            template="plotly_white", height=420, margin=dict(l=70, r=10, t=50, b=40),
            title=dict(text="Mean program share per cohort<br><sub>bold = cohorts shown in the browser heatmaps</sub>", font_size=14),
            xaxis_title="NMF program",
        )
        _msg = None if _focus.height else mo.callout(mo.md(f"{sample16} is not part of the {model_dd.value} factorization."), kind="warn")
        _out = mo.vstack([x for x in (_msg, mo.hstack([fig_h, fig_mix], widths=[0.58, 0.42], gap=0.5)) if x is not None])
    _out
    return


@app.cell
def _(PROGRAMS, mo, pl, read_h, rows_df, sections):
    # ---- Cohort: the tumors currently shown as heatmap rows ----
    _out = None
    if sections.value == "Cohort":
        _h = read_h("adjusted").with_columns(
            pl.concat_list(PROGRAMS).list.arg_max().add(1).cast(pl.String).alias("_k"),
            (pl.max_horizontal(PROGRAMS) / pl.sum_horizontal(PROGRAMS)).alias("top_share"),
        ).select("sample", ("NMF" + pl.col("_k")).alias("top_adjusted_program"), "top_share")
        _tbl = (
            rows_df.join(_h, left_on="sample16", right_on="sample", how="left")
            .select(
                "label", "cohort", "sample16", "sex", "age", "diagnosis",
                pl.col("encode_tissue").alias("normal_reference"), "healthy_status",
                "top_adjusted_program", pl.col("top_share").round(3),
            )
        )
        _out = mo.vstack(
            [
                mo.md(f"**{_tbl.height} tumors** in the heatmaps, in row order.").style({"font-size": "0.85rem"}),
                mo.ui.table(_tbl.to_pandas(), page_size=15, selection=None, show_column_summaries=False),
            ],
            gap=0.3,
        )
    _out
    return


@app.cell
def _(COHORT_MAX, DETAIL_MAX, EMBEDDINGS, cnv, mo, sections):
    _out = None
    if sections.value == "About":
        _out = mo.hstack(
            [
                mo.md(
                    f"""
    **What this is.** One GenomeSpy view that joins the per-patient genome browser and the cohort locus
    browser. A *focus tumor* is drawn along its whole genome (ASCAT copy number, ATAC z, matched ENCODE
    normal, NMF loadings), and the *selected cohorts* are drawn as heatmaps on the same axis: ASCAT copy
    number for every tumor genome-wide, and per-cCRE ATAC z (with matched normals) once you zoom in.
    CATv1 sequence models (predicted accessibility, dynseq contribution letters, motif hits) and
    enhancer → gene links sit underneath.

    **Why it no longer restarts.** The view is created once per session. Controls never rebuild it:
    navigation writes the `nav` interval that drives the x domain; cohort choice and row order write the
    `cohortRows` / `atacRows` parameters that define the heatmap rows; switching CATv1 models rewrites URL
    parameters, so GenomeSpy streams the new bigWigs in place; the focus tumor, NMF model and visible
    window only swap datasets. Display options (colour mode, program filter, no-read cCREs, heatmap
    range) are GenomeSpy inputs under the view and never reach Python.

    **What streams when.** Genome-wide: the focus tumor's 5,000 bins and every tumor's CN segments.
    ≤ {DETAIL_MAX:,} cCREs in view: per-cCRE focus detail, stacked loadings, links and genes.
    ≤ {COHORT_MAX:,} cCREs: the cohort ATAC heatmap, read in 16k-row blocks of the detail parquet.
    Below ~300 bp: hg38 letters scaled by contribution score (browser-side FASTA + bigWig lookup).
    """
                ).style({"font-size": "0.85rem"}),
                mo.md(
                    f"""
    **Data** (all hosted or public, read on demand; nothing is written to disk)
    - Epifolio CNV × ATAC payload: `{cnv.REMOTE}` (detail z-scores, ASCAT membership, sample metadata)
    - NMF cCRE weights and sample activities: `{EMBEDDINGS}/{{adjusted,confounded,tcga.atac}}.nmf.{{ccre,sample}}.pq`
    - Cohort CN segments are computed in memory from `summary_ccre_membership.parquet` at start-up.
    - [CATv1](https://huggingface.co/programmable-genomics/CATv1) → ENCODE ChromBPNet bigWig/bigBed files (byte-range streamed)
    - ENCODE-rE2G links for the focus tumor's normal tissue
    - hg38 RefSeq genes, cytobands and indexed FASTA from genomespy.app

    **Caveats**
    - z = −10 marks cCREs with no reads, not repression; the heatmap leaves them blank.
    - CESC, ESCA, HNSC and SKCM have no matched ENCODE normal.
    - Program numbers are not aligned across the three NMF models.
    - The default compare model is a cancer cell line where one exists (MCF-7, HCT116, A549, HepG2, PC-3, HeLa-S3), else K562.
    """
                ).style({"font-size": "0.85rem"}),
            ],
            widths="equal",
            gap=2,
        )
    _out
    return


if __name__ == "__main__":
    app.run()
