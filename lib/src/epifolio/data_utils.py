"""Data loading and preparation for NMF visualizations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from epifolio import ASSETS
from epifolio.color_utils import load_color_map, resolve_cancer_colors

_CONFIG_KEYS = {
    "DEFAULT_CSV_FILENAME",
    "GRANDSCATTER_CSV_FILENAME",
    "HEATMAP_CSV_FILENAME",
    "UMAP_FILENAME",
    "ANALYSIS_METADATA_FILENAME",
    "JSON_FILENAME_CANCER_TYPE_COLORS",
    "JSON_FILENAME_COMPONENT_COLORS",
    "JSON_FILENAME_ORGAN_SYSTEM",
    "JSON_FILENAME_EMBRYONIC_LAYER",
    "JSON_FILENAME_VOCAB",
    "STRIP_METADATA_FILENAME",
}

# Raw NMF loading columns are named ``NMF1``..``NMFk``; the visualization layer
# expects row-normalized proportions named ``Comp_0``..``Comp_{k-1}``.
_RAW_NMF_RE = re.compile(r"(?i)^nmf[_\s-]*(\d+)$")

_SAMPLE_ID_CANDIDATES = [
    "sample_id",
    "Sample_ID",
    "Sample ID",
    "sample",
    "Unnamed: 0",
]


def _is_url(path: str | Path) -> bool:
    """True for remote sources (``http://``, ``https://``, ``s3://``, ...)."""
    return "://" in str(path)


def resolve_asset_path(
    path: str | Path,
    *,
    default_dir: str | None = None,
) -> str | Path:
    """Resolve a path against packaged epifolio assets when relative.

    Remote URLs are passed through unchanged so loaders can read them
    directly (pandas reads ``http(s)://`` sources for CSV and parquet).
    """
    if _is_url(path):
        return str(path)

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate

    candidates: list[Path] = []
    if default_dir:
        candidates.append(Path(ASSETS / default_dir / candidate))
    candidates.append(Path(ASSETS / candidate))
    candidates.append(candidate)

    for resolved in candidates:
        if resolved.exists():
            return resolved
    return candidates[0]


def _normalize_cfg_paths(cfg: dict) -> dict:
    normalized = dict(cfg)
    for key, value in list(normalized.items()):
        if key not in _CONFIG_KEYS or not isinstance(value, (str, Path)):
            continue
        if _is_url(value):
            continue

        default_dir = "conf" if str(value).endswith(".json") else "data"
        normalized[key] = resolve_asset_path(value, default_dir=default_dir)
    return normalized


def load_cfg(path: str | Path = "config.json") -> dict:
    """Read a JSON configuration file."""
    resolved = resolve_asset_path(path, default_dir="conf")
    with open(resolved) as f:
        return json.load(f)


def resolve_analysis_cfg(
    cfg_or_path: str | Path | dict,
    analysis_name: str | None = None,
) -> dict:
    """Merge a per-analysis overlay into the base config."""
    cfg = (
        load_cfg(cfg_or_path)
        if isinstance(cfg_or_path, (str, Path))
        else dict(cfg_or_path)
    )
    analyses = cfg.get("ANALYSES")
    if not analyses:
        return _normalize_cfg_paths(cfg)

    selected = analysis_name or cfg.get("DEFAULT_ANALYSIS") or next(iter(analyses))
    if selected not in analyses:
        raise ValueError(
            f"Unknown analysis '{selected}'. Available: {', '.join(analyses)}"
        )

    resolved = {key: value for key, value in cfg.items() if key != "ANALYSES"}
    resolved.update(analyses[selected])
    resolved["SELECTED_ANALYSIS"] = selected
    return _normalize_cfg_paths(resolved)


def get_available_analyses(cfg_path: str | Path = "conf/config.json") -> list[str]:
    cfg = load_cfg(cfg_path)
    analyses = cfg.get("ANALYSES")
    if not analyses:
        return [cfg.get("DEFAULT_ANALYSIS", "Default")]
    return list(analyses.keys())


@lru_cache(maxsize=64)
def _read_table_cached(source: str) -> pd.DataFrame:
    if source.lower().endswith((".pq", ".parquet")):
        return pd.read_parquet(source)
    return pd.read_csv(source)


def _get_dataframe(filepath: str | Path) -> pd.DataFrame:
    """Read a CSV or parquet table from a packaged asset or remote URL."""
    return _read_table_cached(str(resolve_asset_path(filepath, default_dir="data"))).copy()


def _coerce_proportions(
    df: pd.DataFrame,
    sample_id_column: str = "sample_id",
) -> pd.DataFrame:
    """Row-normalize raw NMF loadings into component proportions.

    Remote ``*.nmf.sample.pq`` tables carry raw ``NMF1``..``NMFk`` loadings.
    The heatmap / grandscatter layers expect proportions named
    ``Comp_0``..``Comp_{k-1}`` (``NMF{i}`` maps to ``Comp_{i-1}``). Frames that
    already use ``Comp_*`` columns (legacy bundled CSVs) pass through unchanged.
    """
    raw = {
        col: int(match.group(1))
        for col in df.columns
        if (match := _RAW_NMF_RE.match(str(col).strip()))
    }
    if len(raw) < 2:
        return df

    ordered = sorted(raw, key=raw.get)
    matrix = df[ordered].to_numpy(dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    proportions = matrix / totals

    out = df.drop(columns=ordered).copy()
    for position, col in enumerate(ordered):
        out[f"Comp_{raw[col] - 1}"] = proportions[:, position]
    return out


def _standardize_sample_id_column(
    df: pd.DataFrame,
    target: str = "sample_id",
) -> tuple[pd.DataFrame, str]:
    """Ensure the dataframe has a consistently named sample-ID column."""
    if target in df.columns:
        return df, target

    for candidate in _SAMPLE_ID_CANDIDATES:
        if candidate in df.columns:
            return df.rename(columns={candidate: target}), target

    first = df.columns[0]
    if pd.api.types.is_object_dtype(df[first]):
        return df.rename(columns={first: target}), target

    raise ValueError(f"Cannot identify a sample-ID column in: {list(df.columns)}")


def _component_index(col: str) -> int | None:
    match = re.match(r"(?i)^comp(?:onent)?[_\s-]*(\d+)$", col.strip())
    return int(match.group(1)) if match else None


def _extract_component_columns(
    df: pd.DataFrame,
    sample_id_column: str = "sample_id",
) -> list[str]:
    component_like = [
        col
        for col in df.columns
        if col != sample_id_column and _component_index(col) is not None
    ]
    if component_like:
        return component_like

    return [
        col
        for col in df.columns
        if col != sample_id_column and pd.api.types.is_numeric_dtype(df[col])
    ]


def _canonicalize_component_columns(
    columns: list[str],
) -> tuple[list[str], np.ndarray | None]:
    """Sort by numeric component ID; return (sorted_cols, reorder_indices)."""
    indexed = [(_component_index(column), column, i) for i, column in enumerate(columns)]
    ids = [idx for idx, _, _ in indexed]

    if any(idx is None for idx in ids) or len(set(ids)) != len(ids):
        return columns, None

    sorted_idx = sorted(indexed, key=lambda item: item[0])  # type: ignore[arg-type]
    return (
        [column for _, column, _ in sorted_idx],
        np.array([i for _, _, i in sorted_idx], dtype=int),
    )


def _resolve_cancer_types(
    sample_ids: list[str],
    cfg: dict,
) -> list[str]:
    """Look up cancer types from metadata or fall back to the 4-char prefix."""
    metadata_path = cfg.get("ANALYSIS_METADATA_FILENAME")
    if not metadata_path:
        return [sid[:4] for sid in sample_ids]

    meta_df = _get_dataframe(metadata_path)
    meta_df, meta_sid_col = _standardize_sample_id_column(
        meta_df,
        str(cfg.get("ANALYSIS_METADATA_SAMPLE_ID_COLUMN", "sample_id")),
    )
    cancer_col = str(cfg.get("ANALYSIS_CANCER_TYPE_COLUMN", "cancer_type"))
    if cancer_col not in meta_df.columns:
        return [sid[:4] for sid in sample_ids]

    lookup = meta_df.drop_duplicates(subset=[meta_sid_col]).set_index(meta_sid_col)[
        cancer_col
    ]
    return [
        str(lookup.get(sid, "Unknown")) if pd.notna(lookup.get(sid)) else "Unknown"
        for sid in sample_ids
    ]


def _get_prepared_data(
    filepath: str | Path,
    sample_id_column: str = "sample_id",
    selection: list[int] | None = None,
    metadata_path: str | Path | None = None,
    metadata_sample_id_column: str = "sample_id",
    cancer_type_column: str = "cancer_type",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return ``(H, sample_ids, cancer_types)`` ready for visualization.

    ``filepath`` may be a CSV/parquet asset or a remote URL holding either raw
    NMF loadings (``NMF1``..``NMFk``) or proportions (``Comp_*``); raw loadings
    are row-normalized into proportions by :func:`_coerce_proportions`.
    """
    df = _coerce_proportions(_get_dataframe(filepath))
    df, sample_id_column = _standardize_sample_id_column(df, sample_id_column)

    if selection is not None:
        df = df.iloc[selection]

    sample_ids = df[sample_id_column].astype(str).tolist()
    component_columns = _extract_component_columns(df, sample_id_column)
    sorted_columns, _ = _canonicalize_component_columns(component_columns)

    if not component_columns:
        raise ValueError(f"No component columns found in {filepath}")
    H = df[sorted_columns].to_numpy()

    if metadata_path is None:
        cancer_types = [sample_id[:4] for sample_id in sample_ids]
    else:
        metadata_df = _get_dataframe(metadata_path)
        metadata_df, metadata_sample_id_column = _standardize_sample_id_column(
            metadata_df,
            metadata_sample_id_column,
        )
        if cancer_type_column in metadata_df.columns:
            cancer_lookup = (
                metadata_df.drop_duplicates(subset=[metadata_sample_id_column])
                .set_index(metadata_sample_id_column)[cancer_type_column]
            )
            cancer_types = [
                str(cancer_lookup.get(sample_id, "Unknown"))
                if pd.notna(cancer_lookup.get(sample_id, "Unknown"))
                else "Unknown"
                for sample_id in sample_ids
            ]
        else:
            cancer_types = [sample_id[:4] for sample_id in sample_ids]

    return H, sample_ids, cancer_types


def load_analysis_umap_data(
    cfg_or_path: str | Path | dict = "conf/config.json",
    analysis_name: str | None = None,
) -> pd.DataFrame:
    cfg = resolve_analysis_cfg(cfg_or_path, analysis_name)
    umap_df = _get_dataframe(cfg.get("UMAP_FILENAME", "data/umap.parquet"))

    rename_map = {}
    if "UMAP_1" in umap_df.columns:
        rename_map["UMAP_1"] = "UMAP-1"
    if "UMAP_2" in umap_df.columns:
        rename_map["UMAP_2"] = "UMAP-2"
    if "Sample_ID" in umap_df.columns:
        rename_map["Sample_ID"] = "Sample ID"
    if "Dominant_Component" in umap_df.columns:
        rename_map["Dominant_Component"] = "Dominant Component"
    umap_df = umap_df.rename(columns=rename_map)
    umap_df, umap_sample_id_column = _standardize_sample_id_column(umap_df, "Sample ID")

    component_df = _get_dataframe(cfg.get("DEFAULT_CSV_FILENAME", "data/all_H_component_contributions.csv"))
    component_df, component_sample_id_column = _standardize_sample_id_column(component_df)
    sample_ids = component_df[component_sample_id_column].astype(str).tolist()
    cancer_types = _resolve_cancer_types(sample_ids, cfg)

    aligned = (
        umap_df.drop_duplicates(subset=[umap_sample_id_column])
        .set_index(umap_sample_id_column)
        .reindex(sample_ids)
        .reset_index()
        .rename(columns={"index": "Sample ID"})
    )
    aligned["Sample ID"] = aligned["Sample ID"].astype(str)
    aligned["Cancer Type"] = cancer_types
    return aligned


def prepare_grandscatter_data(
    cfg_path: str | Path = "conf/config.json",
    selection: list[int] | None = None,
    analysis_name: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Build the dataframe and colors needed by ``grandscatter.Scatter``."""
    cfg = resolve_analysis_cfg(cfg_path, analysis_name)
    source = cfg.get(
        "GRANDSCATTER_CSV_FILENAME",
        cfg.get("DEFAULT_CSV_FILENAME", "data/all_H_component_contributions_k16.csv"),
    )

    component_df = _coerce_proportions(_get_dataframe(source))
    component_df, sample_id_column = _standardize_sample_id_column(component_df)

    if selection is not None:
        component_df = component_df.iloc[selection].reset_index(drop=True)

    component_columns = _extract_component_columns(component_df, sample_id_column)
    if not component_columns:
        raise ValueError(f"No component columns found in grandscatter source: {source}")

    axis_fields, _ = _canonicalize_component_columns(component_columns)
    H_prop = component_df[axis_fields].to_numpy()

    df = pd.DataFrame(H_prop.astype(np.float32), columns=axis_fields)
    sample_ids = component_df[sample_id_column].astype(str).tolist()
    df.insert(0, "sample_id", sample_ids)
    df["cancer_type"] = _resolve_cancer_types(sample_ids, cfg)
    df["cancer_type"] = df["cancer_type"].fillna("Unknown").astype("category")

    user_color_map = load_color_map(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS", ""))
    label_colors = resolve_cancer_colors(df["cancer_type"].astype(str).tolist(), user_color_map)
    return df, axis_fields, label_colors
