"""Data loading and preparation for NMF visualizations."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_cfg(path: str | Path = "config.json") -> dict:
    """Read a JSON configuration file."""
    with open(path) as f:
        return json.load(f)


def resolve_analysis_cfg(
    cfg_or_path: str | Path | dict,
    analysis_name: str | None = None,
) -> dict:
    """Merge a per-analysis overlay into the base config.

    If the config has an ``ANALYSES`` dict, the selected analysis's keys
    are merged on top of the base (non-``ANALYSES``) keys.
    """
    cfg = load_cfg(cfg_or_path) if isinstance(cfg_or_path, (str, Path)) else dict(cfg_or_path)
    analyses = cfg.get("ANALYSES")
    if not analyses:
        return cfg

    selected = analysis_name or cfg.get("DEFAULT_ANALYSIS") or next(iter(analyses))
    if selected not in analyses:
        raise ValueError(
            f"Unknown analysis '{selected}'. Available: {', '.join(analyses)}"
        )

    resolved = {k: v for k, v in cfg.items() if k != "ANALYSES"}
    resolved.update(analyses[selected])
    resolved["SELECTED_ANALYSIS"] = selected
    return resolved


def get_available_analyses(cfg_path: str | Path = "conf/config.json") -> list[str]:
    cfg = load_cfg(cfg_path)
    analyses = cfg.get("ANALYSES")
    if not analyses:
        return [cfg.get("DEFAULT_ANALYSIS", "Default")]
    return list(analyses.keys())


# ---------------------------------------------------------------------------
# DataFrame / column helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=64)
def _read_csv_cached(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def _get_dataframe(filepath: Path) -> pd.DataFrame:
    return _read_csv_cached(str(filepath)).copy()


_SAMPLE_ID_CANDIDATES = [
    "sample_id", "Sample_ID", "Sample ID", "sample", "Unnamed: 0",
]


def _standardize_sample_id_column(
    df: pd.DataFrame,
    target: str = "sample_id",
) -> tuple[pd.DataFrame, str]:
    """Ensure the dataframe has a consistently-named sample-ID column."""
    if target in df.columns:
        return df, target

    for candidate in _SAMPLE_ID_CANDIDATES:
        if candidate in df.columns:
            return df.rename(columns={candidate: target}), target

    # Last resort: first string-typed column
    first = df.columns[0]
    if pd.api.types.is_object_dtype(df[first]):
        return df.rename(columns={first: target}), target

    raise ValueError(f"Cannot identify a sample-ID column in: {list(df.columns)}")


def _component_index(col: str) -> int | None:
    m = re.match(r"(?i)^comp(?:onent)?[_\s-]*(\d+)$", col.strip())
    return int(m.group(1)) if m else None


def _extract_component_columns(
    df: pd.DataFrame,
    sample_id_column: str = "sample_id",
) -> list[str]:
    comp_like = [
        c for c in df.columns
        if c != sample_id_column and _component_index(c) is not None
    ]
    if comp_like:
        return comp_like
    return [
        c for c in df.columns
        if c != sample_id_column and pd.api.types.is_numeric_dtype(df[c])
    ]


def _canonicalize_component_columns(
    columns: list[str],
) -> tuple[list[str], np.ndarray | None]:
    """Sort by numeric component ID; return (sorted_cols, reorder_indices)."""
    indexed = [(_component_index(c), c, i) for i, c in enumerate(columns)]
    ids = [idx for idx, _, _ in indexed]

    if any(idx is None for idx in ids) or len(set(ids)) != len(ids):
        return columns, None

    sorted_idx = sorted(indexed, key=lambda x: x[0])  # type: ignore[arg-type]
    return (
        [c for _, c, _ in sorted_idx],
        np.array([i for _, _, i in sorted_idx], dtype=int),
    )


# ---------------------------------------------------------------------------
# Core data loaders
# ---------------------------------------------------------------------------

def _resolve_cancer_types(
    sample_ids: list[str],
    cfg: dict,
) -> list[str]:
    """Look up cancer types from metadata or fall back to 4-char prefix."""
    metadata_path = cfg.get("ANALYSIS_METADATA_FILENAME")
    if not metadata_path:
        return [sid[:4] for sid in sample_ids]

    meta_df = _get_dataframe(Path(metadata_path))
    meta_df, meta_sid_col = _standardize_sample_id_column(
        meta_df, cfg.get("ANALYSIS_METADATA_SAMPLE_ID_COLUMN", "sample_id"),
    )
    cancer_col = cfg.get("ANALYSIS_CANCER_TYPE_COLUMN", "cancer_type")
    if cancer_col not in meta_df.columns:
        return [sid[:4] for sid in sample_ids]

    lookup = (
        meta_df.drop_duplicates(subset=[meta_sid_col])
        .set_index(meta_sid_col)[cancer_col]
    )
    return [
        str(lookup.get(sid, "Unknown")) if pd.notna(lookup.get(sid)) else "Unknown"
        for sid in sample_ids
    ]


def load_h_matrix(
    csv_path: Path,
    *,
    npy_path: Path | None = None,
    selection: list[int] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Load an NMF H matrix with sample IDs and component column names.

    Returns ``(H, sample_ids, component_names)``.
    """
    df = _get_dataframe(csv_path)
    df, sid_col = _standardize_sample_id_column(df)

    if selection is not None:
        df = df.iloc[selection]

    sample_ids = df[sid_col].astype(str).tolist()
    comp_cols = _extract_component_columns(df, sid_col)
    sorted_cols, reorder = _canonicalize_component_columns(comp_cols)

    if npy_path is not None:
        H = np.load(npy_path)
        if selection is not None:
            H = H[selection]
        if reorder is not None and len(comp_cols) == H.shape[1]:
            H = H[:, reorder]
    else:
        if not comp_cols:
            raise ValueError(f"No component columns found in {csv_path}")
        H = df[sorted_cols].values

    return H, sample_ids, sorted_cols


def load_umap(
    cfg: dict,
    sample_ids: list[str],
    cancer_types: list[str],
) -> pd.DataFrame:
    """Load UMAP coordinates aligned to *sample_ids*."""
    umap_path = Path(cfg.get("UMAP_FILENAME", "data/umap.parquet"))

    if umap_path.suffix.lower() == ".parquet":
        umap_df = pd.read_parquet(umap_path)
    else:
        umap_df = pd.read_csv(umap_path)

    rename = {}
    if "UMAP_1" in umap_df.columns:
        rename["UMAP_1"] = "UMAP-1"
    if "UMAP_2" in umap_df.columns:
        rename["UMAP_2"] = "UMAP-2"
    umap_df = umap_df.rename(columns=rename)
    umap_df, sid_col = _standardize_sample_id_column(umap_df, "sample_id")

    aligned = (
        umap_df.drop_duplicates(subset=[sid_col])
        .set_index(sid_col)
        .reindex(sample_ids)
        .reset_index()
    )
    aligned["sample_id"] = aligned["sample_id"].astype(str)
    aligned["cancer_type"] = cancer_types
    return aligned
