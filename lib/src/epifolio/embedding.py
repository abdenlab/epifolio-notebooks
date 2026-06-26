"""PCA/UMAP embedding display and HiPlot color helpers for the explorer notebooks."""

from __future__ import annotations

import re

import pandas as pd

from epifolio.colors import distinct_palette, unique_sorted_unknown_last
from epifolio.metadata import _FLAG_COLUMNS, format_flag
from epifolio.tcga import cancer_type_from_sample_id, patient_id_from_sample_id


def prepare_embedding_sample_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Shape a loaded embedding frame: rename ``sample``->``Sample ID``; add patient/cancer.

    Pure transform — the caller reads the parquet explicitly and passes the frame.
    """
    if "sample" not in df.columns:
        raise ValueError("Expected a 'sample' column in the embedding frame")
    df = df.rename(columns={"sample": "Sample ID"})
    df["patient_id"] = df["Sample ID"].map(patient_id_from_sample_id)
    df["Cancer Type"] = df["Sample ID"].map(cancer_type_from_sample_id)
    return df


_COLOR_FIELD_CANDIDATES = [
    "Cancer Type",
    "cancer_type_full",
    "primary_diagnosis",
    "molecular_subtype",
    "vital_status",
    "sample_types",
    "gender",
    "race",
    "ethnicity",
    "ajcc_stage_simple",
    "received_surgery",
    "received_chemotherapy",
    "received_radiation",
    "received_hormone_therapy",
    "received_immunotherapy",
]

def _cohort_color_lookup(
    values: list[str],
    cancer_color_map: dict[str, str],
    unknown_color: str,
) -> dict[str, str]:
    lookup = {}
    for value in unique_sorted_unknown_last(values):
        if value == "Unknown":
            lookup[value] = unknown_color
            continue

        preferred_keys = [value, f"{value}x", f"{value}xx"]
        color = None
        for key in preferred_keys:
            color = cancer_color_map.get(key)
            if color is not None:
                break
        if color is None:
            for key, candidate in cancer_color_map.items():
                if str(key).startswith(value):
                    color = candidate
                    break
        lookup[value] = color or unknown_color
    return lookup

def build_strip_color_lookup(
    values: list[str],
    selected_color_column: str,
    strip_spec_by_column: dict[str, dict],
    cancer_color_map: dict[str, str],
    unknown_color: str,
) -> dict[str, str]:
    strip_spec = strip_spec_by_column.get(selected_color_column)
    if strip_spec is None:
        categories = unique_sorted_unknown_last(values)
        auto_palette = distinct_palette(max(len(categories), 1))
        lookup = {}
        auto_index = 0
        for category in categories:
            if category == "Unknown":
                lookup[category] = unknown_color
            else:
                lookup[category] = auto_palette[auto_index % len(auto_palette)]
                auto_index += 1
        return lookup

    if strip_spec.get("palette_source") == "cancer_type_colors":
        return _cohort_color_lookup(values, cancer_color_map, unknown_color)

    if "color_map" in strip_spec:
        lookup = {str(key): str(color) for key, color in strip_spec["color_map"].items()}
        lookup.setdefault("Unknown", unknown_color)
        return lookup

    palette = [str(color) for color in strip_spec.get("palette", [])]
    categories = unique_sorted_unknown_last(values)
    if not palette:
        return {category: unknown_color for category in categories}

    lookup = {}
    for index, category in enumerate(categories):
        if category == "Unknown":
            lookup[category] = unknown_color
        else:
            lookup[category] = palette[index % len(palette)]
    return lookup

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    pattern = re.compile(r"^(NMF|PC)(\d+)$")
    feature_columns = [column for column in df.columns if pattern.match(str(column))]
    return sorted(
        feature_columns,
        key=lambda column: (
            pattern.match(str(column)).group(1),
            int(pattern.match(str(column)).group(2)),
        ),
    )

def available_color_fields(clinical_df: pd.DataFrame) -> list[str]:
    return [
        field
        for field in _COLOR_FIELD_CANDIDATES
        if field == "Cancer Type" or field in clinical_df.columns
    ]

def _build_label_colors(
    display_df: pd.DataFrame,
    color_by: str,
    cancer_color_map: dict[str, str] | None = None,
) -> dict[str, str]:
    cancer_color_map = cancer_color_map or {}
    categories = sorted(display_df["Color Label"].astype(str).unique())

    if color_by == "Cancer Type":
        auto_palette = distinct_palette(len(categories))
        colors = {
            category: cancer_color_map.get(category, auto_palette[index])
            for index, category in enumerate(categories)
        }
    elif color_by == "cancer_type_full":
        cancer_lookup = (
            display_df[["Color Label", "Cancer Type"]]
            .dropna(subset=["Color Label", "Cancer Type"])
            .drop_duplicates(subset=["Color Label"])
            .set_index("Color Label")["Cancer Type"]
            .to_dict()
        )
        auto_palette = distinct_palette(len(categories))
        colors = {
            category: cancer_color_map.get(cancer_lookup.get(category), auto_palette[index])
            for index, category in enumerate(categories)
        }
    else:
        auto_palette = distinct_palette(len(categories))
        colors = {category: auto_palette[index] for index, category in enumerate(categories)}

    if "Unknown" in colors:
        colors["Unknown"] = "#9CA3AF"
    return colors

def prepare_embedding_for_display(
    embedding_df: pd.DataFrame,
    clinical_df: pd.DataFrame,
    color_by: str,
    cancer_color_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if "patient_id" not in embedding_df.columns:
        embedding_df = embedding_df.copy()
        embedding_df["patient_id"] = embedding_df["Sample ID"].map(patient_id_from_sample_id)

    display_df = embedding_df.merge(
        clinical_df,
        on="patient_id",
        how="left",
        suffixes=("", "_metadata"),
    )

    if "cancer_type" in display_df.columns:
        display_df["Cancer Type"] = (
            display_df["cancer_type"]
            .fillna(display_df.get("Cancer Type"))
            .fillna("Unknown")
            .astype(str)
        )
    elif "Cancer Type" in display_df.columns:
        display_df["Cancer Type"] = display_df["Cancer Type"].fillna("Unknown").astype(str)
    else:
        display_df["Cancer Type"] = "Unknown"

    if color_by == "Cancer Type":
        color_values = display_df["Cancer Type"]
    else:
        if color_by not in display_df.columns:
            raise ValueError(f"Unknown metadata color field: {color_by}")
        color_values = display_df[color_by]

    if color_by in _FLAG_COLUMNS:
        color_values = color_values.map(format_flag)

    display_df["Color Label"] = color_values.fillna("Unknown").astype(str)
    label_colors = _build_label_colors(display_df, color_by, cancer_color_map)
    return display_df, label_colors

def sample_ids_to_indices(
    embedding_df: pd.DataFrame,
    selected_sample_ids: list[str],
) -> list[int]:
    index_lookup = {
        sample_id: index
        for index, sample_id in enumerate(embedding_df["Sample ID"].astype(str).tolist())
    }
    return [index_lookup[sample_id] for sample_id in selected_sample_ids if sample_id in index_lookup]


def default_visible_features(
    feature_columns: list[str],
    limit: int = 10,
) -> list[str]:
    return feature_columns[: min(limit, len(feature_columns))]

def coerce_visible_features(
    feature_columns: list[str],
    selected_features: list[str] | None,
    limit: int = 10,
) -> list[str]:
    if not selected_features:
        return default_visible_features(feature_columns, limit=limit)

    feature_set = set(feature_columns)
    ordered = [feature for feature in selected_features if feature in feature_set]
    return ordered or default_visible_features(feature_columns, limit=limit)
