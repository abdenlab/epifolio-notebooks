"""Helpers for PCA embedding display, metadata shaping, and HiPlot color handling."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from epifolio import ASSETS
from epifolio.color_utils import distinct_palette, load_color_map
from epifolio.data_utils import resolve_asset_path

_REMOTE_METADATA = "https://projects.abdenlab.org/itcr/epifolio/metadata"
CLINICAL_METADATA_PATH = f"{_REMOTE_METADATA}/unified_clinical_metadata.csv"
METADATA_DICTIONARY_PATH = f"{_REMOTE_METADATA}/unified_clinical_metadata_dictionary.csv"
CANCER_COLOR_PATH = ASSETS / "conf" / "cancer_type_color_map.json"

_METADATA_GROUPS = {
    "Overview": [
        "selection_index",
        "Sample ID",
        "patient_id",
        "Cancer Type",
        "Color Label",
        "cancer_type_full",
        "primary_diagnosis",
        "molecular_subtype",
        "vital_status",
        "age_at_diagnosis_years",
        "sample_types",
        "tss_enrichment_mean",
        "frip_mean",
        "final_reads_mean",
    ],
    "Demographics": [
        "selection_index",
        "Sample ID",
        "patient_id",
        "Cancer Type",
        "Color Label",
        "cancer_type_full",
        "age_at_diagnosis_years",
        "gender",
        "race",
        "ethnicity",
        "sample_types",
        "laterality",
    ],
    "Diagnosis & Stage": [
        "selection_index",
        "Sample ID",
        "patient_id",
        "Cancer Type",
        "Color Label",
        "primary_diagnosis",
        "morphology_code",
        "tissue_or_organ_of_origin",
        "tumor_grade",
        "residual_disease",
        "ajcc_stage_simple",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
    ],
    "Survival & Treatment": [
        "selection_index",
        "Sample ID",
        "patient_id",
        "vital_status",
        "os_event",
        "os_time_days",
        "os_time_months",
        "days_to_death",
        "days_to_last_followup",
        "received_surgery",
        "received_chemotherapy",
        "received_radiation",
        "received_hormone_therapy",
        "received_immunotherapy",
        "treatment_types",
        "treatment_outcomes",
    ],
    "Molecular & QC": [
        "selection_index",
        "Sample ID",
        "patient_id",
        "molecular_subtype",
        "subtype_method",
        "subtype_source",
        "subtype_available",
        "receptor_status",
        "brca_ic10",
        "msi_mantis_score",
        "msi_sensor_score",
        "batch_numbers",
        "tss_enrichment_mean",
        "frip_mean",
        "final_reads_mean",
    ],
}

_BASE_TOOLTIPS = {
    "selection_index": "Row index in the linked assay views used for synchronized selections.",
    "Sample ID": "ATAC-seq sample barcode shown in the linked views.",
    "Cancer Type": "Four-letter TCGA cancer type prefix derived from the sample barcode.",
    "Color Label": "Current metadata field used to color the linked view.",
    "sample_count": "Number of assay rows currently linked to the same patient_id.",
    "sample_ids": "Semicolon-delimited sample barcodes linked to this patient in the current view.",
    "cancer_types": "Cancer type labels represented by the repeated assay rows.",
}

_WRAPPED_COLUMNS = [
    "Sample ID",
    "sample_ids",
    "cancer_type_full",
    "primary_diagnosis",
    "tissue_or_organ_of_origin",
    "treatment_types",
    "treatment_outcomes",
    "molecular_subtype",
    "subtype_method",
    "subtype_source",
    "receptor_status",
    "batch_numbers",
]

_FORMAT_MAPPING = {
    "age_at_diagnosis_years": "{:.1f}",
    "os_time_days": "{:.0f}",
    "os_time_months": "{:.1f}",
    "days_to_death": "{:.0f}",
    "days_to_last_followup": "{:.0f}",
    "msi_mantis_score": "{:.3f}",
    "msi_sensor_score": "{:.3f}",
    "tss_enrichment_mean": "{:.2f}",
    "frip_mean": "{:.3f}",
    "final_reads_mean": "{:,.0f}",
}

_FLAG_COLUMNS = [
    "os_event",
    "received_surgery",
    "received_chemotherapy",
    "received_radiation",
    "received_hormone_therapy",
    "received_immunotherapy",
    "subtype_available",
]

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


def patient_id_from_sample_id(sample_id: str) -> str | None:
    parts = str(sample_id).split("-")
    if len(parts) >= 4 and parts[1] == "TCGA":
        return "-".join(parts[1:4])
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return None


def cancer_type_from_sample_id(sample_id: str) -> str:
    return str(sample_id)[:4]


def summarize_cancer_types(cancer_types) -> str:
    unique_types = sorted({str(value) for value in cancer_types if pd.notna(value)})
    if len(unique_types) <= 6:
        return ", ".join(unique_types)
    return f"{', '.join(unique_types[:6])} + {len(unique_types) - 6} more"


def format_flag(value):
    if pd.isna(value):
        return None

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return "Yes"
    if normalized in {"false", "0", "no"}:
        return "No"
    return value


for column in _FLAG_COLUMNS:
    _FORMAT_MAPPING[column] = format_flag


def _unique_sorted_unknown_last(values: list[str]) -> list[str]:
    unique_values = sorted({str(value) for value in values if pd.notna(value)})
    if "Unknown" in unique_values:
        unique_values = [value for value in unique_values if value != "Unknown"] + ["Unknown"]
    return unique_values


def _cohort_color_lookup(
    values: list[str],
    cancer_color_map: dict[str, str],
    unknown_color: str,
) -> dict[str, str]:
    lookup = {}
    for value in _unique_sorted_unknown_last(values):
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
        categories = _unique_sorted_unknown_last(values)
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
    categories = _unique_sorted_unknown_last(values)
    if not palette:
        return {category: unknown_color for category in categories}

    lookup = {}
    for index, category in enumerate(categories):
        if category == "Unknown":
            lookup[category] = unknown_color
        else:
            lookup[category] = palette[index % len(palette)]
    return lookup


def load_clinical_metadata(path: str | Path = CLINICAL_METADATA_PATH) -> pd.DataFrame:
    return pd.read_csv(resolve_asset_path(path, default_dir="data"))


def load_metadata_tooltips(path: str | Path = METADATA_DICTIONARY_PATH) -> dict[str, str]:
    return (
        pd.read_csv(resolve_asset_path(path, default_dir="data"))
        .dropna(subset=["description"])
        .drop_duplicates(subset=["column"])
        .set_index("column")["description"]
        .to_dict()
    )


def load_cancer_color_map(path: str | Path = CANCER_COLOR_PATH) -> dict[str, str]:
    return load_color_map(resolve_asset_path(path, default_dir="conf"))


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


def load_embedding_sample_frame(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(resolve_asset_path(path, default_dir="data")).copy()
    if "sample" not in df.columns:
        raise ValueError(f"Expected a 'sample' column in {path}")

    df = df.rename(columns={"sample": "Sample ID"})
    df["patient_id"] = df["Sample ID"].map(patient_id_from_sample_id)
    df["Cancer Type"] = df["Sample ID"].map(cancer_type_from_sample_id)
    return df


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


def subset_by_sample_ids(
    embedding_df: pd.DataFrame,
    selected_sample_ids: list[str] | None,
) -> pd.DataFrame:
    if not selected_sample_ids:
        return embedding_df.copy()

    sample_id_set = set(selected_sample_ids)
    return (
        embedding_df[embedding_df["Sample ID"].isin(sample_id_set)]
        .copy()
        .reset_index(drop=True)
    )


def sample_ids_to_indices(
    embedding_df: pd.DataFrame,
    selected_sample_ids: list[str],
) -> list[int]:
    index_lookup = {
        sample_id: index
        for index, sample_id in enumerate(embedding_df["Sample ID"].astype(str).tolist())
    }
    return [index_lookup[sample_id] for sample_id in selected_sample_ids if sample_id in index_lookup]


def _categorical_color_lookup(
    values: list[str],
    preferred_colors: dict[str, str] | None = None,
) -> dict[str, str]:
    preferred_colors = preferred_colors or {}
    unique_values = sorted({str(value) for value in values if pd.notna(value)})
    auto_palette = distinct_palette(len(unique_values))
    lookup = {
        value: preferred_colors.get(value, auto_palette[index])
        for index, value in enumerate(unique_values)
    }
    if "Unknown" in {str(value) for value in values}:
        lookup.setdefault("Unknown", "#9CA3AF")
    return lookup


def _color_to_hiplot(color: str) -> str:
    color = str(color).strip()
    if color.startswith(("rgb(", "rgba(", "hsl(", "hsla(")):
        return color.replace("rgba(", "rgb(").replace("hsla(", "hsl(")
    if color.startswith("#"):
        hex_color = color[1:]
        if len(hex_color) == 3:
            hex_color = "".join(channel * 2 for channel in hex_color)
        if len(hex_color) == 6:
            red = int(hex_color[0:2], 16)
            green = int(hex_color[2:4], 16)
            blue = int(hex_color[4:6], 16)
            return f"rgb({red}, {green}, {blue})"
    return color


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


def build_metadata_views(
    display_df: pd.DataFrame,
    selected_sample_ids: list[str] | None,
):
    assay_df = subset_by_sample_ids(display_df, selected_sample_ids)
    is_filtered = bool(selected_sample_ids)

    assay_df = assay_df.reset_index().rename(columns={"index": "selection_index"})
    duplicate_patients = (
        assay_df.groupby("patient_id", dropna=False)
        .agg(
            sample_count=("Sample ID", "size"),
            cancer_types=("Cancer Type", summarize_cancer_types),
            sample_ids=("Sample ID", lambda values: "; ".join(map(str, values))),
        )
        .reset_index()
    )
    duplicate_patients = duplicate_patients[duplicate_patients["sample_count"] > 1]

    matched_column = "case_uuid" if "case_uuid" in assay_df.columns else None
    matched_sample_count = (
        int(assay_df[matched_column].notna().sum())
        if matched_column
        else int(len(assay_df))
    )
    summary = {
        "is_filtered": is_filtered,
        "sample_count": int(len(assay_df)),
        "unique_patient_count": int(assay_df["patient_id"].nunique(dropna=True)),
        "matched_sample_count": matched_sample_count,
        "duplicate_patient_count": int(len(duplicate_patients)),
        "cancer_type_count": int(assay_df["Cancer Type"].nunique(dropna=True)),
        "cancer_types_label": summarize_cancer_types(assay_df["Cancer Type"]),
    }
    return assay_df, duplicate_patients, summary


def metadata_presentation() -> tuple[dict[str, list[str]], dict[str, str], list[str], dict[str, object]]:
    return _METADATA_GROUPS, _BASE_TOOLTIPS, _WRAPPED_COLUMNS, _FORMAT_MAPPING
