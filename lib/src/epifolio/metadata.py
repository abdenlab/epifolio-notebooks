"""Clinical-metadata shaping and table presentation for the embedding notebooks."""

from __future__ import annotations

import pandas as pd


def metadata_tooltips(dictionary_df: pd.DataFrame) -> dict[str, str]:
    """Build a ``{column: description}`` tooltip map from a loaded data dictionary."""
    return (
        dictionary_df.dropna(subset=["description"])
        .drop_duplicates(subset=["column"])
        .set_index("column")["description"]
        .to_dict()
    )


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
