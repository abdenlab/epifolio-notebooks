"""TCGA sample-barcode parsing helpers."""

from __future__ import annotations


def patient_id_from_sample_id(sample_id: str) -> str | None:
    """Extract the TCGA patient barcode (``TCGA-XX-XXXX``) from a sample ID."""
    parts = str(sample_id).split("-")
    if "TCGA" in parts:
        idx = parts.index("TCGA")
        if idx + 2 < len(parts):
            return "-".join(parts[idx : idx + 3])
    if len(parts) >= 4 and parts[1] == "TCGA":
        return "-".join(parts[1:4])
    if len(parts) >= 3 and parts[0] == "TCGA":
        return "-".join(parts[:3])
    return None


def cancer_type_from_sample_id(sample_id: str) -> str:
    """Return the 4-character TCGA cancer-type code prefix of a sample ID."""
    return str(sample_id)[:4]
