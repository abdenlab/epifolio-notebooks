"""Sample ordering strategies for NMF heatmaps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bar_sort_order(mat: np.ndarray) -> np.ndarray:
    """Order samples by winning component, then by activity within each."""
    winners = np.argmax(mat, axis=1)
    order: list[int] = []
    for comp in range(mat.shape[1]):
        idx = np.argsort(-mat[:, comp])
        order.extend(idx[winners[idx] == comp])
    return np.asarray(order, dtype=int)


def _group_then_component_sort(
    H: np.ndarray,
    group_labels: list[str],
) -> np.ndarray:
    """Sort by group label, then by component within each group."""
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    order: list[int] = []
    for group in sorted(set(group_labels)):
        sub = np.where(np.array(group_labels) == group)[0]
        if len(sub) == 0:
            continue
        sub_order = bar_sort_order(H_ord[sub])
        order.extend(sub[sub_order])
    return np.array(order, dtype=int)


def _load_grouping_labels(
    sample_ids: list[str],
    grouping_path: str | Path,
) -> list[str]:
    """Map 4-letter cancer codes from sample IDs to grouping labels."""
    with open(grouping_path) as f:
        groups = json.load(f).get("organ_system_groupings", [])
    code_to_group = {}
    for g in groups:
        for code in g["cancer_codes"]:
            code_to_group[code[:4]] = g["group_name"]
    return [code_to_group.get(sid[:4], "Unknown") for sid in sample_ids]


def get_sample_order(
    method: str,
    H: np.ndarray,
    sample_ids: list[str],
    cancer_types: list[str],
    *,
    organ_system_path: str | Path | None = None,
    embryonic_layer_path: str | Path | None = None,
) -> np.ndarray:
    """Dispatch to a sorting strategy by name.

    Parameters
    ----------
    method : str
        One of ``"component"``, ``"alphabetical"``, ``"cancer_type"``,
        ``"organ_system"``, ``"embryonic_layer"``.
    """
    comp_order = np.argsort(-H.sum(axis=0))
    H_ord = H[:, comp_order]

    if method == "component":
        return bar_sort_order(H_ord)

    if method == "alphabetical":
        return np.argsort(sample_ids)

    if method == "cancer_type":
        return _group_then_component_sort(H, cancer_types)

    if method == "organ_system" and organ_system_path:
        labels = _load_grouping_labels(sample_ids, organ_system_path)
        return _group_then_component_sort(H, labels)

    if method == "embryonic_layer" and embryonic_layer_path:
        labels = _load_grouping_labels(sample_ids, embryonic_layer_path)
        return _group_then_component_sort(H, labels)

    # Fallback
    return bar_sort_order(H_ord)
