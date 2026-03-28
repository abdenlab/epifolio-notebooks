"""Interactive scatter plot builders (jscatter, grandscatter)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from epifolio.color_utils import resolve_cancer_colors
from epifolio.data_utils import (
    _get_dataframe,
    _standardize_sample_id_column,
    _extract_component_columns,
    _canonicalize_component_columns,
    _resolve_cancer_types,
    resolve_analysis_cfg,
)


def create_umap_scatter(
    umap_df: pd.DataFrame,
    cancer_color_map: dict[str, str],
) -> "jscatter.Scatter":
    """Create a jscatter UMAP plot with lasso selection.

    Parameters
    ----------
    umap_df : DataFrame
        Must contain ``UMAP-1``, ``UMAP-2``, ``sample_id``, ``cancer_type``.
    cancer_color_map : dict
        Cancer-type -> hex color mapping.
    """
    import jscatter

    scatter = jscatter.Scatter(
        data=umap_df,
        x="UMAP-1",
        y="UMAP-2",
        color_by="cancer_type",
        color_map=cancer_color_map,
        height=600,
        width=600,
        lasso_callback=True,
        selection_mode="lasso",
    )
    scatter.tooltip(enable=True, properties=["sample_id", "cancer_type"])
    scatter.size(default=5)
    return scatter


def prepare_grandscatter_data(
    cfg_or_path: str | Path | dict = "conf/config.json",
    *,
    selection: list[int] | None = None,
    analysis_name: str | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Build a DataFrame for ``grandscatter.Scatter``.

    Returns ``(df, axis_fields, label_colors)`` where *df* has one row
    per sample with component columns plus ``cancer_type``.
    """
    from epifolio.color_utils import distinct_palette, load_color_map

    cfg = resolve_analysis_cfg(cfg_or_path, analysis_name)

    csv_path = Path(
        cfg.get("GRANDSCATTER_CSV_FILENAME",
                 cfg.get("DEFAULT_CSV_FILENAME",
                          "data/all_H_component_contributions_k16.csv"))
    )
    npy_path = Path(
        cfg.get("GRANDSCATTER_NPY_PROPORTIONS_FILENAME",
                 cfg.get("NPY_PROPORTIONS_FILENAME",
                          "data/tcga_bulk_k16_H_proportions.npy"))
    )

    meta_df = _get_dataframe(csv_path)
    meta_df, sid_col = _standardize_sample_id_column(meta_df)
    H = np.load(npy_path)

    if selection is not None:
        meta_df = meta_df.iloc[selection].reset_index(drop=True)
        H = H[selection]

    comp_cols = _extract_component_columns(meta_df, sid_col)
    axis_fields, reorder = _canonicalize_component_columns(comp_cols)
    if reorder is not None:
        H = H[:, reorder]

    df = pd.DataFrame(H.astype(np.float32), columns=axis_fields)
    sample_ids = meta_df[sid_col].astype(str).tolist()
    df["cancer_type"] = _resolve_cancer_types(sample_ids, cfg)

    # Colors
    user_colors = load_color_map(cfg.get("JSON_FILENAME_CANCER_TYPE_COLORS"))
    unique = sorted(df["cancer_type"].unique())
    auto = distinct_palette(len(unique))
    label_colors = {ct: user_colors.get(ct, auto[i]) for i, ct in enumerate(unique)}

    return df, axis_fields, label_colors


def create_grandscatter_widget(
    cfg_or_path: str | Path | dict = "conf/config.json",
    *,
    selection: list[int] | None = None,
    analysis_name: str | None = None,
) -> "grandscatter.Scatter":
    """Return a ``grandscatter.Scatter`` anywidget for NMF proportions."""
    from grandscatter import Scatter

    df, axis_fields, label_colors = prepare_grandscatter_data(
        cfg_or_path, selection=selection, analysis_name=analysis_name,
    )
    return Scatter(
        df,
        axis_fields=axis_fields,
        label_field="cancer_type",
        label_colors=label_colors,
    )
