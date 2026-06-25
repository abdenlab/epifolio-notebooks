# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.20.2",
#     "anywidget>=0.9.18",
#     "hiplot>=0.1.33",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "pyarrow>=18.0.0",
#     "plotly>=6.1.2",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
# ///

import marimo

__generated_with = "0.21.1"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    lib_src = repo_root / "lib" / "src"
    if lib_src.exists():
        lib_src_str = str(lib_src)
        if lib_src_str not in sys.path:
            sys.path.insert(0, lib_src_str)
    return


@app.cell
def _():
    import anywidget
    import html
    import json
    import marimo as mo
    import pandas as pd
    import traitlets

    from epifolio import ASSETS
    from epifolio.data_utils import resolve_asset_path
    from epifolio.itcr_embedding_utils import (
        _categorical_color_lookup as categorical_color_lookup,
        _color_to_hiplot as color_to_hiplot,
        available_color_fields,
        build_metadata_views,
        build_strip_color_lookup,
        coerce_visible_features,
        get_feature_columns,
        load_cancer_color_map,
        load_clinical_metadata,
        load_embedding_sample_frame,
        load_metadata_tooltips,
        metadata_presentation,
        prepare_embedding_for_display,
        subset_by_sample_ids,
    )

    return (
        ASSETS,
        anywidget,
        available_color_fields,
        build_metadata_views,
        build_strip_color_lookup,
        categorical_color_lookup,
        coerce_visible_features,
        color_to_hiplot,
        get_feature_columns,
        html,
        json,
        load_cancer_color_map,
        load_clinical_metadata,
        load_embedding_sample_frame,
        load_metadata_tooltips,
        metadata_presentation,
        mo,
        pd,
        prepare_embedding_for_display,
        resolve_asset_path,
        subset_by_sample_ids,
        traitlets,
    )


@app.cell(hide_code=True)
def _(mo):
    title = mo.md("# Explore published ITCR PCA outcomes in HiPlot")
    title
    return


@app.cell
def _(ASSETS, json, pd, resolve_asset_path):
    strip_cfg = json.loads((ASSETS / "conf" / "upt_pub_nmf_config.json").read_text())
    strip_specs = strip_cfg.get("HEATMAP_STRIPS", [])
    strip_label_to_column = {
        spec.get("label", spec["column"]): spec["column"] for spec in strip_specs
    }
    strip_spec_by_column = {spec["column"]: spec for spec in strip_specs}
    strip_id_column = strip_cfg.get("STRIP_METADATA_ID_COLUMN", "submitter_id")
    strip_metadata = (
        pd.read_csv(resolve_asset_path(strip_cfg["STRIP_METADATA_FILENAME"], default_dir="data"))
        .drop_duplicates(subset=[strip_id_column])
        .rename(columns={strip_id_column: "patient_id"})
    )
    strip_unknown_color = strip_cfg.get("STRIP_UNKNOWN_COLOR", "#CCCCCC")
    return strip_label_to_column, strip_metadata, strip_spec_by_column, strip_unknown_color


@app.cell
def _():
    _EMBEDDINGS = "https://projects.abdenlab.org/itcr/epifolio/embeddings"
    pca_result_options = {
        "Canonical PCA": f"{_EMBEDDINGS}/tcga.atac.pca.sample.pq",
        "CN-aware adjusted PCA": f"{_EMBEDDINGS}/adjusted.pca.sample.pq",
        "CN-aware confounded PCA": f"{_EMBEDDINGS}/confounded.pca.sample.pq",
    }
    pca_result_labels = list(pca_result_options)
    return pca_result_labels, pca_result_options


@app.cell
def _(
    available_color_fields,
    load_cancer_color_map,
    load_clinical_metadata,
    load_metadata_tooltips,
    metadata_presentation,
    strip_label_to_column,
    strip_metadata,
):
    cancer_color_map = load_cancer_color_map()
    clinical_metadata = load_clinical_metadata().merge(
        strip_metadata,
        on="patient_id",
        how="left",
    )
    metadata_tooltips = load_metadata_tooltips()
    metadata_groups, base_tooltips, wrapped_columns, format_mapping = metadata_presentation()
    base_fields = available_color_fields(clinical_metadata)
    color_field_options = []
    for option in [
        "Cancer Type",
        *[field for field in base_fields if field != "Cancer Type"],
        *strip_label_to_column.keys(),
    ]:
        if option not in color_field_options:
            color_field_options.append(option)
    color_field_to_column = {field: field for field in base_fields} | strip_label_to_column
    return (
        base_tooltips,
        cancer_color_map,
        clinical_metadata,
        color_field_options,
        color_field_to_column,
        format_mapping,
        metadata_groups,
        metadata_tooltips,
        wrapped_columns,
    )


@app.cell
def _(color_field_options, mo, pca_result_labels):
    pca_result_dropdown = mo.ui.dropdown(
        options=pca_result_labels,
        value=pca_result_labels[0],
        label="PCA result set:",
    )
    color_by_dropdown = mo.ui.dropdown(
        options=color_field_options,
        value="Cancer Type",
        label="Color samples by:",
    )
    return color_by_dropdown, pca_result_dropdown


@app.cell
def _(load_embedding_sample_frame, pca_result_dropdown, pca_result_options):
    pca_embedding_df = load_embedding_sample_frame(
        pca_result_options[pca_result_dropdown.value]
    )
    return (pca_embedding_df,)


@app.cell
def _(
    build_strip_color_lookup,
    cancer_color_map,
    clinical_metadata,
    color_by_dropdown,
    color_field_to_column,
    pca_embedding_df,
    prepare_embedding_for_display,
    strip_spec_by_column,
    strip_unknown_color,
):
    selected_color_column = color_field_to_column[color_by_dropdown.value]
    pca_display_df, pca_label_colors = prepare_embedding_for_display(
        pca_embedding_df,
        clinical_metadata,
        selected_color_column,
        cancer_color_map,
    )
    if selected_color_column in strip_spec_by_column:
        pca_label_colors = build_strip_color_lookup(
            pca_display_df["Color Label"].tolist(),
            selected_color_column,
            strip_spec_by_column,
            cancer_color_map,
            strip_unknown_color,
        )
    return pca_display_df, pca_label_colors


@app.cell
def _(coerce_visible_features, get_feature_columns, pca_display_df):
    pca_feature_columns = get_feature_columns(pca_display_df)
    default_features = pca_feature_columns[: min(10, len(pca_feature_columns))]
    optional_metadata_columns = [
        column
        for column in [
            "tissue_or_organ_of_origin",
            "primary_diagnosis",
            "molecular_subtype",
            "patient_id",
        ]
        if column in pca_display_df.columns
    ]
    hiplot_column_options = [*pca_feature_columns, *optional_metadata_columns]
    default_hiplot_features = coerce_visible_features(
        hiplot_column_options,
        default_features,
        limit=10,
    )
    return default_hiplot_features, hiplot_column_options


@app.cell
def _(default_hiplot_features, hiplot_column_options, mo):
    hiplot_feature_picker = mo.ui.multiselect(
        options=hiplot_column_options,
        value=default_hiplot_features,
        label="Visible HiPlot columns:",
    )
    return (hiplot_feature_picker,)


@app.cell
def _(coerce_visible_features, hiplot_column_options, hiplot_feature_picker):
    selected_hiplot_features = coerce_visible_features(
        hiplot_column_options,
        list(hiplot_feature_picker.value),
        limit=10,
    )
    return (selected_hiplot_features,)


@app.cell
def _(html, json, pd, subset_by_sample_ids):
    selection_message_type = "pca-hiplot-selection"

    def create_hiplot_iframe_html(
        embedding_df: pd.DataFrame,
        visible_columns: list[str],
        colorby_column: str,
        label_colors: dict[str, str],
        cancer_color_map: dict[str, str],
        categorical_color_lookup,
        color_to_hiplot,
        *,
        height: int = 820,
        selection_message_type_override: str = selection_message_type,
    ) -> str:
        import hiplot as hip

        working_df = subset_by_sample_ids(embedding_df, None).copy()
        working_df["Sample / Cancer Type"] = (
            working_df["Sample ID"].astype(str)
            + " | "
            + working_df["Cancer Type"].fillna("Unknown").astype(str)
        )

        all_plot_columns = [
            "Sample ID",
            "Sample / Cancer Type",
            "patient_id",
            "Cancer Type",
            "Color Label",
            "tissue_or_organ_of_origin",
            "primary_diagnosis",
            "molecular_subtype",
            *visible_columns,
        ]
        available_columns = [
            column for column in all_plot_columns if column in working_df.columns
        ]
        plot_df = working_df[available_columns].copy()
        plot_df.insert(0, "uid", working_df["Sample ID"].astype(str))

        records = plot_df.where(pd.notna(plot_df), None).to_dict(orient="records")
        experiment = hip.Experiment.from_iterable(records)
        experiment.enabledDisplays = [hip.Displays.PARALLEL_PLOT]

        left_to_right = [
            "Cancer Type",
            *[column for column in visible_columns if column != "Cancer Type"],
        ]
        left_to_right = [
            column for column in left_to_right if column in plot_df.columns
        ]
        hidden_columns = [
            column for column in plot_df.columns if column not in left_to_right
        ]
        experiment.display_data(hip.Displays.PARALLEL_PLOT).update(
            {"order": list(reversed(left_to_right)), "hide": hidden_columns}
        )

        experiment.colorby = (
            colorby_column if colorby_column in plot_df.columns else "Cancer Type"
        )
        if "Color Label" in plot_df.columns:
            experiment.parameters_definition["Color Label"] = hip.ValueDef(
                value_type=hip.ValueType.CATEGORICAL,
                colors={
                    label: color_to_hiplot(color)
                    for label, color in label_colors.items()
                },
            )
        if "Cancer Type" in plot_df.columns:
            cancer_lookup = categorical_color_lookup(
                plot_df["Cancer Type"].fillna("Unknown").astype(str).tolist(),
                preferred_colors=cancer_color_map,
            )
            experiment.parameters_definition["Cancer Type"] = hip.ValueDef(
                value_type=hip.ValueType.CATEGORICAL,
                colors={
                    label: color_to_hiplot(color)
                    for label, color in cancer_lookup.items()
                },
            )
        if experiment.colorby in experiment.parameters_definition:
            experiment.parameters_definition[experiment.colorby].type = (
                hip.ValueType.CATEGORICAL
            )
        experiment.parameters_definition["Sample ID"].label_html = "Sample ID"
        experiment.parameters_definition["Sample / Cancer Type"].label_html = (
            "Sample | Cancer Type"
        )

        hiplot_html = experiment.to_html()
        selection_bridge_script = f"""/*ON_LOAD_SCRIPT_INJECT*/
        const selectionMessageType = {json.dumps(selection_message_type_override)};
        const bridgeState = {{
          selected_uids: [],
          filtered_uids: [],
        }};
        const normalizeSelectionPayload = (...args) => {{
          for (const arg of args) {{
            if (Array.isArray(arg)) {{
              return arg.map((value) => String(value));
            }}
          }}
          return [];
        }};
        const normalizeEventType = (fallback, ...args) => {{
          for (const arg of args) {{
            if (typeof arg === "string") {{
              return arg;
            }}
          }}
          return fallback;
        }};
        const getController = () => {{
          const root = document.querySelector(".hip_thm--light, .hip_thm--dark");
          if (!root) {{
            return null;
          }}
          const fiberKey = Object.keys(root).find((key) =>
            key.startsWith("__reactFiber$")
          );
          return fiberKey ? root[fiberKey]?.return?.stateNode ?? null : null;
        }};
        const normalizeBridgeSampleIds = (sampleIds) => {{
          const controller = getController();
          const totalRowCount = controller?.state?.rows_all_unfiltered?.length ?? 0;
          if (totalRowCount > 0 && sampleIds.length >= totalRowCount) {{
            return [];
          }}
          return sampleIds;
        }};

        function publishSelection(fallback, ...args) {{
          const eventType = normalizeEventType(fallback, ...args);
          const sampleIds = normalizeBridgeSampleIds(
            normalizeSelectionPayload(...args)
          );
          if (eventType === "filtered_uids") {{
            bridgeState.filtered_uids = sampleIds;
          }} else {{
            bridgeState.selected_uids = sampleIds;
          }}
          window.parent.postMessage({{
            type: selectionMessageType,
            bridge_state: {{
              selected_uids: [...bridgeState.selected_uids],
              filtered_uids: [...bridgeState.filtered_uids],
            }},
          }}, "*");
        }}

        Object.assign(options, {{
          onChange: {{
            selected_uids: (...args) => publishSelection("selected_uids", ...args),
            filtered_uids: (...args) => publishSelection("filtered_uids", ...args),
          }},
        }});
        """
        hiplot_html = hiplot_html.replace(
            "/*ON_LOAD_SCRIPT_INJECT*/",
            selection_bridge_script,
        )
        return (
            '<iframe style="width: 100%; height: '
            f'{height}px; border: 1px solid #d1d5db; border-radius: 12px; background: white;" '
            'sandbox="allow-scripts" '
            f'srcdoc="{html.escape(hiplot_html, quote=True)}"></iframe>'
        )

    return create_hiplot_iframe_html, selection_message_type


@app.cell
def _(anywidget, json, selection_message_type, traitlets):
    empty_selection_state = json.dumps(
        {"selected_uids": [], "filtered_uids": []},
        separators=(",", ":"),
    )

    class HiplotSelectionBridge(anywidget.AnyWidget):
        selection_state = traitlets.Unicode(empty_selection_state).tag(sync=True)
        message_type = traitlets.Unicode(selection_message_type).tag(sync=True)
        _esm = """
        function normalizeState(rawState) {
          const state =
            rawState && typeof rawState === "object" ? rawState : {};
          const normalizeList = (value) =>
            Array.isArray(value) ? value.map((item) => String(item)) : [];
          return {
            selected_uids: normalizeList(state.selected_uids),
            filtered_uids: normalizeList(state.filtered_uids),
          };
        }

        function render({ model, el }) {
          el.className = "hiplot-selection-bridge";
          const handleMessage = (event) => {
            const data = event?.data;
            if (!data || data.type !== model.get("message_type")) {
              return;
            }

            const nextValue = JSON.stringify(normalizeState(data.bridge_state));
            if (model.get("selection_state") === nextValue) {
              return;
            }

            model.set("selection_state", nextValue);
            model.save_changes();
          };
          window.addEventListener("message", handleMessage);
          return () => window.removeEventListener("message", handleMessage);
        }

        export default { render };
        """
        _css = """
        .hiplot-selection-bridge {
          display: none;
        }
        """

    return HiplotSelectionBridge, empty_selection_state


@app.cell
def _(HiplotSelectionBridge, empty_selection_state, mo):
    bridge = HiplotSelectionBridge(selection_state=empty_selection_state)
    get_selection_state, set_selection_state = mo.state(bridge.selection_state)
    bridge.observe(
        lambda _: set_selection_state(bridge.selection_state),
        names=["selection_state"],
    )
    bridge_view = mo.ui.anywidget(bridge)
    return bridge, bridge_view, get_selection_state


@app.cell
def _(
    cancer_color_map,
    categorical_color_lookup,
    color_to_hiplot,
    create_hiplot_iframe_html,
    mo,
    pca_display_df,
    pca_label_colors,
    selected_hiplot_features,
):
    hiplot_iframe = mo.Html(
        create_hiplot_iframe_html(
            embedding_df=pca_display_df,
            visible_columns=selected_hiplot_features,
            colorby_column="Color Label",
            label_colors=pca_label_colors,
            cancer_color_map=cancer_color_map,
            categorical_color_lookup=categorical_color_lookup,
            color_to_hiplot=color_to_hiplot,
            height=820,
        )
    )
    return (hiplot_iframe,)


@app.cell
def _(get_selection_state, json, pca_display_df):
    try:
        raw_selection_state = json.loads(get_selection_state() or "{}")
    except json.JSONDecodeError:
        raw_selection_state = {}

    if not isinstance(raw_selection_state, dict):
        raw_selection_state = {}

    valid_sample_ids = set(pca_display_df["Sample ID"].astype(str))
    total_sample_count = len(valid_sample_ids)

    def normalize_sample_ids(values):
        if not isinstance(values, list):
            return []
        return [
            sample_id
            for sample_id in values
            if isinstance(sample_id, str) and sample_id in valid_sample_ids
        ]

    selected_sample_ids = normalize_sample_ids(raw_selection_state.get("selected_uids", []))
    filtered_sample_ids = normalize_sample_ids(raw_selection_state.get("filtered_uids", []))

    if filtered_sample_ids and len(filtered_sample_ids) < total_sample_count:
        selected_hiplot_sample_ids = filtered_sample_ids
    elif selected_sample_ids and len(selected_sample_ids) < total_sample_count:
        selected_hiplot_sample_ids = selected_sample_ids
    else:
        selected_hiplot_sample_ids = []

    hiplot_selection_summary = {
        "selected_count": len(selected_sample_ids),
        "filtered_count": (
            len(filtered_sample_ids) if filtered_sample_ids else total_sample_count
        ),
        "is_filtered": bool(filtered_sample_ids)
        and len(filtered_sample_ids) < total_sample_count,
    }
    return hiplot_selection_summary, selected_hiplot_sample_ids


@app.cell
def _(build_metadata_views, pca_display_df, selected_hiplot_sample_ids):
    filtered_metadata_df, duplicate_patients_df, metadata_summary = build_metadata_views(
        pca_display_df,
        selected_hiplot_sample_ids,
    )
    return duplicate_patients_df, filtered_metadata_df, metadata_summary


@app.cell
def _(
    base_tooltips,
    bridge_view,
    color_by_dropdown,
    duplicate_patients_df,
    filtered_metadata_df,
    format_mapping,
    hiplot_feature_picker,
    hiplot_iframe,
    metadata_groups,
    metadata_summary,
    metadata_tooltips,
    mo,
    pca_result_dropdown,
    hiplot_selection_summary,
    selected_hiplot_sample_ids,
    wrapped_columns,
):
    header_tooltips = {**metadata_tooltips, **base_tooltips}

    def make_table(dataframe, columns, label):
        available_columns = [column for column in columns if column in dataframe.columns]
        return mo.ui.table(
            dataframe[available_columns],
            pagination=True,
            page_size=14,
            show_data_types=False,
            show_column_summaries=False,
            freeze_columns_left=[
                column for column in ["selection_index", "Sample ID", "patient_id"] if column in available_columns
            ],
            wrapped_columns=[
                column for column in wrapped_columns if column in available_columns
            ],
            header_tooltip={
                column: header_tooltips[column]
                for column in available_columns
                if column in header_tooltips
            },
            format_mapping={
                column: format_mapping[column]
                for column in available_columns
                if column in format_mapping
            },
            max_height=420,
            label=label,
        )

    metadata_tabs = mo.ui.tabs(
        {
            name: make_table(filtered_metadata_df, columns, f"{name} metadata")
            for name, columns in metadata_groups.items()
        }
        | {
            "All metadata": make_table(
                filtered_metadata_df,
                list(filtered_metadata_df.columns),
                "All PCA metadata",
            )
        }
    )

    duplicate_view = (
        mo.accordion(
            {
                "Patients with multiple ATAC-seq samples in this view": make_table(
                    duplicate_patients_df,
                    ["patient_id", "sample_count", "cancer_types", "sample_ids"],
                    "Repeated patients",
                )
            },
            lazy=True,
        )
        if not duplicate_patients_df.empty
        else mo.md("_No repeated patient IDs in the current view._")
    )

    selection_summary = (
        "Showing all samples in the current PCA result."
        if not selected_hiplot_sample_ids
        else f"Showing {len(selected_hiplot_sample_ids)} selected samples from HiPlot."
    )
    if hiplot_selection_summary["is_filtered"]:
        selection_summary = (
            f"Showing {len(selected_hiplot_sample_ids)} filtered samples from HiPlot "
            f"({hiplot_selection_summary['selected_count']} explicitly selected)."
        )

    mo.vstack(
        [
            mo.hstack([pca_result_dropdown, color_by_dropdown], widths=[1, 1]),
            hiplot_feature_picker,
            bridge_view,
            hiplot_iframe,
            mo.md("### Expanded metadata table"),
            mo.md(selection_summary),
            mo.hstack(
                [
                    mo.stat(metadata_summary["sample_count"], label="Samples shown", bordered=True),
                    mo.stat(metadata_summary["unique_patient_count"], label="Unique patients shown", bordered=True),
                    mo.stat(metadata_summary["cancer_type_count"], label="Cancer types shown", bordered=True),
                ],
                widths=[1, 1, 1],
            ),
            metadata_tabs,
            duplicate_view,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
