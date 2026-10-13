# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.14",
#     "numpy>=2.3,<3",
#     "pandas>=3.0.0",
#     "plotly>=6.1.2",
#     "anndata>=0.13.4,<0.14",
#     "zarr>=3.1.5,<4",
#     "scipy>=1.17.1,<1.18",
#     "fsspec>=2026.9.0",
#     "aiohttp>=3.14",
#     "epifolio @ git+https://github.com/abdenlab/epifolio-notebooks.git#subdirectory=lib",
# ]
#
# [tool.marimo.runtime]
# output_max_bytes = 20_000_000
# ///

# Tabbed redesign of CONTRIB_Analysis_marimo.py: run with `uv run marimo edit tabcopy_contrib.py`.
# Dependencies, including the local tcpa package, are declared in pyproject.toml.
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Patient Cohorts")


@app.cell
def _():
    import marimo as mo
    from epifolio.tcpa.loaders import load_tcga_rppa500
    from epifolio.tcpa.text import dataset_name, label_options, assay_table, display_labels, dataset_description, dataset_options, available_groupings, source_group_label

    return dataset_name, load_tcga_rppa500, mo, label_options, assay_table, display_labels, dataset_description, dataset_options, available_groupings, source_group_label


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Patient Cohorts

    **RPPA500 TCGA and contributed studies.**
    Explore protein measurements in one source dataset at a time. Contributed studies
    include experimental and non-patient material.
    Historical PanCan11, RPPA500 TCGA and other contributed studies have separate samples and assay panels.
    RPPA antibody assays measure proteins or specific protein modifications;
    source normalization and available annotations are in the About the Data tab.

    Group tests and correlation p-values are shown only for datasets that support
    exploratory inference. Heatmap z-scores use all selected observations
    as the reference population (using sample standard deviation).
    """)
    return


@app.cell
def _(mo, dataset_options):
    from epifolio.tcpa.loaders import load_dataset_catalog
    from epifolio.tcpa.text import catalog_display
    dataset_catalog = load_dataset_catalog(anchor=globals().get("__file__"))
    cohort_catalog = dataset_catalog.loc[dataset_catalog.resource_family.isin(['CONTRIB', 'RPPA500-TCGA'])]
    cohort_options = dataset_options(cohort_catalog)
    cohort_default = ('RPPA500-TCGA' if ((dataset_catalog.dataset_id == 'RPPA500-TCGA') & (dataset_catalog.artifact_status == 'validated')).any() else 'CONTRIB-TCPA00000002')
    cohort_selector = mo.ui.dropdown(options=cohort_options,
        value=next(label for label, key in cohort_options.items() if key == cohort_default), label="Source dataset")
    mo.vstack([cohort_selector, mo.accordion({'Available datasets': mo.ui.table(catalog_display(cohort_catalog))})])
    return cohort_selector, cohort_catalog


@app.cell
def _(mo, cohort_selector, cohort_catalog):
    selected_source = cohort_catalog.loc[cohort_catalog.dataset_id.eq(cohort_selector.value)].iloc[0]
    mo.stop(selected_source.artifact_status != 'validated', mo.md(f"**{selected_source.dataset_id} unavailable:** {selected_source.block_reason}"))
    selected_dataset_id = selected_source.dataset_id
    return (selected_dataset_id,)


@app.cell
def _(selected_dataset_id, mo, load_tcga_rppa500):
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from epifolio.tcpa.loaders import load_contrib

    source_groups, source_expr, source_obs, source_proteins = (load_tcga_rppa500(anchor=globals().get("__file__")) if selected_dataset_id == "RPPA500-TCGA" else load_contrib(anchor=globals().get("__file__"), dataset_id=selected_dataset_id))
    mo.stop(source_expr.empty, mo.md("No measured protein assays are available for this dataset."))
    return source_expr, source_obs, pd, px, go, np


@app.cell
def _(source_obs, mo):
    from epifolio.tcpa.observations import observation_metadata, observation_options, select_observations
    source_observation_metadata = observation_metadata(source_obs)
    observation_fields = observation_options(source_observation_metadata)
    observation_filters = mo.ui.dictionary({field: mo.ui.multiselect(
        options=sorted(source_observation_metadata[field].unique()), value=[], label=label)
        for field, label in observation_fields.items()})
    grouping_default = ('species' if 'species' in observation_fields and source_observation_metadata.species.nunique() > 1
                        else 'cancer_group' if 'cancer_group' in observation_fields else next(iter(observation_fields)))
    observation_grouping = mo.ui.dropdown(options={label:field for field,label in observation_fields.items()},
        value=observation_fields[grouping_default], label="Group observations by")
    mo.vstack([mo.md("## Select Observations\n\nThese filters apply to every plot and summary. Leave a filter empty to include all values. Concentration, time and replicate labels retain the source's units and meaning."),
        mo.accordion({"Filter observations": observation_filters.vstack()}), observation_grouping,
        mo.md("Subtype is the source cancer classification: molecular for BRCA and histological for UCEC. Unreported includes missing labels and cancers outside this bounded source coverage. BRCA subtype Normal is not a specimen type. Subtype grouping remains descriptive.") if "cancer_subtype" in observation_fields else mo.md("")])
    return observation_filters, observation_grouping, select_observations, observation_fields, source_observation_metadata, observation_metadata, observation_options, grouping_default


@app.cell
def _(source_expr, source_obs, observation_filters, observation_grouping, select_observations, observation_fields, selected_dataset_id, mo):
    df_expr, selected_obs, selected_metadata = select_observations(source_expr, source_obs,
        observation_filters.value, group_by=observation_grouping.value)
    mo.stop(df_expr.empty, mo.md(f"**{len(selected_obs):,} of {len(source_expr):,} source observations selected.** " + ("No measured observations match these filters. " if selected_obs.empty else "No finite assay measurements are available in this selection. ") + "Clear or change a filter to continue."))
    df_obs = selected_obs.assign(analysis_group=selected_metadata[observation_grouping.value])
    cancer_types = df_obs.analysis_group.unique().tolist()
    tumor_labels = {value:(source_expr.attrs.get('tumor_labels', {}).get(value, value)
                           if observation_grouping.value == 'cancer_group' else value)
                     for value in cancer_types}
    feature_labels = df_expr.attrs['feature_labels']
    sample_labels = {key:source_expr.attrs['sample_labels'][key] for key in df_expr.index}
    protein_list = df_expr.columns.tolist()
    grouping_label = observation_fields[observation_grouping.value]
    mo.md(f"Source ID: {selected_dataset_id} · **{len(df_expr):,} of {len(source_expr):,} source observations selected.** All plots below use this selection. Rows are retained individually; filters do not establish independent biological replicates.")
    return df_expr, df_obs, selected_metadata, cancer_types, tumor_labels, feature_labels, sample_labels, protein_list, grouping_label


@app.cell
def _():
    from epifolio.tcpa.stats import correlate, correlate_one_vs_all, population_summary, two_group_test, feature_summary, zscore
    from epifolio.tcpa.figures import correlation_figure, heatmap_figure, volcano_frame, explain_empty_volcano
    return correlate, correlate_one_vs_all, population_summary, two_group_test, feature_summary, zscore, correlation_figure, heatmap_figure, volcano_frame, explain_empty_volcano


@app.cell
def _(mo):
    # Remembers the open tab: the tabs cell re-runs (and would reset) whenever a control changes.
    get_tab, set_tab = mo.state("One Protein")
    return get_tab, set_tab


@app.cell(hide_code=True)
def _(mo, get_tab, set_tab, one_protein_tab, relationships_tab, compare_groups_tab, ranked_tab, heatmap_tab, about_tab):
    sections = mo.ui.tabs(
        {
            "One Protein": one_protein_tab,
            "Protein Relationships": relationships_tab,
            "Compare Groups": compare_groups_tab,
            "Ranked Proteins": ranked_tab,
            "Protein Heatmap": heatmap_tab,
            "About the Data": about_tab,
        },
        value=get_tab(),
        on_change=set_tab,
        lazy=True,
    )
    sections
    return (sections,)




@app.cell
def _(mo):
    get_assays, set_assays = mo.state({})
    return get_assays, set_assays


@app.cell(hide_code=True)
def _(get_assays, set_assays, label_options, feature_labels, mo, protein_list):
    # === Single Protein Explorer — controls ===
    # Menus are ordered by readable label and search assay names; values remain assay IDs.
    _assays = label_options(protein_list, feature_labels, sort=True)
    protein_dropdown = mo.ui.dropdown(
        options=_assays, value=feature_labels[get_assays().get("one")] if get_assays().get("one") in protein_list else next(iter(_assays)), searchable=True, allow_select_none=False, label="Protein / antibody assay",
        on_change=lambda value: set_assays(lambda previous: {**previous, "one": value}),
    )
    plot_type_dropdown = mo.ui.dropdown(
        options=['Box Plot by Selected Group', 'Histogram', 'Violin Plot'],
        value='Box Plot by Selected Group',
        label="Plot Type",
    )
    return plot_type_dropdown, protein_dropdown


@app.cell
def _(grouping_label, dataset_name, tumor_labels, feature_labels, sample_labels, df_expr, df_obs, mo, pd, plot_type_dropdown, protein_dropdown, px, np, feature_summary):
    def _view():
        protein = protein_dropdown.value
        plot_type = plot_type_dropdown.value

        if protein not in df_expr.columns:
            return mo.md(f"**Protein {feature_labels.get(protein, protein)} not found in expression data**")

        plot_df = pd.DataFrame({'Expression': df_expr[protein]})
        plot_df['Observation ID'] = plot_df.index.astype(str)
        plot_df['Observation'] = plot_df.index.map(sample_labels)
        if 'analysis_group' in df_obs.columns:
            plot_df['Tumor'] = df_obs['analysis_group'].reindex(plot_df.index).map(tumor_labels)
        if 'dataset_id' in df_obs.columns:
            plot_df['Dataset'] = dataset_name(df_expr.attrs['dataset_contract'])
        plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Expression'])
        missing_tumor_n = int(plot_df['Tumor'].isna().sum())
        plot_df['Tumor'] = plot_df['Tumor'].astype(object).fillna('Unreported')

        if plot_type == 'Box Plot by Selected Group' and 'Tumor' in plot_df.columns:
            fig = px.box(
                plot_df.dropna(subset=['Tumor']),
                x='Tumor', y='Expression', color='Tumor', hover_name='Observation', hover_data=['Observation ID'],
                title=f'{feature_labels[protein]} Expression by {grouping_label} (n={len(plot_df)})',
                labels={'Expression': f'{feature_labels[protein]} (existing units)', 'Tumor':grouping_label},
            )
            fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)
        elif plot_type == 'Histogram':
            if 'Tumor' in plot_df.columns:
                fig = px.histogram(
                    plot_df.dropna(subset=['Tumor']), x='Expression', color='Tumor', hover_data=['Observation ID'],
                    title=f'{feature_labels[protein]} Expression Distribution',
                    nbins=50, barmode='overlay', opacity=0.7, labels={'Tumor':grouping_label}
                )
            else:
                fig = px.histogram(plot_df, x='Expression', title=f'{feature_labels[protein]} Expression Distribution', nbins=50)
            fig.update_layout(height=500)
        elif plot_type == 'Violin Plot' and 'Tumor' in plot_df.columns:
            fig = px.violin(
                plot_df.dropna(subset=['Tumor']), x='Tumor', y='Expression', color='Tumor', hover_name='Observation', hover_data=['Observation ID'],
                box=True, labels={'Tumor':grouping_label}, title=f'{feature_labels[protein]} Expression (Violin) by {grouping_label}'
            )
            fig.update_layout(xaxis_tickangle=-45, height=500, showlegend=False)
        else:
            fig = px.histogram(plot_df, x='Expression', title=f'{feature_labels[protein]} Distribution')

        summary = feature_summary(plot_df[["Expression"]]).loc["Expression"]
        # Both tables describe the plotted population: observations with a finite value for this assay.
        statistics = pd.DataFrame([{
            'Protein / assay': feature_labels[protein], 'Assay ID': protein,
            'Finite n': len(plot_df), 'Mean': summary['mean'], 'SD': summary['std'],
            'Minimum': summary['min'], 'Maximum': summary['max'],
            'Finite n without group label': missing_tumor_n,
        }])
        group_counts = plot_df['Tumor'].value_counts().rename_axis(grouping_label).reset_index(name='Observations with finite assay value')
        return mo.vstack([
            fig,
            mo.ui.table(statistics, selection=None, pagination=False, label=f"{feature_labels[protein]} summary statistics",
                format_mapping={column: '{:.4f}'.format for column in ['Mean','SD','Minimum','Maximum']}),
            mo.ui.table(group_counts, selection=None, pagination=True, page_size=10, label=f"Finite observations by {grouping_label}"),
        ])

    def one_protein_tab():
        return mo.vstack([
            mo.md("Compare measured values across samples in the selected dataset. Type an assay name in the menu to search it."),
            mo.hstack([protein_dropdown, plot_type_dropdown], justify="start"),
            _view(),
        ])
    return (one_protein_tab,)


@app.cell(hide_code=True)
def _(grouping_label, get_assays, set_assays, label_options, feature_labels, cancer_types, df_expr, mo, protein_list):
    # === Protein-Protein Correlation — controls ===
    # Default to the two best-measured proteins so a real plot shows on load.
    _dense = df_expr.notna().sum().sort_values(ascending=False).index.tolist()
    protein_a_dropdown = mo.ui.dropdown(
        options=label_options(protein_list, feature_labels, sort=True), value=feature_labels[get_assays().get("a") if get_assays().get("a") in protein_list else _dense[0]], searchable=True, allow_select_none=False, label="Protein / antibody assay A",
        on_change=lambda value: set_assays(lambda previous: {**previous, "a": value}),
    )
    protein_b_dropdown = mo.ui.dropdown(
        options=label_options(protein_list, feature_labels, sort=True), value=feature_labels[get_assays().get("b") if get_assays().get("b") in protein_list else _dense[min(1, len(_dense)-1)]], searchable=True, allow_select_none=False, label="Protein / antibody assay B",
        on_change=lambda value: set_assays(lambda previous: {**previous, "b": value}),
    )
    corr_method_dropdown = mo.ui.dropdown(options=['pearson', 'spearman'], value='pearson', label="Method")
    color_by_dropdown = mo.ui.dropdown(options={'None':'None', grouping_label:'Selected grouping'}, value='None', label='Color by')
    return color_by_dropdown, corr_method_dropdown, protein_a_dropdown, protein_b_dropdown


@app.cell
def _(df_expr, protein_a_dropdown, corr_method_dropdown, correlate_one_vs_all):
    correlation_family = correlate_one_vs_all(df_expr, protein_a_dropdown.value,
        method=corr_method_dropdown.value, inference=df_expr.attrs["dataset_contract"].get("analysis_policy") == "exploratory")
    return (correlation_family,)


@app.cell
def _(mo, correlation_family, feature_labels, protein_b_dropdown):
    # Changing the manual B control clears the table override; it does not recorrect BH.
    _manual_b = protein_b_dropdown.value
    _results = correlation_family.reset_index().rename(columns={'feature':'Assay ID'})
    _results.insert(1, 'Protein / assay', _results['Assay ID'].map(feature_labels))
    correlation_results = mo.ui.table(_results, selection='single', pagination=True,
        label='One-versus-all results', max_height=350)
    return (correlation_results,)


@app.cell
def _(grouping_label, sample_labels, feature_labels, tumor_labels, color_by_dropdown, corr_method_dropdown, correlate, correlation_figure, df_expr, df_obs, mo, protein_a_dropdown, protein_b_dropdown, correlation_results, interaction_view):
    from epifolio.tcpa.text import analysis_status
    def _view():
        a = protein_a_dropdown.value
        _selected = correlation_results.value
        b = _selected.iloc[0]["Assay ID"] if len(_selected) else protein_b_dropdown.value
        if a not in df_expr or b not in df_expr:
            return mo.md("Select two available protein assays.")
        result = correlate(df_expr[a], df_expr[b], method=corr_method_dropdown.value, inference=df_expr.attrs["dataset_contract"].get("analysis_policy") == "exploratory")
        fig = correlation_figure(result, labels=sample_labels, groups=df_obs['analysis_group'].map(tumor_labels) if color_by_dropdown.value == 'Selected grouping' else None, x_label=feature_labels[a], y_label=feature_labels[b])
        if color_by_dropdown.value == 'Selected grouping':
            fig.update_layout(legend_title_text=grouping_label)
        return mo.vstack([fig, mo.md(f"{analysis_status(result['status'])}. Paired finite measurements: {result['n']}. " + ("P-values describe this exploratory pair only; interactive pair selection has no correction for multiple comparisons." if df_expr.attrs["dataset_contract"].get("analysis_policy") == "exploratory" else "This descriptive comparison shows no p-value or confidence interval because independence and normalization compatibility have not been established. Open One Protein for descriptive distributions."))])

    def relationships_tab():
        return mo.vstack([
            mo.md("Compare two assays using samples with finite measurements for both. Type an assay name in a menu to search it."),
            mo.hstack([protein_a_dropdown, protein_b_dropdown, corr_method_dropdown, color_by_dropdown], justify="start"),
            _view(),
            mo.accordion({'Discover related assays': mo.vstack([
                mo.md("One reference assay versus every other assay in the selected population. Search filters the displayed results only. BH q-values cover the complete eligible one-versus-all family when exploratory inference is permitted; they are not all-pairs q-values. Select a row to plot that pair; changing assay B clears the row override."),
                correlation_results,
            ])}),
            mo.accordion({'Curated interaction network': interaction_view}, lazy=True),
        ])
    return (relationships_tab,)


@app.cell
def _(mo):
    from epifolio.tcpa.loaders import load_interaction_reference, interaction_human_population
    from epifolio.tcpa.stats import interaction_network
    from epifolio.tcpa.figures import interaction_figure
    interaction_source, interaction_provenance = load_interaction_reference(anchor=globals().get("__file__"))
    network_search = mo.ui.text(label="Search network assays / targets / sites", debounce=True)
    network_scope = mo.ui.dropdown(options=['Reference assay neighborhood','All mapped assays'],
        value='Reference assay neighborhood', label="Network scope")
    network_limit = mo.ui.dropdown(options={'50 pairs':50,'100 pairs':100,'200 pairs':200},
        value='200 pairs', label="Draw limit")
    return interaction_source, interaction_provenance, interaction_human_population, interaction_network, interaction_figure, network_search, network_scope, network_limit


@app.cell
def _(mo, df_expr, selected_metadata, feature_labels, protein_a_dropdown, corr_method_dropdown,
      interaction_source, interaction_provenance, interaction_human_population,
      interaction_network, interaction_figure, network_search, network_scope, network_limit):
    def interaction_view():
        human, reason = interaction_human_population(df_expr.attrs['dataset_id'], selected_metadata)
        nodes, relations = interaction_network(df_expr, df_expr.attrs['feature_metadata'], interaction_source,
            human_population=human,
            reference=protein_a_dropdown.value if network_scope.value == 'Reference assay neighborhood' else None,
            search=network_search.value, method=corr_method_dropdown.value)
        fig, drawn = interaction_figure(nodes,relations,labels=feature_labels,edge_limit=network_limit.value)
        fig.update_layout(meta=dict(fig.layout.meta, search=network_search.value, scope=network_scope.value,
            reference=protein_a_dropdown.value, population_ids=df_expr.index.tolist(),
            source_sha256=interaction_provenance['source']['sha256']))
        message = (reason if not human else
            "No curated relations match these mapped assays. Change assay A, network scope or search." if relations.empty else
            f"{len(relations):,} source relations; {fig.layout.meta['total_pairs']:,} directed assay pairs. Drawing {len(drawn):,} pairs in stable assay-ID order.")
        return mo.vstack([
            mo.md("SIGNOR curated causal scaffold with descriptive correlations in the selected population. Each node is an antibody assay. Gene-level relations do not establish phosphosite-specific interactions."),
            mo.hstack([network_scope,network_search,network_limit],justify='start'),
            mo.md(message),
            fig if not relations.empty else mo.md("No network is drawn for this selection."),
            mo.md("Arrows follow the curated source direction. Color shows measured correlation: red nonnegative, blue negative, gray unavailable; width follows |r|. No p/q values are calculated. Search and the draw limit restrict display only; the complete matching source relations remain in the table."),
            mo.ui.table(relations[['assay_a','assay_b','ENTITYA','ENTITYB','SIGNOR_ID','EFFECT','MECHANISM','RESIDUE','TAX_ID','CELL_DATA','TISSUE_DATA','PMID','DIRECT','SCORE','r','finite_n','correlation_status','correlation_method']],selection=None,pagination=True,max_height=350,label='Curated source relations and measured correlations'),
            mo.accordion({'Assay mapping coverage':mo.ui.table(nodes.reset_index(),selection=None,pagination=True,max_height=350,label='Network assay mappings'),
                'Interaction source details':mo.vstack([mo.md("[SIGNOR](https://signor.uniroma2.it/) direct human protein relations, date/hash-pinned live export; [CC BY 4.0](https://signor.uniroma2.it/about/). Source citation: [Lo Surdo et al., SIGNOR4.0](https://doi.org/10.1093/nar/gkaf1237). Source effect, score, organism, residue, context and PMID are distinct from assay sites and measured r/n. Exact verified gene symbols only; all multi-target mappings remain separate. No orthology/alias transfer, physical-binding or HPRD parity is claimed."),mo.ui.table([interaction_provenance['source']],selection=None)])}),
        ])
    return (interaction_view,)


@app.cell(hide_code=True)
def _(label_options, tumor_labels, cancer_types, mo):
    # === Differential Expression (Volcano) — controls ===
    group_a_dropdown = mo.ui.dropdown(
        options=label_options(cancer_types, tumor_labels), value=tumor_labels[cancer_types[0]], label="Group A",
    )
    group_b_dropdown = mo.ui.dropdown(
        options=label_options(cancer_types, tumor_labels), value=tumor_labels[cancer_types[1] if len(cancer_types) > 1 else cancer_types[0]], label="Group B",
    )
    fc_threshold = mo.ui.slider(start=0.1, stop=2.0, step=0.1, value=0.5, label="|mean difference| cutoff", debounce=True)
    qval_threshold = mo.ui.dropdown(
        options={"0.0001": 0.0001, "0.001": 0.001, "0.01": 0.01, "0.05": 0.05, "0.1": 0.1},
        value="0.05",
        label="BH q cutoff",
    )
    return fc_threshold, group_a_dropdown, group_b_dropdown, qval_threshold


@app.cell
def _(feature_labels, tumor_labels, df_expr, df_obs, two_group_test, volcano_frame, explain_empty_volcano, fc_threshold, group_a_dropdown, group_b_dropdown, mo, np, px, qval_threshold):
    def _build_comparison():
        if df_expr.attrs["dataset_contract"].get("analysis_policy") != "exploratory":
            return None, None, {}, mo.md("**Descriptive cohort:** group inference is disabled because independence and normalization compatibility have not been established. Open One Protein for descriptive distributions.")
        a, b = group_a_dropdown.value, group_b_dropdown.value
        if a == b:
            return None, None, {}, mo.md("Select different groups for comparison.")
        if a not in set(df_obs['analysis_group'].dropna()) or b not in set(df_obs['analysis_group'].dropna()):
            return None, None, {}, mo.md("Selected group has no samples.")
        results = two_group_test(df_expr, df_obs['analysis_group'], a, b)
        plotted = volcano_frame(results, alpha=qval_threshold.value, min_effect=fc_threshold.value)
        from epifolio.tcpa.text import analysis_status
        plotted['display_protein'] = plotted.index.map(feature_labels)
        plotted['result_label'] = plotted['status'].map(analysis_status)
        plotted['comparison_label'] = plotted['classification'].map({'higher_a': 'Higher in A', 'higher_b': 'Higher in B', 'not_selected': 'Below cutoffs', 'invalid': 'Not testable'})
        plotted['assay_id'] = plotted.index
        fig = px.scatter(plotted, x='mean_diff', y='neg_log10_q', color='comparison_label', color_discrete_map={'Higher in A':'red','Higher in B':'blue','Below cutoffs':'gray','Not testable':'lightgray'}, hover_name='display_protein', custom_data=['assay_id'], hover_data=['mean_a','mean_b','n_a','n_b','p_value','q_value','result_label'], title=f"Welch comparison: {tumor_labels.get(a,a)} vs {tumor_labels.get(b,b)}", labels={'mean_diff':'Mean difference in existing units (A − B)','neg_log10_q':'−log₁₀(BH q-value)', 'comparison_label':'Comparison', 'result_label':'Result', 'mean_a':'Mean A', 'mean_b':'Mean B', 'n_a':'Finite n, A', 'n_b':'Finite n, B', 'p_value':'p-value', 'q_value':'BH q-value'})
        fig.add_hline(y=-np.log10(qval_threshold.value), line_dash='dash')
        fig.add_vline(x=fc_threshold.value, line_dash='dash')
        fig.add_vline(x=-fc_threshold.value, line_dash='dash')
        explain_empty_volcano(fig, results)
        fig.update_layout(dragmode='lasso')
        point_ids = {(curve, point): row[0] for curve, trace in enumerate(fig.data)
                     for point, row in enumerate(trace.customdata)}
        return mo.ui.plotly(fig), plotted, point_ids, None

    comparison_plot, comparison_frame, comparison_point_ids, comparison_message = _build_comparison()
    return comparison_plot, comparison_frame, comparison_point_ids, comparison_message


@app.cell
def _(assay_table, feature_labels, mo, comparison_plot, comparison_frame, comparison_point_ids, comparison_message, group_a_dropdown, group_b_dropdown, fc_threshold, qval_threshold):
    def _view():
        if comparison_frame is None:
            return comparison_message
        # Marimo sends trace/point indices, not customdata; map them to exact assay IDs.
        selected_ids = {comparison_point_ids[(point['curveNumber'], point.get('pointIndex', point.get('pointNumber')))]
                        for point in comparison_plot.value}
        shown = comparison_frame.loc[comparison_frame.index.isin(selected_ids)] if selected_ids else comparison_frame
        eligible = comparison_frame['status'].eq('ok') & comparison_frame['q_value'].notna()
        return mo.vstack([
            comparison_plot,
            mo.md(f"Welch tests require at least 3 finite samples per group; BH correction covers all {int(eligible.sum())} eligible assays before display cutoffs. Lasso or box-select points to filter the table; clear the selection to show all assays. Showing {len(shown)} of {len(comparison_frame)} assays."),
            mo.ui.table(assay_table(shown, feature_labels, {'n_a':'Finite n, A','n_b':'Finite n, B','mean_diff':'Mean difference','p_value':'p-value','q_value':'BH q-value','result_label':'Result'}), selection=None, pagination=True, label='Compared protein assays'),
        ])

    def compare_groups_tab():
        return mo.vstack([
            mo.md("Compare mean protein measurements between two groups when the dataset supports exploratory testing."),
            mo.hstack([group_a_dropdown, group_b_dropdown, fc_threshold, qval_threshold], justify="start"),
            _view(),
        ])
    return (compare_groups_tab,)


@app.cell(hide_code=True)
def _(mo):
    # === Protein Ranking Explorer — controls ===
    rank_by_dropdown = mo.ui.dropdown(options=['range', 'variance', 'mean'], value='range', label="Rank by")
    top_n_slider = mo.ui.slider(start=5, stop=50, step=5, value=20, label="Proteins shown", debounce=True)
    return rank_by_dropdown, top_n_slider


@app.cell
def _(assay_table, feature_labels, df_expr, mo, feature_summary, px, rank_by_dropdown, top_n_slider):
    def _view():
        by = rank_by_dropdown.value
        rankings = feature_summary(df_expr).sort_values(by, ascending=False).head(top_n_slider.value)
        from epifolio.tcpa.text import analysis_status
        rankings['display_protein'] = rankings.index.map(feature_labels)
        rankings['status'] = rankings['status'].map(analysis_status)
        fig = px.bar(rankings, x='display_protein', y=by, color=by, hover_data=['n','mean','std','variance','range','status'], title=f"Proteins ranked by {by}", labels={'display_protein':'Protein / assay','n':'Finite measurements','std':'Standard deviation','status':'SD / variance status'})
        return mo.vstack([fig, mo.ui.table(assay_table(rankings, feature_labels, {'n':'Finite measurements','mean':'Mean','std':'Standard deviation','variance':'Variance','range':'Range','min':'Minimum','max':'Maximum','status':'SD / variance status'}, rank=True), selection=None, pagination=True, label='Ranked protein assays')])

    def ranked_tab():
        return mo.vstack([
            mo.md("Find assays with the largest range, variance or mean across the selected observations."),
            mo.hstack([rank_by_dropdown, top_n_slider], justify="start"),
            _view(),
        ])
    return (ranked_tab,)


@app.cell(hide_code=True)
def _(mo):
    # === Expression Heatmap — controls ===
    n_proteins_slider = mo.ui.slider(start=10, stop=100, step=10, value=30, label="Proteins shown", debounce=True)
    n_samples_slider = mo.ui.slider(start=50, stop=500, step=50, value=100, label="Samples shown", debounce=True)
    colorscale_dropdown = mo.ui.dropdown(
        options=['RdBu_r', 'Viridis', 'Plasma', 'Inferno', 'Turbo', 'Picnic'], value='RdBu_r', label="Colors",
    )
    return colorscale_dropdown, n_proteins_slider, n_samples_slider


@app.cell
def _(feature_labels, sample_labels, colorscale_dropdown, df_expr, feature_summary, zscore, heatmap_figure, n_proteins_slider, n_samples_slider, mo, hm_order, hm_cut):
    def _view():
        features = feature_summary(df_expr)['variance'].dropna().sort_values(ascending=False).head(n_proteins_slider.value).index
        heatmap_z = zscore(df_expr.loc[:, features], axis=0, ddof=1).iloc[:n_samples_slider.value]
        fig_heatmap = heatmap_figure(heatmap_z, cluster_mode=hm_order.value, cluster_cut=hm_cut.value, raw_data=df_expr.loc[heatmap_z.index, features], feature_labels=feature_labels, sample_labels=sample_labels, colorscale=colorscale_dropdown.value, title='Protein patterns across samples', colorbar_title='Z-score')
        return mo.vstack([fig_heatmap, mo.md(f"Colors show z-scores relative to all {len(df_expr)} selected observations (using sample standard deviation). The display uses the first {len(heatmap_z)} selected observations. Current order preserves source order; optional local clustering reorders only the displayed matrix. Assays with constant values or fewer than two finite measurements have no z-score.")])

    def heatmap_tab():
        return mo.vstack([
            mo.md("Compare the most variable assays across the first selected number of observations. Heatmap order controls optional local clustering."),
            mo.hstack([n_proteins_slider, n_samples_slider, colorscale_dropdown], justify="start"),
            mo.accordion({'Optional local clustering': mo.vstack([mo.hstack([hm_order,hm_cut],justify='start'),mo.md("Clusters use the displayed matrix with Euclidean distance and average linkage. Only complete entities are clustered; incomplete entities remain at the end with no cluster. No missing values are filled. The cut requests at most this many clusters; ties can give fewer. This local method is separate from the portal's unspecified clustering.")])}),
            _view(),
        ])
    return (heatmap_tab,)



@app.cell(hide_code=True)
def _(mo, dataset_description, df_expr, df_obs, population_summary):
    from epifolio.tcpa.text import assay_display

    from epifolio.tcpa.text import phenotype_input_gates, phenotype_gate_description

    def about_tab():
        _counts, _finite = population_summary(df_expr, df_obs['analysis_group'])
        return mo.vstack([mo.md(dataset_description(df_expr)),
            mo.accordion({'Analyses requiring additional inputs': mo.vstack([mo.md(phenotype_gate_description('patient')), mo.ui.table(phenotype_input_gates('patient'), selection=None, pagination=False, label='Phenotype input gates')])}),
            mo.accordion({"Selected cohort summary": mo.vstack([mo.md("Counts describe the selected observations and grouping; finite n counts measurements, not whether an assay was performed."),mo.ui.table(_counts),mo.ui.table(_finite)]), "Selected observation metadata": mo.ui.table(df_obs.reset_index().rename(columns=lambda name: name.replace("tumour", "tumor"))), "Protein and Antibody Details": mo.ui.table(assay_display(df_expr.attrs['panel_metadata']))})])
    return (about_tab,)


@app.cell(hide_code=True)
def _(mo):
    hm_order = mo.ui.dropdown(options={'Current order':'source','Local row clustering':'rows','Local assay clustering':'columns','Local rows and assays':'both'}, value='Current order', label='Heatmap order')
    hm_cut = mo.ui.slider(start=1,stop=10,value=3,label='Maximum local clusters',debounce=True)
    return hm_order, hm_cut


if __name__ == "__main__":
    app.run()
