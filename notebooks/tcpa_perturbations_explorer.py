# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.14",
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

# Tabbed redesign of CPPA_Analysis_marimo.py: run with `uv run marimo edit tabcopy_cppa.py`.
# Dependencies, including the local tcpa package, are declared in pyproject.toml.
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Perturbed Proteomics")


@app.cell
def _():
    import marimo as mo
    from epifolio.tcpa.text import label_options, assay_table, display_labels, dataset_description, dataset_options, available_groupings, source_group_label, cppa_reconstruction_description, catalog_display, condition_display, assay_display

    return mo, label_options, assay_table, display_labels, dataset_description, dataset_options, available_groupings, source_group_label, cppa_reconstruction_description, catalog_display, condition_display, assay_display


@app.cell(hide_code=True)
def _(source_response, source_treatment, mo):
    mo.vstack([mo.md(f"""
    # Perturbed Proteomics

    Explore how protein measurements change after treatment in the Cancer Perturbed Proteomics Atlas (CPPA).
    **Source experiment: {len(source_response):,} conditions · {source_treatment['cell_line_name'].nunique()} cell lines · {source_treatment['compound_name_1'].nunique()} primary compounds.**

    Δp is the published, replicate-averaged change from control to treatment.
    Positive values indicate an increase; negative values indicate a decrease.
    These are protein responses, not measures of drug sensitivity.
    The portal's own sensitivity analyses are in the Published Drug Analyses tab.
    """), mo.md(f"{source_response.attrs['coverage']['measured_features']} of {source_response.attrs['coverage']['total_features']} assays have measurements. Exposure time is unreported for {int(source_treatment.time.fillna('').astype(str).str.strip().eq('').sum())} conditions; primary dose for {int(source_treatment.dosage_1.fillna('').astype(str).str.strip().eq('').sum())}."),
    mo.accordion({'How to read these responses': mo.md("Values retain the published units and are not filled in or recalculated for display. Conditions can share controls, so these plots describe patterns without significance tests. Treatment and control checks are in the About the Data tab; an unresolved check does not remove the published response.")})])
    return


@app.cell
def _(mo, dataset_options, catalog_display):
    from epifolio.tcpa.loaders import load_dataset_catalog
    dataset_catalog = load_dataset_catalog(anchor=globals().get("__file__"))
    cppa_catalog = dataset_catalog.loc[dataset_catalog.resource_family.isin(['CPPA'])]
    cppa_options = dataset_options(cppa_catalog)
    cppa_default = 'CPPA-set120'
    cppa_selector = mo.ui.dropdown(options=cppa_options,
        value=next(label for label, key in cppa_options.items() if key == cppa_default), label="Source dataset")
    mo.vstack([cppa_selector, mo.accordion({'Available experiments': mo.ui.table(catalog_display(cppa_catalog))})])
    return cppa_selector, cppa_catalog


@app.cell
def _(mo, cppa_selector, cppa_catalog):
    selected_source = cppa_catalog.loc[cppa_catalog.dataset_id.eq(cppa_selector.value)].iloc[0]
    mo.stop(selected_source.artifact_status != 'validated', mo.md(f"**{selected_source.dataset_id} unavailable:** {selected_source.block_reason}"))
    selected_dataset_id = selected_source.dataset_id
    return (selected_dataset_id,)


@app.cell
def _(selected_dataset_id, mo):
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from epifolio.tcpa.loaders import load_cppa

    cell_lines, compounds, source_response, source_treatment, resources, times, source_long, source_proteins = load_cppa(anchor=globals().get("__file__"), set_id=selected_dataset_id.removeprefix("CPPA-"))
    mo.stop(source_response.empty, mo.md("No measured responses are available for this experiment."))
    return source_response, source_treatment, source_long, pd, px, go


@app.cell
def _(source_treatment, source_response, label_options, mo):
    from epifolio.tcpa.observations import cppa_condition_metadata, cppa_condition_options, select_cppa_conditions
    condition_metadata = cppa_condition_metadata(source_treatment)
    condition_fields = cppa_condition_options(condition_metadata)
    condition_filters = mo.ui.dictionary({field: mo.ui.multiselect(
        options=({f"{source_response.attrs['condition_labels'][uid]} [{uid}]":uid for uid in condition_metadata.index}
                 if field == 'condition_uid' else sorted(condition_metadata[field].unique())),
        value=[], label=label) for field,label in condition_fields.items()})
    mo.vstack([mo.md("### Select Conditions\n\nThese filters apply to every tab. Empty filters include all values. Each compound/dose slot belongs to a complete regimen; selecting a slot retains the whole condition. Missing design values remain Unreported."),
        mo.accordion({'Filter conditions':condition_filters.vstack()})])
    return condition_filters, select_cppa_conditions


@app.cell
def _(source_response, source_treatment, condition_filters, select_cppa_conditions, selected_dataset_id, mo):
    df_cppa_wide, df_treatment, selected_condition_metadata = select_cppa_conditions(
        source_response, source_treatment, condition_filters.value)
    mo.stop(df_cppa_wide.empty, mo.md(f"**{len(df_treatment):,} of {len(source_response):,} source conditions selected.** " +
        ("No conditions match these filters. " if df_treatment.empty else "No finite response measurements are available in this selection. ") + "Clear or change a condition filter to continue."))
    feature_labels = df_cppa_wide.attrs['feature_labels']
    condition_labels = df_cppa_wide.attrs['condition_labels']
    protein_list = df_cppa_wide.columns.tolist()
    mo.md(f"Source ID: {selected_dataset_id} · **{len(df_treatment):,} of {len(source_response):,} source conditions selected.** Every tab below uses these condition UIDs. Published Δp is unchanged; conditions can share controls.")
    return df_cppa_wide, df_treatment, feature_labels, condition_labels, protein_list, selected_condition_metadata


@app.cell
def _(mo):
    get_choices, set_choices = mo.state({})
    return get_choices, set_choices


@app.cell
def _():
    from epifolio.tcpa.stats import feature_summary, correlate, correlate_one_vs_all, population_summary
    from epifolio.tcpa.figures import heatmap_figure, correlation_figure, response_curve_figure

    return feature_summary, correlate, correlate_one_vs_all, population_summary, heatmap_figure, correlation_figure, response_curve_figure


@app.cell
def _(mo):
    # Remembers the open tab: the tabs cell re-runs (and would reset) whenever a control changes.
    get_tab, set_tab = mo.state("Browse Treatments")
    return get_tab, set_tab


@app.cell(hide_code=True)
def _(mo, get_tab, set_tab, browse_tab, response_changes_tab, vary_together_tab, response_map_tab, compare_conditions_tab, drug_analyses_tab, about_tab):
    sections = mo.ui.tabs(
        {
            "Browse Treatments": browse_tab,
            "Largest Response Changes": response_changes_tab,
            "Responses That Vary Together": vary_together_tab,
            "Response Map": response_map_tab,
            "Compare Two Conditions": compare_conditions_tab,
            "Published Drug Analyses": drug_analyses_tab,
            "About the Data": about_tab,
        },
        value=get_tab(),
        on_change=set_tab,
        lazy=True,
    )
    sections
    return (sections,)




@app.cell(hide_code=True)
def _(cppa_reconstruction_description, condition_display, df_cppa_wide, df_treatment, mo):
    reconstruction_columns = [column for column in ['UID', 'cell_line_name', 'compound_name_1', 'treated_n', 'control_n', 'reconstruction_status', 'reconstruction_reason'] if column in df_treatment]
    reconstruction_table = condition_display(df_treatment[reconstruction_columns])

    def response_checks_panel():
        return mo.vstack([
            mo.md(cppa_reconstruction_description(df_cppa_wide)),
            mo.accordion({'Response checks': mo.vstack([mo.md('Candidate controls are groups suggested by the published sample information. For unresolved responses, the original control group is not confirmed; unknown counts are shown as Unreported.'), mo.ui.table(reconstruction_table)])}),
        ])
    return (response_checks_panel,)


@app.cell
def _(source_group_label, condition_display, df_treatment, mo, px):
    def _view():
        filtered = df_treatment.copy()

        fig = None
        if len(filtered) > 0 and 'compound_name_1' in filtered.columns and 'cell_line_name' in filtered.columns:
            summary = filtered.groupby(['compound_name_1', 'cell_line_name']).size().reset_index(name='count')
            if len(summary) > 0:
                fig = px.treemap(
                    summary, path=['compound_name_1', 'cell_line_name'], values='count',
                    labels={'compound_name_1': 'Primary compound', 'cell_line_name': 'Cell line', 'count': 'Conditions'},
                    title=f'Treatment coverage · {len(filtered)} conditions',
                    color='count', color_continuous_scale='Blues'
                )
                fig.update_layout(height=500)

        summary = [{'Conditions shown': len(filtered), 'Conditions in shared selection': len(df_treatment)}]
        for column, label in [('compound_name_1', 'Primary compound labels'), ('cell_line_name', 'Cell-line labels')]:
            if column in filtered.columns:
                summary[0][label] = int(filtered[column].nunique())
        tables = [mo.ui.table(summary, selection=None, pagination=False, label='Treatment coverage summary')]
        for column, label in [('compound_name_1', 'Primary compound'), ('cell_line_name', 'Cell line'), ('time', 'Time point')]:
            if column in filtered.columns:
                counts = filtered[column].value_counts(dropna=False).rename_axis(label).reset_index(name='Conditions')
                counts[label] = counts[label].map(source_group_label)
                tables.append(mo.ui.table(counts, selection=None, pagination=True, page_size=5, label=f'Conditions by {label.lower()}'))

        display_cols = ['UID', 'set', 'display_label', 'compound_name_1', 'cell_line_name', 'dosage_1', 'compound_name_2', 'dosage_2', 'compound_name_3', 'dosage_3', 'compound_name_4', 'dosage_4', 'time', 'resource', '2D_3D', 'stimuli', 'treated_n', 'control_n', 'reconstruction_status', 'reconstruction_reason']
        display_cols = [c for c in display_cols if c in filtered.columns]
        preview = condition_display(filtered[display_cols])
        text = mo.vstack(tables)
        details = mo.accordion({'Matching conditions': mo.ui.table(preview)})
        return mo.vstack([fig, text, details]) if fig is not None else mo.vstack([text, details])

    def browse_tab():
        return mo.vstack([
            mo.md('Compare selected condition counts by primary compound and cell line. The complete matching treatment table retains every regimen slot.'),
            _view(),
        ])
    return (browse_tab,)


@app.cell(hide_code=True)
def _(mo):
    response_metric = mo.ui.dropdown(options={'Absolute mean Δp':'abs_mean','Response range':'range','Response variance':'variance'}, value='Absolute mean Δp', label='Ranking quantity')
    response_top_n = mo.ui.slider(start=10, stop=50, step=5, value=25, label='Proteins shown', debounce=True)
    return response_metric, response_top_n


@app.cell(hide_code=True)
def _(mo):
    response_view = mo.ui.dropdown(options=['Population bars','Full condition curve','Treatment connectivity'],value='Population bars',label='Response view')
    return (response_view,)


@app.cell(hide_code=True)
def _(mo, df_cppa_wide, condition_labels, label_options, get_choices, set_choices):
    # The menu is ordered by label; the fallback choice stays the first condition in source order.
    response_condition = mo.ui.dropdown(options=label_options(df_cppa_wide.index,condition_labels,sort=True),
        value=condition_labels[get_choices().get('curve') if get_choices().get('curve') in df_cppa_wide.index else df_cppa_wide.index[0]],searchable=True,allow_select_none=False,label='Response condition',
        on_change=lambda value: set_choices(lambda previous: {**previous,'curve':value}))
    return (response_condition,)


@app.cell
def _(mo, df_cppa_wide, response_condition, feature_labels, feature_summary):
    _summary = feature_summary(df_cppa_wide.loc[[response_condition.value]])
    _ranked = _summary.loc[_summary.n.gt(0)].sort_values('mean',ascending=False,kind='stable')
    _table = _ranked.reset_index().rename(columns={'feature':'Assay ID','mean':'Published Δp','n':'Finite n'})[['Assay ID','Published Δp','Finite n']]
    _table.insert(0,'Rank',range(1,len(_table)+1))
    _table.insert(2,'Protein / assay',_table['Assay ID'].map(feature_labels))
    response_results = mo.ui.table(_table,selection='single',
        pagination=True,label='Full condition responses',max_height=350)
    return (response_results,)


@app.cell
def _(assay_table, display_labels, feature_labels,
    df_cppa_wide,
    df_treatment,
    go,
    mo,
    feature_summary,
    response_metric,
    response_top_n, response_view, response_condition, response_results, response_curve_figure, selected_dataset_id, connectivity_view,
):
    def _view():
        if response_view.value == 'Treatment connectivity':
            return connectivity_view()
        if response_view.value == 'Full condition curve':
            uid = response_condition.value
            selected = response_results.value
            assay = selected.iloc[0]['Assay ID'] if len(selected) else None
            curve, _table = response_curve_figure(df_cppa_wide.loc[uid],feature_labels,highlight=assay)
            if _table.empty:
                return mo.vstack([response_condition,mo.md('No finite published responses for this condition. Choose another curve condition or change the shared condition filters.')])
            detail = mo.md("Select a response row to inspect its individual sample measurements.")
            if assay is not None:
                from epifolio.tcpa.loaders import load_cppa_sample_detail
                samples, diagnostic = load_cppa_sample_detail(uid,assay,anchor=globals().get('__file__'),set_id=selected_dataset_id.removeprefix('CPPA-'))
                detail = mo.vstack([mo.md(f"Condition {uid} · Assay {assay} · Published Δp: {df_cppa_wide.loc[uid,assay]:.9g}. Reconstruction: **{diagnostic.reconstruction_status}**. {diagnostic.reconstruction_reason}"),
                    mo.md("Individual P0/P1 source measurements are shown by persisted membership. Unresolved controls are candidates; no candidate baseline mean or reconstructed response is calculated here. Source integrity and reconstruction status are separate checks."),mo.ui.table(samples,label='Selected individual samples',selection=None,pagination=True,max_height=350)])
            return mo.vstack([mo.hstack([response_condition],justify='start'),curve,response_results,
                mo.accordion({'Individual sample measurements':detail})])
        conditions = df_cppa_wide.index.tolist()
        top_n = response_top_n.value

        if not conditions:
            return mo.md("⚠️ **Select at least one condition.**")

        from epifolio.tcpa.text import analysis_status
        rankings = feature_summary(df_cppa_wide)
        rankings['abs_mean'] = rankings['mean'].abs()
        metric = response_metric.value
        metric_label = {'abs_mean':'Absolute mean Δp', 'range':'Response range', 'variance':'Response variance'}[metric]
        rankings = rankings.loc[(rankings['n'] > 0) & rankings[metric].notna()].sort_values(metric, ascending=False, kind='stable').head(top_n)
        if rankings.empty:
            return mo.md("No finite values support this ranking quantity. For one condition, choose Absolute mean Δp; variance requires at least two finite measurements.")

        colors = ['red' if x > 0 else 'blue' for x in rankings['mean']]
        fig = go.Figure(go.Bar(
            x=display_labels(rankings.index, feature_labels), y=rankings['mean'], marker_color=colors,
            text=rankings['mean'].round(3), textposition='outside',
            customdata=rankings.reset_index()[[rankings.index.name or 'index','n',metric]].to_numpy(),
            hovertemplate='%{x}<br>Assay ID: %{customdata[0]}<br>Finite n: %{customdata[1]}<br>Ranking value: %{customdata[2]}<br>Mean Δp: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title=f'Mean response · {len(conditions)} conditions',
            xaxis_title='Protein', yaxis_title='Mean Δp',
            xaxis_tickangle=-45, height=500, showlegend=False
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray')

        fig2 = go.Figure(go.Bar(
            x=display_labels(rankings.index, feature_labels), y=rankings[metric], marker_color='teal',
            text=rankings[metric].round(3), textposition='outside'
        ))
        fig2.update_layout(
            title=metric_label,
            xaxis_title='Protein', yaxis_title=metric_label,
            xaxis_tickangle=-45, height=400, showlegend=False
        )

        return mo.vstack([fig, fig2, mo.ui.table(assay_table(rankings.assign(status=rankings['status'].map(analysis_status)), feature_labels, {'n':'Finite conditions','mean':'Mean Δp','abs_mean':'Absolute mean Δp','std':'Δp standard deviation','variance':'Response variance','range':'Response range','min':'Minimum Δp','max':'Maximum Δp','status':'SD / variance status'}, rank=True), selection=None, pagination=True, label='Ranked response assays')])

    def response_changes_tab():
        return mo.vstack([
            mo.md("Rank the shared selected population by the declared quantity. The first chart shows signed mean Δp; the second shows the ranking value. Absolute mean Δp gives absolute Δp for one condition. Display cutoffs do not alter published values. Full condition curve shows every finite assay response, sorted by signed Δp for one selected condition; row selection highlights an assay and opens its sample detail. Treatment connectivity shows one complete regimen linked to measured assays. Ranking quantity and Proteins shown apply to Population bars. Type a condition name in the Response condition menu to search it."),
            mo.hstack([response_view, response_metric, response_top_n], justify="start"),
            _view(),
        ])
    return (response_changes_tab,)


@app.cell
def _(mo):
    from epifolio.tcpa.stats import regimen_response_edges
    from epifolio.tcpa.figures import regimen_connectivity_figure
    connectivity_search = mo.ui.text(label='Search connectivity assays / targets / sites', debounce=True)
    connectivity_limit = mo.ui.dropdown(options={'25 assays':25,'50 assays':50,'100 assays':100},
        value='50 assays',label='Maximum drawn assays')
    return regimen_response_edges, regimen_connectivity_figure, connectivity_search, connectivity_limit


@app.cell
def _(mo, df_cppa_wide, df_treatment, feature_labels, selected_dataset_id, response_condition,
      regimen_response_edges, regimen_connectivity_figure, connectivity_search, connectivity_limit):
    def connectivity_view():
        from epifolio.tcpa.text import condition_display, source_group_label
        uid = response_condition.value
        condition = df_treatment.set_index('UID',drop=False).loc[uid]
        edges = regimen_response_edges(df_cppa_wide.loc[uid],condition,
            df_cppa_wide.attrs['feature_metadata'],set_id=selected_dataset_id.removeprefix('CPPA-'))
        query = connectivity_search.value.strip()
        matching = edges
        if query and not edges.empty:
            text = edges[['assay_id','raw_name','assay_label','gene_symbols','candidate_gene_symbols','modification','residue_site']].fillna('').astype(str).agg(' '.join,axis=1)
            matching = edges.loc[text.str.contains(query,case=False,regex=False)]
        fig, drawn = regimen_connectivity_figure(matching,feature_labels=feature_labels,edge_limit=connectivity_limit.value)
        fig.update_layout(meta=dict(fig.layout.meta,search=query,population_uids=df_cppa_wide.index.tolist(),
            condition_uid=uid,set_id=selected_dataset_id.removeprefix('CPPA-')))
        context = condition_display(condition.to_frame().T).map(source_group_label)
        message = (f"{len(edges):,} finite assay responses for {uid}; {len(matching):,} match search. Drawing {len(drawn):,} assays by absolute Δp, with assay-ID tie breaks."
                   if not matching.empty else
                   'No finite published responses for this condition. Choose another response condition or change the shared condition filters.' if edges.empty else
                   'No assays match this search. Clear or change the connectivity search.')
        return mo.vstack([
            mo.hstack([response_condition,connectivity_search,connectivity_limit],justify='start'),
            mo.md("One node represents the complete regimen and exposure condition. Each edge carries that condition's published signed Δp for an antibody assay; a combination response belongs to the full regimen."),
            mo.ui.table(context,selection=None,label='Complete connectivity condition',pagination=True,max_height=200),
            mo.md(message),fig if not matching.empty else mo.md('No connectivity graph is drawn for this selection.'),
            mo.md("Red is positive Δp, blue negative, gray zero. Width follows |Δp| relative to the largest drawn response. Search and the draw limit affect display only; all matching source responses remain in the table. Assay targets, candidate targets and phosphosites retain their annotation status."),
            mo.ui.table(matching,selection=None,pagination=True,max_height=350,label='Regimen-to-assay responses'),
        ])
    return (connectivity_view,)


@app.cell(hide_code=True)
def _(get_choices, set_choices, label_options, feature_labels, df_cppa_wide, mo):
    # === Descriptive Response Correlations — controls ===
    target_protein = mo.ui.dropdown(
        options=label_options(df_cppa_wide.columns.tolist(), feature_labels, sort=True),
        value=feature_labels[get_choices().get("target") if get_choices().get("target") in df_cppa_wide else sorted(df_cppa_wide.columns.tolist())[0]],
        searchable=True, allow_select_none=False,
        on_change=lambda value: set_choices(lambda previous: {**previous,"target":value}),
        label="Reference protein",
    )
    predictive_top_n = mo.ui.slider(start=10, stop=50, step=5, value=20, label="Proteins shown", debounce=True)
    return predictive_top_n, target_protein


@app.cell
def _(mo, df_cppa_wide, target_protein, correlate_one_vs_all, feature_labels):
    cppa_correlation_family = correlate_one_vs_all(df_cppa_wide,target_protein.value,inference=False)
    _results = cppa_correlation_family.reset_index().rename(columns={'feature':'Assay ID'})
    _results.insert(1,'Protein / assay',_results['Assay ID'].map(feature_labels))
    cppa_correlation_results = mo.ui.table(_results,selection='single',pagination=True,
        label='One-versus-all response results',max_height=350)
    return cppa_correlation_family, cppa_correlation_results


@app.cell
def _(assay_table, display_labels, feature_labels,
    df_cppa_wide,
    df_treatment,
    go,
    mo,
    correlate_one_vs_all,
    predictive_top_n,
    target_protein, cppa_correlation_family, cppa_correlation_results, correlate, correlation_figure, condition_labels,
):
    def _view():
        target = target_protein.value
        top_n = predictive_top_n.value

        markers = cppa_correlation_family
        markers = markers.loc[markers['status'] == 'ok']
        markers = markers.reindex(markers['r'].abs().sort_values(
            ascending=False, kind='stable'
        ).index).head(top_n)
        if markers.empty:
            return mo.md("**No correlations can be calculated from the available paired measurements.** Select at least three conditions with varying finite paired Δp values, or choose another reference assay.")

        colors = ['red' if x > 0 else 'blue' for x in markers['r']]
        fig = go.Figure(go.Bar(
            x=display_labels(markers.index, feature_labels), y=markers['r'], marker_color=colors,
            text=markers['r'].round(3), textposition='outside'
        ))
        fig.update_layout(
            title=f'Proteins Most Correlated with {feature_labels[target]}',
            xaxis_title='Protein', yaxis_title='Pearson Correlation',
            xaxis_tickangle=-45, height=500, yaxis_range=[-1, 1]
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray')

        return mo.vstack([
            fig,
            mo.ui.table(assay_table(markers[['r', 'n']], feature_labels,
                        {'r': 'Pearson r', 'n': 'Paired finite conditions'}, rank=True),
                        selection=None, pagination=True, label='Top correlated assays',
                        format_mapping={'Pearson r': '{:.3f}'.format}),
            mo.md("Correlations describe patterns, not treatment sensitivity. Shared controls mean conditions are not independent replicates."),
        ])

    def _discovery():
        selected = cppa_correlation_results.value
        pair = mo.md("Select a result row to plot the paired condition responses.")
        if len(selected):
            a, b = target_protein.value, selected.iloc[0]['Assay ID']
            result = correlate(df_cppa_wide[a],df_cppa_wide[b],inference=False)
            pair = correlation_figure(result,labels=condition_labels,x_label=feature_labels[a],y_label=feature_labels[b])
        return mo.vstack([mo.md("This is one reference assay versus all other assays in the shared selected condition population. All statuses and finite counts remain searchable. No p/q values are estimated for dependent response conditions. Row selection drives the pair plot below."),cppa_correlation_results,pair])

    def vary_together_tab():
        return mo.vstack([
            mo.md('Compare each protein with the reference across the shared selected conditions. Positive correlations move together; negative correlations move in opposite directions. Type an assay name in the menu to search it.'),
            mo.hstack([target_protein, predictive_top_n], justify="start"),
            _view(),
            mo.accordion({'Discover related response assays':_discovery()}),
        ])
    return (vary_together_tab,)


@app.cell(hide_code=True)
def _(get_choices, set_choices, mo, df_treatment, available_groupings):
    # === Δp Response Heatmap — controls ===
    heatmap_n_proteins = mo.ui.slider(start=10, stop=50, step=5, value=30, label="Proteins shown", debounce=True)
    heatmap_colorscale = mo.ui.dropdown(
        options=['RdBu_r', 'RdYlBu_r', 'Picnic', 'Portland', 'Spectral', 'Viridis'], value='RdBu_r', label="Colors",
    )
    heatmap_group_by = mo.ui.dropdown(
        options={label: key for label, key in {'Primary compound':'compound_name_1', 'Cell line':'cell_line_name', 'Exposure time':'time', 'Primary dose':'dosage_1', 'None':'None'}.items() if key in available_groupings(df_treatment)}, value=next(label for label,key in {"Primary compound":"compound_name_1", "Cell line":"cell_line_name", "Exposure time":"time", "Primary dose":"dosage_1", "None":"None"}.items() if key == (get_choices().get("context") if get_choices().get("context") in available_groupings(df_treatment) else "compound_name_1" if "compound_name_1" in available_groupings(df_treatment) else "None")), label="Row label context",
        on_change=lambda value: set_choices(lambda previous: {**previous,"context":value}),
    )
    return heatmap_colorscale, heatmap_group_by, heatmap_n_proteins


@app.cell
def _(assay_table, source_group_label, condition_labels, display_labels, feature_labels, df_cppa_wide, df_treatment, feature_summary, heatmap_figure, heatmap_colorscale, heatmap_group_by, heatmap_n_proteins, mo, hm_order, hm_cut):
    def _view():
        n_proteins = heatmap_n_proteins.value
        colorscale = heatmap_colorscale.value
        group_by = heatmap_group_by.value

        variances = (feature_summary(df_cppa_wide)['variance'] if len(df_cppa_wide)>1 else df_cppa_wide.iloc[0].abs()).dropna().sort_values(ascending=False, kind='stable')
        top_proteins = variances.head(n_proteins).index.tolist()

        heatmap_data = df_cppa_wide[top_proteins].copy()

        if group_by != 'None':
            groups = df_treatment.set_index('UID')[group_by].reindex(heatmap_data.index)
            row_labels = [f"{source_group_label(groups.loc[key])} | {condition_labels[key]}" for key in heatmap_data.index]
        else:
            row_labels = display_labels(heatmap_data.index, condition_labels)

        fig = heatmap_figure(
            heatmap_data, cluster_mode=hm_order.value, cluster_cut=hm_cut.value, feature_labels=feature_labels,
            sample_labels=dict(zip(heatmap_data.index, row_labels)),
            colorscale=colorscale, colorbar_title='Δp', row_id_label='Condition ID',
            title=f'Response map · {len(top_proteins)} proteins, {len(heatmap_data)} conditions',
        )
        fig.update_layout(xaxis_title='Protein', yaxis_title='Condition',
                          height=min(900, max(500, len(heatmap_data) * 20)), width=900)

        metric_label = 'Response variance' if len(df_cppa_wide) > 1 else 'Absolute Δp'
        return mo.vstack([
            fig,
            mo.ui.table([{'Conditions': len(heatmap_data), 'Proteins': len(top_proteins),
                          'Ranking metric': metric_label}], selection=None, pagination=False, label='Response map summary'),
            mo.ui.table(assay_table(variances.head(5).rename(metric_label).to_frame(), feature_labels, {metric_label: metric_label}, rank=True),
                        selection=None, pagination=False, label='Top response map assays'),
        ])

    def response_map_tab():
        return mo.vstack([
            mo.md('For one selected condition, columns use absolute Δp; otherwise variance. Rows include all selected conditions. Current order preserves source order; optional local clustering reorders the displayed matrix. Columns use the declared response magnitude or variance rule. Color shows raw published Δp. Row label context prefixes labels without grouping, averaging or clustering. Large maps retain all rows in a bounded chart; filter conditions for closer inspection.'),
            mo.hstack([heatmap_n_proteins, heatmap_colorscale, heatmap_group_by], justify="start"),
            mo.accordion({'Optional local clustering': mo.vstack([mo.hstack([hm_order,hm_cut],justify='start'),mo.md("Clusters use the displayed matrix with Euclidean distance and average linkage. Only complete entities are clustered; incomplete entities remain at the end with no cluster. No missing values are filled. The cut requests at most this many clusters; ties can give fewer. This local method is separate from the portal's unspecified clustering.")])}),
            _view(),
        ])
    return (response_map_tab,)


@app.cell(hide_code=True)
def _(get_choices, set_choices, label_options, display_labels, condition_labels, df_cppa_wide, mo):
    # === Condition Comparison — controls ===
    # Descriptive difference of two source-linked Δp responses.
    _uids = df_cppa_wide.index.tolist()
    compare_cond_a = mo.ui.dropdown(options=label_options(_uids, condition_labels, sort=True), value=condition_labels[get_choices().get("a") if get_choices().get("a") in _uids else _uids[0]], searchable=True, allow_select_none=False, label="Condition A", on_change=lambda value: set_choices(lambda previous: {**previous,"a":value}))
    compare_cond_b = mo.ui.dropdown(
        options=label_options(_uids, condition_labels, sort=True), value=condition_labels[get_choices().get("b") if get_choices().get("b") in _uids else _uids[min(1,len(_uids)-1)]], searchable=True, allow_select_none=False, label="Condition B",
        on_change=lambda value: set_choices(lambda previous: {**previous,"b":value}),
    )
    compare_top_n = mo.ui.slider(start=5, stop=40, step=5, value=20, label="Proteins shown", debounce=True)
    return compare_cond_a, compare_cond_b, compare_top_n


@app.cell
def _(display_labels, feature_labels, condition_labels, compare_cond_a, compare_cond_b, compare_top_n, df_cppa_wide, go, mo, pd, px):
    def _view():
        cond_a = compare_cond_a.value
        cond_b = compare_cond_b.value
        top_n = compare_top_n.value

        if cond_a not in df_cppa_wide.index or cond_b not in df_cppa_wide.index:
            return mo.md("⚠️ **Select two available conditions.**")
        if cond_a == cond_b:
            return mo.md("Select at least two conditions using the shared filters, then choose different conditions A and B.")

        vals_a = df_cppa_wide.loc[cond_a]
        vals_b = df_cppa_wide.loc[cond_b]
        diff = (vals_a - vals_b).dropna()
        if diff.empty:
            return mo.md("⚠️ **No overlapping proteins between the two conditions**")

        ranked = diff.reindex(diff.abs().sort_values(ascending=False).index).head(top_n)
        colors = ['red' if x > 0 else 'blue' for x in ranked.values]
        fig = go.Figure(go.Bar(
            x=display_labels(ranked.index, feature_labels), y=ranked.values, marker_color=colors,
            text=[f"{v:.3f}" for v in ranked.values], textposition='outside'
        ))
        fig.update_layout(
            title=f'Largest response differences · {len(ranked)} proteins',
            xaxis_title='Protein', yaxis_title='Response difference (A − B)',
            xaxis_tickangle=-45, height=450, showlegend=False
        )
        fig.add_hline(y=0, line_dash='dash', line_color='gray')

        common = vals_a.dropna().index.intersection(vals_b.dropna().index)
        scatter_df = pd.DataFrame({
            'Protein': display_labels(common, feature_labels),
            'A': vals_a.loc[common].values,
            'B': vals_b.loc[common].values,
        })
        fig2 = px.scatter(
            scatter_df, x='A', y='B', hover_name='Protein',
            labels={'A': condition_labels[cond_a], 'B': condition_labels[cond_b]},
            title='Response in A vs B'
        )
        if len(scatter_df):
            _lo = float(min(scatter_df['A'].min(), scatter_df['B'].min()))
            _hi = float(max(scatter_df['A'].max(), scatter_df['B'].max()))
            fig2.add_shape(type='line', x0=_lo, y0=_lo, x1=_hi, y1=_hi,
                           line=dict(dash='dash', color='gray'))
        fig2.update_layout(height=500)

        up = int((ranked > 0).sum())
        down = int((ranked < 0).sum())
        summary = [{
            'Condition A': condition_labels[cond_a], 'Condition A ID': cond_a,
            'Condition B': condition_labels[cond_b], 'Condition B ID': cond_b,
            'Proteins compared': len(diff), 'Top assays shown': len(ranked),
            'Higher in A (shown)': up, 'Higher in B (shown)': down,
        }]
        return mo.vstack([fig, fig2, mo.ui.table(summary, selection=None,
                          pagination=False, label='Condition comparison summary')])

    def compare_conditions_tab():
        return mo.vstack([
            mo.md(
                "Compare published responses for A and B. Positive bars mean a larger response in A; "
                "negative bars mean a larger response in B. Each scatter point is one protein; "
                "the dashed line marks equal responses. No significance test is applied. Type a condition name in a menu to search it."
            ),
            mo.hstack([compare_cond_a, compare_cond_b, compare_top_n], justify="start"),
            _view(),
        ])
    return (compare_conditions_tab,)


@app.cell(hide_code=True)
def _(mo):
    # === Published portal drug analyses — controls ===
    from epifolio.tcpa.loaders import load_cppa_portal_analyses
    from epifolio.tcpa.stats import published_family
    from epifolio.tcpa.figures import published_volcano_figure, published_box_figure
    portal_tables = load_cppa_portal_analyses(anchor=globals().get("__file__"))
    portal_analysis = mo.ui.dropdown(options={'Protein response to a drug': 'protein_response',
        'Predictive markers of sensitivity': 'predictive_markers', 'Sensitive vs resistant cell lines': 'sensitive_resistant'},
        value='Protein response to a drug', label="Published analysis")
    portal_sensitivity = mo.ui.dropdown(options=sorted(portal_tables['predictive_markers'].drug_dataset.unique()), label="Sensitivity dataset",
        value=sorted(portal_tables['predictive_markers'].drug_dataset.unique())[0])
    return portal_tables, portal_analysis, portal_sensitivity, published_family, published_volcano_figure, published_box_figure


@app.cell(hide_code=True)
def _(mo, portal_tables, portal_analysis, portal_sensitivity, df_treatment):
    portal_scope = portal_tables[portal_analysis.value]
    if portal_analysis.value == 'predictive_markers':
        portal_scope = portal_scope.loc[portal_scope.drug_dataset.eq(portal_sensitivity.value)]
    _drugs = sorted(portal_scope.drug_name.unique())
    _selected = set(df_treatment.compound_name_1.astype(str))
    # Prefer a drug used in the selected experiment; the tables themselves are portal-wide.
    portal_drug = mo.ui.dropdown(options=_drugs, value=next((name for name in _drugs if name in _selected), _drugs[0]),
        searchable=True, allow_select_none=False, label="Drug")
    _levels = {'p0': 'Baseline (P0)', 'p1': 'After treatment (P1)', 'deltaP': 'Response (Δp)', 'deltap': 'Response (Δp)'}
    _present = sorted(portal_scope.protein_level.unique()) if 'protein_level' in portal_scope else []
    portal_level = mo.ui.dropdown(options={_levels[key]: key for key in _present} or {'Response (Δp)': 'deltap'},
        value=_levels[_present[0]] if _present else 'Response (Δp)', label="Protein level")
    return portal_scope, portal_drug, portal_level


@app.cell
def _(portal_analysis, portal_scope, portal_drug, portal_level, published_family):
    _rows = portal_scope.loc[portal_scope.drug_name.eq(portal_drug.value)]
    if 'protein_level' in _rows:
        _rows = _rows.loc[_rows.protein_level.eq(portal_level.value)]
    _rows = _rows.set_index('protein_name')
    # Published adjusted values are used where the portal provides them; predictive markers have none.
    _effect, _adjusted, portal_alpha = {'protein_response': ('diff', 'fdr', 0.05), 'predictive_markers': ('pearson_r', None, 0.05),
                                        'sensitive_resistant': ('median_diff', 'p_adjust', 0.05)}[portal_analysis.value]
    portal_family = published_family(_rows[_effect], _rows.p_value, alpha=portal_alpha,
        adjusted=None if _adjusted is None else _rows[_adjusted]).join(_rows.drop(columns=[_effect, 'p_value']))
    portal_family['result'] = portal_family.classification.map({'positive': 'Positive', 'negative': 'Negative', 'not_selected': 'Below cutoff'})
    _columns = {'effect': {'protein_response': 'Published difference', 'predictive_markers': "Pearson's r", 'sensitive_resistant': 'Published median difference'}[portal_analysis.value],
                'p_value': 'p-value', 'q_value': 'Published adjusted p-value', 'result': 'Result'}
    if _adjusted is None:
        del _columns['q_value']
    if portal_analysis.value == 'sensitive_resistant':
        _columns.update({f'{group}_{stat}': f'{group.capitalize()} {stat}' for group in ['sensitive', 'resistant'] for stat in ['min', 'q1', 'median', 'q3', 'max']})
    portal_columns = _columns
    portal_effect_label = _columns['effect']
    return portal_family, portal_columns, portal_alpha, portal_effect_label


@app.cell
def _(mo, portal_family, portal_analysis, portal_drug, portal_effect_label, published_volcano_figure):
    titles = {'protein_response': 'Protein response', 'predictive_markers': 'Predictive markers', 'sensitive_resistant': 'Sensitive vs resistant'}
    fig = published_volcano_figure(portal_family, effect_label=portal_effect_label, title=f"{titles[portal_analysis.value]}: {portal_drug.value}")
    fig.update_layout(dragmode='lasso')
    portal_point_ids = {(curve, point): row[0] for curve, trace in enumerate(fig.data)
                        for point, row in enumerate(trace.customdata)}
    portal_plot = mo.ui.plotly(fig)
    return portal_plot, portal_point_ids


@app.cell
def _(mo, portal_family, portal_columns, portal_analysis, portal_plot, portal_point_ids):
    selected_ids = {portal_point_ids[(point['curveNumber'], point.get('pointIndex', point.get('pointNumber')))]
                    for point in portal_plot.value}
    family = portal_family.loc[portal_family.index.isin(selected_ids)] if selected_ids else portal_family
    _display = family.sort_values('p_value', kind='stable').loc[:, list(portal_columns)].rename(columns=portal_columns).rename_axis('Protein / assay').reset_index()
    portal_results = mo.ui.table(_display, selection='single' if portal_analysis.value == 'sensitive_resistant' else None,
        initial_selection=[0] if portal_analysis.value == 'sensitive_resistant' and len(_display) else None,
        pagination=True, page_size=10, label='Published portal results',
        format_mapping={name: '{:.3g}'.format for name in _display.columns[1:] if name != 'Result'})
    portal_shown_count = len(_display)
    return portal_results, portal_shown_count


@app.cell
def _(mo, portal_tables, portal_analysis, portal_sensitivity, portal_drug, portal_level, portal_family, portal_results, portal_alpha, portal_effect_label, portal_plot, portal_shown_count, published_box_figure, df_treatment):
    levels = {'p0': 'Baseline protein level (P0)', 'p1': 'Protein level after treatment (P1)', 'deltaP': 'Protein response (Δp)', 'deltap': 'Protein response (Δp)'}

    def _view():
        kind = portal_analysis.value
        if portal_family.empty:
            return mo.md("No published results for this drug and protein level.")
        basis = 'unadjusted p' if kind == 'predictive_markers' else 'published adjusted p'
        parts = [portal_plot, mo.md(f"{len(portal_family)} published assays for **{portal_drug.value}**; {int(portal_family.classification.eq('positive').sum())} positive and {int(portal_family.classification.eq('negative').sum())} negative at {basis} ≤ {portal_alpha}."),
                 mo.md(f"Lasso or box-select points to filter the table. Use the plot's Reset state control to show all rows. Showing {portal_shown_count} of {len(portal_family)} published assays; selection does not change the published statistics."),
                 portal_results]
        if kind == 'sensitive_resistant' and len(portal_results.value):
            name = portal_results.value.iloc[0]['Protein / assay']
            parts.append(published_box_figure(portal_family.loc[name], {'sensitive': 'Sensitive', 'resistant': 'Resistant'},
                title=f"{name}: published summaries for {portal_drug.value}", y_label=levels[portal_level.value]))
        return mo.vstack(parts)

    def drug_analyses_tab():
        kind = portal_analysis.value
        controls = [portal_analysis] + ([portal_sensitivity] if kind == 'predictive_markers' else []) + [portal_drug] + ([] if kind == 'protein_response' else [portal_level])
        in_selection = portal_drug.value in set(df_treatment.compound_name_1.astype(str))
        notes = {
            'protein_response': "The portal's protein response for each drug: a published difference with its p-value and FDR. Positive values mean the protein increased after treatment. Points are colored at published FDR ≤ 0.05, a display cutoff chosen here; the portal lists the table without one.",
            'predictive_markers': "Pearson correlation between protein level and drug sensitivity in the chosen sensitivity dataset, for baseline (P0), post-treatment (P1) or response (Δp) levels. The portal publishes no adjusted p-value and colors at unadjusted p ≤ 0.05; the same rule is used here, so expect false positives across this many assays. The source does not state which direction of the sensitivity measure means more sensitive.",
            'sensitive_resistant': "The portal's comparison of protein levels between sensitive and resistant cell lines. Points are colored at the published adjusted p ≤ 0.05, a display cutoff chosen here. Select a table row to draw both groups from the published five-number summaries. The published median difference is not the difference of the two box medians, and the source does not define it or how cell lines were classified.",
        }
        return mo.vstack([
            mo.md("Browse the CPPA portal's published drug analyses. These tables are pooled across CPPA experiments by the provider; the experiment and condition filters above do not change them."),
            mo.hstack(controls, justify="start"),
            mo.md(notes[kind] + (" " if in_selection else f" {portal_drug.value} is not a primary compound in the selected conditions.")),
            _view(),
            mo.accordion({'Published tables and sources': mo.vstack([
                mo.md("Values are the portal documents unchanged; nothing is recalculated from local measurements, and the underlying sensitivity outcomes are not published."),
                mo.ui.table([{'Table': name.replace('_', ' ').capitalize(), 'Rows': len(portal_tables[name]), 'Drugs': portal_tables[name].drug_name.nunique(),
                              'Assays': portal_tables[name].protein_name.nunique(), 'Portal document type': portal_tables['contract']['tables'][name]['portal_type']}
                             for name in ['protein_response', 'predictive_markers', 'sensitive_resistant']], selection=None, pagination=False, label='Published portal tables'),
            ])}),
        ])
    return (drug_analyses_tab,)


@app.cell(hide_code=True)
def _(mo, dataset_description, assay_display, df_cppa_wide, df_treatment, population_summary, response_checks_panel):
    from epifolio.tcpa.text import phenotype_input_gates, phenotype_gate_description

    def about_tab():
        _groups = df_treatment.set_index('UID')['cell_line_name'].reindex(df_cppa_wide.index)
        _counts,_finite = population_summary(df_cppa_wide,_groups)
        _counts = _counts.rename(columns={'Observations':'Response conditions'})
        return mo.vstack([
            response_checks_panel(),
            mo.accordion({'Selected cell-line summary':mo.vstack([mo.md("Counts are selected response conditions, not independent individual samples or sensitivity outcomes."),mo.ui.table(_counts),mo.ui.table(_finite)])}),
            mo.md(dataset_description(df_cppa_wide)),
            mo.accordion({'Connectivity scope':mo.md("Treatment connectivity uses unchanged published Δp for one selected full regimen, joined by condition UID and assay ID. The graph ranks absolute responses only for drawing; its table retains every matching finite assay. Missing design values and unresolved reconstruction remain explicit. No sample baseline is recomputed. Cross-condition pooling needs an aggregation and threshold contract; pathway colors need validated membership data. This view is descriptive and supplies no sensitivity outcome or predictive-marker test.")}),
            mo.accordion({'Analyses requiring additional inputs': mo.vstack([mo.md(phenotype_gate_description('cppa')), mo.ui.table(phenotype_input_gates('cppa'), selection=None, pagination=False, label='Phenotype input gates')])}),
            mo.accordion({"Assay guide": mo.ui.table(assay_display(df_cppa_wide.attrs['panel_metadata']))}),
        ])
    return (about_tab,)


@app.cell(hide_code=True)
def _(mo):
    hm_order = mo.ui.dropdown(options={'Current order':'source','Local row clustering':'rows','Local assay clustering':'columns','Local rows and assays':'both'}, value='Current order', label='Heatmap order')
    hm_cut = mo.ui.slider(start=1,stop=10,value=3,label='Maximum local clusters',debounce=True)
    return hm_order, hm_cut


if __name__ == "__main__":
    app.run()
