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
    from epifolio.tcpa.text import label_options, assay_table, display_labels, dataset_description, available_groupings, source_group_label, cppa_reconstruction_description, condition_display, assay_display, cppa_atlas_display

    return mo, label_options, assay_table, display_labels, dataset_description, available_groupings, source_group_label, cppa_reconstruction_description, condition_display, assay_display, cppa_atlas_display


@app.cell(hide_code=True)
def _(atlas, atlas_entries, cppa_atlas_values, mo, portal_tables):
    _published = set().union(*(portal_tables[name].drug_name for name in ['protein_response', 'predictive_markers', 'sensitive_resistant']))
    _sources = atlas_entries[['publication', 'release', 'units', 'source_url']].drop_duplicates()
    mo.vstack([mo.md(f"""
    # Perturbed Proteomics

    How do protein levels change when cancer cell lines are treated with a drug? This page explores the Cancer Perturbed Proteomics Atlas (CPPA):
    **{len(atlas):,} treatment conditions · {len(cppa_atlas_values(atlas, 'compound'))} compounds · {atlas.cell_line.nunique()} cell lines · {atlas.experiment.nunique()} studies from {atlas.lab.nunique()} labs.**
    """),
    mo.accordion({
        'How to use this page': mo.md(f"""
        1. **Choose a compound**, or start from a cell line. Each choice lists how many conditions, cell lines and studies the atlas holds for it.
        2. **Use the tabs.** Conditions shows where it was tested; narrow or tick rows there and every other tab follows. Published Analysis shows the portal's pooled analysis (available for {len(_published)} compounds).
        3. **Explore the measured responses** in the remaining tabs, one study at a time.

        Δp is the published, replicate-averaged change in a protein from control to treatment.
        Positive values mean the protein went up; negative values mean it went down.
        These are protein responses, not measures of drug sensitivity.
        """),
        'About the data': mo.vstack([
            mo.md("Every value on this page is shown as published: names, doses and exposure times keep the studies' own wording, and nothing is filled in or recalculated. "
                  "Compounds are listed under the names the studies used; the atlas does not record drug targets or mechanisms, so none are shown. "
                  "Tissue is shown only where a cell line's name exactly matches a CCLE cell line; other lines are marked \"Not yet mapped\"."),
            mo.ui.table(_sources.rename(columns={'publication': 'Publication', 'release': 'Portal release', 'units': 'Units', 'source_url': 'Source'}),
                        selection=None, pagination=False, label='Source'),
        ]),
        'How to read these responses': mo.md("Each Δp is measured against controls from its own study, batch and lab, so responses are analysed within one study at a time. "
                                             "When your selection spans several studies, the largest is shown first and you can switch between them. "
                                             "Conditions can share controls, so these plots describe patterns without significance tests. "
                                             "Treatment and control checks are in the About the Data tab; an unresolved check does not remove the published response."),
    })])
    return


@app.cell
def _():
    from epifolio.tcpa.loaders import load_cppa_atlas, load_ccle_tissues, load_cppa_portal_analyses
    from epifolio.tcpa.observations import (cppa_tissue_labels, cppa_atlas_conditions, cppa_atlas_values, cppa_atlas_mask,
        cppa_study_labels, cppa_entry_options, cppa_profile, cppa_study_choices, CPPA_ATLAS_FACETS)
    atlas_summary, atlas_entries = load_cppa_atlas(anchor=globals().get("__file__"))
    # Tissue comes from exact CCLE cell-line name matches; other lines stay "Not yet mapped".
    atlas = cppa_atlas_conditions(atlas_summary, cppa_tissue_labels(atlas_summary.cell_line_name.unique(), load_ccle_tissues(anchor=globals().get("__file__"))))
    study_labels = cppa_study_labels(atlas)
    portal_tables = load_cppa_portal_analyses(anchor=globals().get("__file__"))
    return (atlas, atlas_entries, cppa_atlas_values, cppa_atlas_mask, cppa_entry_options, cppa_profile, cppa_study_choices,
            CPPA_ATLAS_FACETS, study_labels, portal_tables)


@app.cell
def _(mo):
    # Selections flow downward only: a new starting point resets the narrowing and the study, never the reverse.
    # Self-loops let each menu's own cell re-run: it reads its choice back from state, not from the menu.
    get_entry, set_entry = mo.state({'mode': 'Compound', 'compound': None, 'compare': None, 'cell_line': None}, allow_self_loops=True)
    get_facets, set_facets = mo.state({}, allow_self_loops=True)
    get_study, set_study = mo.state(None, allow_self_loops=True)

    def set_anchor(**changes):
        set_entry(lambda entry: {**entry, **changes})
        set_facets({})
        set_study(None)
    return get_entry, get_facets, set_facets, get_study, set_study, set_anchor


@app.cell(hide_code=True)
def _(mo, atlas, portal_tables, get_entry, set_anchor, cppa_entry_options):
    entry = get_entry()
    _published = set().union(*(portal_tables[name].drug_name for name in ['protein_response', 'predictive_markers', 'sensitive_resistant']))
    _by_line = atlas.loc[atlas.cell_line.eq(entry['cell_line'])] if entry['mode'] == 'Cell line' and entry['cell_line'] else atlas

    def _dropdown(options, chosen, label, on_change):
        labels = {value: key for key, value in options.items()}
        return mo.ui.dropdown(options=options, value=labels.get(chosen), searchable=True, label=label, on_change=on_change)

    entry_mode = mo.ui.radio(options=['Compound', 'Cell line'], value=entry['mode'], inline=True, label='Browse by',
        on_change=lambda mode: set_anchor(mode=mode, compound=None, compare=None, cell_line=None))
    entry_cell_line = _dropdown(cppa_entry_options(atlas, 'cell_line'), entry['cell_line'], 'Cell line',
        lambda line: set_anchor(cell_line=line, compound=None, compare=None))
    _compounds = cppa_entry_options(_by_line, 'compound', _published)
    entry_compound = _dropdown(_compounds, entry['compound'], 'Compound' if entry['mode'] == 'Compound' else 'Compound (optional)',
        lambda name: set_anchor(compound=name, compare=None))
    entry_compare = _dropdown({key: value for key, value in _compounds.items() if value != entry['compound']}, entry['compare'],
        'Compare with (optional)', lambda name: set_anchor(compare=name))
    return entry, entry_mode, entry_cell_line, entry_compound, entry_compare


@app.cell(hide_code=True)
def _(mo, atlas, entry, entry_mode, entry_cell_line, entry_compound, entry_compare):
    _controls = ([entry_cell_line] if entry['mode'] == 'Cell line' else []) + [entry_compound] + ([entry_compare] if entry['compound'] else [])
    _header = mo.vstack([mo.md("### Select a compound or cell line"), entry_mode, mo.hstack(_controls, justify='start', wrap=True)])
    _missing = ('cell line' if entry['mode'] == 'Cell line' and not entry['cell_line']
                else 'compound' if entry['mode'] == 'Compound' and not entry['compound'] else None)
    mo.stop(_missing is not None, mo.vstack([_header, mo.md(f"Select a {_missing} to see its results. Type in the menu to search.")]))

    anchor_compounds = [name for name in (entry['compound'], entry['compare']) if name]
    _mask = atlas.cell_line.eq(entry['cell_line']) if entry['mode'] == 'Cell line' else atlas.cell_line.notna()
    if anchor_compounds:
        _mask &= atlas.compounds.map(lambda names: bool(set(anchor_compounds) & set(names)))
    anchored = atlas.loc[_mask]
    _header
    return anchor_compounds, anchored


@app.cell(hide_code=True)
def _(mo, entry, anchored, get_facets, set_facets, cppa_atlas_values, cppa_atlas_mask, CPPA_ATLAS_FACETS):
    _fields = (['dose', 'time', 'culture', 'regimen', 'stimulus'] if entry['mode'] == 'Cell line'
               else ['tissue', 'cell_line', 'dose', 'time', 'culture', 'regimen', 'stimulus'])
    _facets = {field: values for field, values in get_facets().items() if field in _fields}

    def _narrow(field):
        # Each menu lists what remains under the other menus, plus its own current choices.
        _remaining = anchored.loc[cppa_atlas_mask(anchored, _facets, exclude=field)]
        _options = sorted(set(cppa_atlas_values(_remaining, field)) | set(_facets.get(field, [])))
        return mo.ui.multiselect(options=_options, value=_facets.get(field, []), label=CPPA_ATLAS_FACETS[field],
            on_change=lambda values: set_facets(lambda facets: {**facets, field: list(values)}))
    narrow_filters = [_narrow(field) for field in _fields]
    narrowed = anchored.loc[cppa_atlas_mask(anchored, _facets)]
    return narrow_filters, narrowed


@app.cell(hide_code=True)
def _(mo, narrowed, study_labels, cppa_atlas_display):
    condition_table = mo.ui.table(cppa_atlas_display(narrowed, study_labels), selection='multi', page_size=10,
        label='Conditions · check rows to analyse only those')
    return (condition_table,)


@app.cell(hide_code=True)
def _(mo, entry, anchored, anchor_compounds, narrowed, narrow_filters, condition_table, set_facets, study_labels, cppa_profile):
    _clear = mo.ui.button(label='Clear narrowing', on_click=lambda _: set_facets({}))

    def conditions_tab():
        subject = ' and '.join(anchor_compounds) if anchor_compounds else entry['cell_line']
        if anchor_compounds and entry['mode'] == 'Cell line':
            subject += f" in {entry['cell_line']}"
        return mo.vstack([
            mo.md(f"**What the studies recorded for {subject}**"),
            mo.ui.table(cppa_profile(anchored, study_labels, anchor_compounds), selection=None, pagination=False, label='As reported by the studies'),
            mo.md(f"**Narrow** · {len(narrowed):,} of {len(anchored):,} conditions. Tick rows to analyse only those."),
            mo.hstack(narrow_filters + [_clear], justify='start', align='end', wrap=True),
            condition_table,
        ])
    return (conditions_tab,)


@app.cell(hide_code=True)
def _(mo, atlas, narrowed, condition_table, anchor_compounds, get_study, set_study, study_labels, cppa_study_choices):
    _checked = condition_table.value
    _selection = atlas.loc[_checked['Condition ID'].tolist()] if len(_checked) else narrowed
    mo.stop(_selection.empty, mo.md("No conditions match these choices. Clear the narrowing to continue."))
    _choices = cppa_study_choices(_selection, study_labels, anchor_compounds)
    _labels = {value: key for key, value in _choices.items()}
    selected_dataset_id = get_study() if get_study() in _labels else next(iter(_choices.values()))
    selected_atlas = _selection.loc[_selection.experiment.eq(selected_dataset_id)]
    # Responses are compared within one study, because each Δp is measured against that study's own controls.
    if len(_choices) > 1:
        _lead = ', shared studies first' if len(anchor_compounds) > 1 else ''
        mo.output.replace(mo.hstack([
            mo.ui.dropdown(options=_choices, value=_labels[selected_dataset_id], label='Study', searchable=True, on_change=set_study),
            mo.md(f"{len(_choices)} studies, largest first{_lead}.")], justify='start', align='center'))
    return selected_atlas, selected_dataset_id


@app.cell
def _():
    from functools import lru_cache
    from epifolio.tcpa.loaders import load_cppa
    # Slicing within one experiment re-selects conditions without re-reading its responses.
    load_cppa_cached = lru_cache(maxsize=4)(load_cppa)
    return (load_cppa_cached,)


@app.cell
def _(selected_dataset_id, load_cppa_cached, mo):
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    cell_lines, compounds, source_response, source_treatment, resources, times, source_long, source_proteins = load_cppa_cached(anchor=globals().get("__file__"), set_id=selected_dataset_id.removeprefix("CPPA-"))
    mo.stop(source_response.empty, mo.md("No measured responses are available for this experiment."))
    return source_response, source_treatment, source_long, pd, px, go


@app.cell(hide_code=True)
def _(source_response, source_treatment, selected_atlas, selected_dataset_id, study_labels, mo):
    from epifolio.tcpa.observations import select_cppa_conditions
    df_cppa_wide, df_treatment, selected_condition_metadata = select_cppa_conditions(
        source_response, source_treatment, {'condition_uid': selected_atlas.index.tolist()})
    mo.stop(df_cppa_wide.empty, mo.md(f"### {study_labels[selected_dataset_id]}\n\nNo finite response measurements are available in this selection. Change the choices above to continue."))
    feature_labels = df_cppa_wide.attrs['feature_labels']
    condition_labels = df_cppa_wide.attrs['condition_labels']
    protein_list = df_cppa_wide.columns.tolist()
    _coverage = df_cppa_wide.attrs['coverage']
    def _unreported(column):
        return int(df_treatment[column].fillna('').astype(str).str.strip().eq('').sum())
    _gaps = [f"{label} unreported for {count}" for label, count in [('time', _unreported('time')), ('dose', _unreported('dosage_1'))] if count]
    mo.md(' · '.join([f"**{study_labels[selected_dataset_id]}**", f"{len(df_treatment):,} of {len(source_response):,} conditions", f"{selected_atlas.cell_line.nunique()} cell lines",
                      f"{_coverage['measured_features']} of {_coverage['total_features']} proteins measured"] + _gaps))
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
    get_tab, set_tab = mo.state("Treatment Coverage")
    return get_tab, set_tab


@app.cell(hide_code=True)
def _(mo, get_tab, set_tab, conditions_tab, published_section, browse_tab, response_changes_tab, vary_together_tab, response_map_tab, compare_conditions_tab, about_tab):
    sections = mo.ui.tabs(
        {
            "Treatment Coverage": browse_tab,
            "Published Analysis": published_section,
            "Largest Response Changes": response_changes_tab,
            "Responses That Vary Together": vary_together_tab,
            "Response Map": response_map_tab,
            "Compare Two Conditions": compare_conditions_tab,
            "About the Data": about_tab,
            "Conditions": conditions_tab,
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
            mo.accordion({'Response checks': mo.vstack([mo.md('Unconfirmed controls are marked as candidates; unknown counts show as Unreported.'), mo.ui.table(reconstruction_table)])}),
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
            mo.md('Selected conditions by compound and cell line.'),
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
                return mo.vstack([response_condition,mo.md('No measured responses for this condition. Choose another.')])
            detail = mo.md("Select a row to see its individual samples.")
            if assay is not None:
                from epifolio.tcpa.loaders import load_cppa_sample_detail
                samples, diagnostic = load_cppa_sample_detail(uid,assay,anchor=globals().get('__file__'),set_id=selected_dataset_id.removeprefix('CPPA-'))
                detail = mo.vstack([mo.md(f"Condition {uid} · Assay {assay} · Published Δp: {df_cppa_wide.loc[uid,assay]:.9g}. Reconstruction: **{diagnostic.reconstruction_status}**. {diagnostic.reconstruction_reason}"),
                    mo.md("Control (P0) and treated (P1) samples. Unconfirmed controls are marked as candidates."),mo.ui.table(samples,label='Selected individual samples',selection=None,pagination=True,max_height=350)])
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
            return mo.md("Not enough data for this ranking. With one condition, use Absolute mean Δp.")

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
            mo.md("Proteins with the largest responses across the selected conditions. Red is up, blue is down. Change **Response view** to see one condition in full."),
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
        message = (f"{len(matching):,} of {len(edges):,} proteins match; the {len(drawn):,} largest are drawn."
                   if not matching.empty else
                   'No measured responses for this condition. Choose another.' if edges.empty else
                   'No proteins match this search.')
        return mo.vstack([
            mo.hstack([response_condition,connectivity_search,connectivity_limit],justify='start'),
            mo.md("The treatment links to each protein it changed. Red is up, blue is down; thicker lines are larger changes."),
            mo.ui.table(context,selection=None,label='Complete connectivity condition',pagination=True,max_height=200),
            mo.md(message),fig if not matching.empty else mo.md('No connectivity graph is drawn for this selection.'),
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
            return mo.md("**Not enough data to correlate.** Select at least three conditions, or choose another reference.")

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
            mo.md("Correlations describe patterns; they are not significance tests."),
        ])

    def _discovery():
        selected = cppa_correlation_results.value
        pair = mo.md("Select a row to plot the pair.")
        if len(selected):
            a, b = target_protein.value, selected.iloc[0]['Assay ID']
            result = correlate(df_cppa_wide[a],df_cppa_wide[b],inference=False)
            pair = correlation_figure(result,labels=condition_labels,x_label=feature_labels[a],y_label=feature_labels[b])
        return mo.vstack([mo.md("Every protein against the reference."),cppa_correlation_results,pair])

    def vary_together_tab():
        return mo.vstack([
            mo.md('Proteins that rise and fall with the reference (positive) or against it (negative).'),
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
            mo.md('Proteins that vary most across the selected conditions. Red is up, blue is down.'),
            mo.hstack([heatmap_n_proteins, heatmap_colorscale, heatmap_group_by], justify="start"),
            mo.accordion({'Optional local clustering': mo.vstack([mo.hstack([hm_order,hm_cut],justify='start'),mo.md("Clusters rows and proteins with no gaps; the rest stay at the end.")])}),
            _view(),
        ])
    return (response_map_tab,)


@app.cell(hide_code=True)
def _(get_choices, set_choices, label_options, display_labels, condition_labels, df_cppa_wide, atlas, anchor_compounds, mo):
    # === Condition Comparison — controls ===
    # Descriptive difference of two source-linked Δp responses.
    _uids = df_cppa_wide.index.tolist()
    # With a compared compound, A and B default to one condition of each.
    _first = lambda name, fallback: next((uid for uid in _uids if name in atlas.compounds.get(uid, ())), fallback)
    _default_a, _default_b = ((_first(anchor_compounds[0], _uids[0]), _first(anchor_compounds[1], _uids[min(1, len(_uids)-1)]))
                              if len(anchor_compounds) > 1 else (_uids[0], _uids[min(1, len(_uids)-1)]))
    compare_cond_a = mo.ui.dropdown(options=label_options(_uids, condition_labels, sort=True), value=condition_labels[get_choices().get("a") if get_choices().get("a") in _uids else _default_a], searchable=True, allow_select_none=False, label="Condition A", on_change=lambda value: set_choices(lambda previous: {**previous,"a":value}))
    compare_cond_b = mo.ui.dropdown(
        options=label_options(_uids, condition_labels, sort=True), value=condition_labels[get_choices().get("b") if get_choices().get("b") in _uids else _default_b], searchable=True, allow_select_none=False, label="Condition B",
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
            return mo.md("Choose two different conditions.")

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
                "Bars above zero are higher in A; below zero, higher in B. Points on the dashed line respond equally."
            ),
            mo.hstack([compare_cond_a, compare_cond_b, compare_top_n], justify="start"),
            _view(),
        ])
    return (compare_conditions_tab,)


@app.cell
def _(mo):
    # The chosen analysis is kept while the compound changes, whenever the new compound has it.
    get_portal_kind, set_portal_kind = mo.state('protein_response', allow_self_loops=True)
    return get_portal_kind, set_portal_kind


@app.cell(hide_code=True)
def _(mo, portal_tables, anchor_compounds, get_portal_kind, set_portal_kind):
    # === Published portal drug analyses — controls ===
    from epifolio.tcpa.stats import published_family
    from epifolio.tcpa.figures import published_volcano_figure, published_box_figure
    portal_drug = anchor_compounds[0] if anchor_compounds else None
    _names = {'Protein response to the drug': 'protein_response', 'Predictive markers of sensitivity': 'predictive_markers',
              'Sensitive vs resistant cell lines': 'sensitive_resistant'}
    _available = {label: key for label, key in _names.items() if portal_tables[key].drug_name.eq(portal_drug).any()}
    portal_kind = get_portal_kind() if get_portal_kind() in _available.values() else next(iter(_available.values()), None)
    portal_analysis = mo.ui.dropdown(options=_available or _names, value=next((label for label, key in _available.items() if key == portal_kind), None),
        label="Published analysis", on_change=set_portal_kind)
    _datasets = sorted(portal_tables['predictive_markers'].loc[lambda rows: rows.drug_name.eq(portal_drug)].drug_dataset.unique())
    portal_sensitivity = mo.ui.dropdown(options=_datasets or ['—'], value=(_datasets or ['—'])[0], label="Sensitivity dataset")
    return portal_drug, portal_kind, portal_analysis, portal_sensitivity, published_family, published_volcano_figure, published_box_figure


@app.cell(hide_code=True)
def _(mo, portal_tables, portal_drug, portal_kind, portal_sensitivity):
    portal_scope = portal_tables[portal_kind or 'protein_response'].loc[lambda rows: rows.drug_name.eq(portal_drug)]
    if portal_kind == 'predictive_markers':
        portal_scope = portal_scope.loc[portal_scope.drug_dataset.eq(portal_sensitivity.value)]
    _levels = {'p0': 'Baseline (P0)', 'p1': 'After treatment (P1)', 'deltaP': 'Response (Δp)', 'deltap': 'Response (Δp)'}
    _present = sorted(portal_scope.protein_level.unique()) if 'protein_level' in portal_scope else []
    portal_level = mo.ui.dropdown(options={_levels[key]: key for key in _present} or {'Response (Δp)': 'deltap'},
        value=_levels[_present[0]] if _present else 'Response (Δp)', label="Protein level")
    return portal_scope, portal_level


@app.cell
def _(portal_kind, portal_scope, portal_level, published_family):
    _rows = portal_scope
    if 'protein_level' in _rows:
        _rows = _rows.loc[_rows.protein_level.eq(portal_level.value)]
    _rows = _rows.set_index('protein_name')
    # Published adjusted values are used where the portal provides them; predictive markers have none.
    _effect, _adjusted, portal_alpha = {'protein_response': ('diff', 'fdr', 0.05), 'predictive_markers': ('pearson_r', None, 0.05),
                                        'sensitive_resistant': ('median_diff', 'p_adjust', 0.05)}[portal_kind or 'protein_response']
    if portal_kind is None or _rows.empty:
        portal_family = _rows.iloc[:0]
    else:
        portal_family = published_family(_rows[_effect], _rows.p_value, alpha=portal_alpha,
            adjusted=None if _adjusted is None else _rows[_adjusted]).join(_rows.drop(columns=[_effect, 'p_value']))
        portal_family['result'] = portal_family.classification.map({'positive': 'Positive', 'negative': 'Negative', 'not_selected': 'Below cutoff'})
    _columns = {'effect': {'protein_response': 'Published difference', 'predictive_markers': "Pearson's r", 'sensitive_resistant': 'Published median difference'}[portal_kind or 'protein_response'],
                'p_value': 'p-value', 'q_value': 'Published adjusted p-value', 'result': 'Result'}
    if _adjusted is None:
        del _columns['q_value']
    if portal_kind == 'sensitive_resistant':
        _columns.update({f'{group}_{stat}': f'{group.capitalize()} {stat}' for group in ['sensitive', 'resistant'] for stat in ['min', 'q1', 'median', 'q3', 'max']})
    portal_columns = _columns
    portal_effect_label = _columns['effect']
    return portal_family, portal_columns, portal_alpha, portal_effect_label


@app.cell
def _(mo, portal_family, portal_kind, portal_drug, portal_effect_label, published_volcano_figure):
    portal_plot, portal_point_ids = None, {}
    if not portal_family.empty:
        titles = {'protein_response': 'Protein response', 'predictive_markers': 'Predictive markers', 'sensitive_resistant': 'Sensitive vs resistant'}
        fig = published_volcano_figure(portal_family, effect_label=portal_effect_label, title=f"{titles[portal_kind]}: {portal_drug}")
        fig.update_layout(dragmode='lasso')
        portal_point_ids = {(curve, point): row[0] for curve, trace in enumerate(fig.data)
                            for point, row in enumerate(trace.customdata)}
        portal_plot = mo.ui.plotly(fig)
    return portal_plot, portal_point_ids


@app.cell
def _(mo, portal_family, portal_columns, portal_kind, portal_plot, portal_point_ids):
    portal_results, portal_shown_count = None, 0
    if portal_plot is not None:
        selected_ids = {portal_point_ids[(point['curveNumber'], point.get('pointIndex', point.get('pointNumber')))]
                        for point in portal_plot.value}
        family = portal_family.loc[portal_family.index.isin(selected_ids)] if selected_ids else portal_family
        _display = family.sort_values('p_value', kind='stable').loc[:, list(portal_columns)].rename(columns=portal_columns).rename_axis('Protein / assay').reset_index()
        portal_results = mo.ui.table(_display, selection='single' if portal_kind == 'sensitive_resistant' else None,
            initial_selection=[0] if portal_kind == 'sensitive_resistant' and len(_display) else None,
            pagination=True, page_size=10, label='Published portal results',
            format_mapping={name: '{:.3g}'.format for name in _display.columns[1:] if name != 'Result'})
        portal_shown_count = len(_display)
    return portal_results, portal_shown_count


@app.cell
def _(mo, portal_tables, portal_kind, portal_analysis, portal_sensitivity, portal_drug, portal_level, portal_family, portal_results, portal_alpha, portal_plot, portal_shown_count, published_box_figure, anchor_compounds):
    levels = {'p0': 'Baseline protein level (P0)', 'p1': 'Protein level after treatment (P1)', 'deltaP': 'Protein response (Δp)', 'deltap': 'Protein response (Δp)'}

    def _view():
        if portal_family.empty:
            return mo.md(f"No published results for {portal_drug} at this protein level.")
        basis = 'unadjusted p' if portal_kind == 'predictive_markers' else 'published adjusted p'
        parts = [portal_plot, mo.md(f"{len(portal_family)} published assays for **{portal_drug}**; {int(portal_family.classification.eq('positive').sum())} positive and {int(portal_family.classification.eq('negative').sum())} negative at {basis} ≤ {portal_alpha}."),
                 mo.md(f"Lasso points to filter the table ({portal_shown_count} of {len(portal_family)} shown)."),
                 portal_results]
        if portal_kind == 'sensitive_resistant' and len(portal_results.value):
            name = portal_results.value.iloc[0]['Protein / assay']
            parts.append(published_box_figure(portal_family.loc[name], {'sensitive': 'Sensitive', 'resistant': 'Resistant'},
                title=f"{name}: published summaries for {portal_drug}", y_label=levels[portal_level.value]))
        return mo.vstack(parts)

    def _compared():
        # Both axes are the portal's own pooled differences, joined by assay name; nothing is recalculated.
        if len(anchor_compounds) < 2 or portal_kind != 'protein_response':
            return None
        import plotly.express as _px
        other = anchor_compounds[1]
        _table = portal_tables['protein_response']
        _pair = _table.loc[_table.drug_name.eq(portal_drug), ['protein_name', 'diff']].merge(
            _table.loc[_table.drug_name.eq(other), ['protein_name', 'diff']], on='protein_name', suffixes=(' A', ' B'))
        if _pair.empty:
            return mo.md(f"No published protein response for {other}.")
        fig = _px.scatter(_pair, x='diff A', y='diff B', hover_name='protein_name',
            labels={'diff A': f'{portal_drug}: published difference', 'diff B': f'{other}: published difference'},
            title=f"Published protein response: {portal_drug} vs {other}")
        fig.add_hline(y=0, line_dash='dot', line_color='grey'); fig.add_vline(x=0, line_dash='dot', line_color='grey')
        return mo.vstack([mo.md(f"One point per protein ({len(_pair)}). Points on the diagonal respond alike."), fig])

    def published_section():
        if not anchor_compounds:
            return mo.md("Choose a compound above to see the CPPA portal's published analyses for it.")
        if portal_kind is None:
            return mo.md(f"The portal has no published analysis for **{portal_drug}**.")
        controls = [portal_analysis] + ([portal_sensitivity] if portal_kind == 'predictive_markers' else []) + ([] if portal_kind == 'protein_response' else [portal_level])
        notes = {
            'protein_response': "Change in each protein after treatment. Colored at FDR ≤ 0.05.",
            'predictive_markers': "Correlation between protein level and drug sensitivity. Colored at unadjusted p ≤ 0.05, so expect some false positives.",
            'sensitive_resistant': "Protein levels in sensitive vs resistant cell lines. Colored at adjusted p ≤ 0.05. Select a row to see both groups.",
        }
        parts = [mo.md(f"Published by the CPPA portal for **{portal_drug}**, pooled across studies."),
                 mo.hstack(controls, justify="start"), mo.md(notes[portal_kind]), _view()]
        compared = _compared()
        if compared is not None:
            parts.append(compared)
        parts.append(mo.accordion({'Published tables and sources': mo.vstack([
            mo.md("Values are as published; nothing is recalculated."),
            mo.ui.table([{'Table': name.replace('_', ' ').capitalize(), 'Rows': len(portal_tables[name]), 'Drugs': portal_tables[name].drug_name.nunique(),
                          'Assays': portal_tables[name].protein_name.nunique(), 'Portal document type': portal_tables['contract']['tables'][name]['portal_type']}
                         for name in ['protein_response', 'predictive_markers', 'sensitive_resistant']], selection=None, pagination=False, label='Published portal tables'),
        ])}))
        return mo.vstack(parts)
    return (published_section,)


@app.cell(hide_code=True)
def _(mo, dataset_description, assay_display, df_cppa_wide, df_treatment, population_summary, response_checks_panel):
    from epifolio.tcpa.text import phenotype_input_gates, phenotype_gate_description

    def about_tab():
        _groups = df_treatment.set_index('UID')['cell_line_name'].reindex(df_cppa_wide.index)
        _counts,_finite = population_summary(df_cppa_wide,_groups)
        _counts = _counts.rename(columns={'Observations':'Response conditions'})
        return mo.vstack([
            response_checks_panel(),
            mo.accordion({'Selected cell-line summary':mo.vstack([mo.md("Counts are conditions, not independent samples."),mo.ui.table(_counts),mo.ui.table(_finite)])}),
            mo.md(dataset_description(df_cppa_wide)),
            mo.accordion({'Treatment network':mo.md("Shows one condition's published Δp per protein. It is descriptive; it does not test sensitivity.")}),
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
