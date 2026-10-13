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

# Tabbed redesign of CCLE500_marimo.py: run with `uv run marimo edit tabcopy_ccle.py`.
# Dependencies, including the local tcpa package, are declared in pyproject.toml.
import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium", app_title="Cancer Cell Lines")


@app.cell
def _():
    import marimo as mo
    from epifolio.tcpa.text import label_options, assay_table, display_labels, dataset_description, dataset_options, available_groupings, source_group_label

    return mo, label_options, assay_table, display_labels, dataset_description, dataset_options, available_groupings, source_group_label


@app.cell
def _():
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from plotly.subplots import make_subplots
    from epifolio.tcpa.loaders import load_ccle

    cell_line_names, df_expr, df_meta, protein_list, tissue_list, tissue_types = load_ccle(anchor=globals().get("__file__"))
    feature_labels = df_expr.attrs["feature_labels"]
    sample_labels = df_expr.attrs["sample_labels"]
    tissue_labels = df_expr.attrs["tissue_labels"]
    return (feature_labels, sample_labels, tissue_labels, cell_line_names, df_expr, df_meta, protein_list, tissue_list, tissue_types, pd, px, go, np, make_subplots)


@app.cell
def _():
    from epifolio.tcpa.stats import correlate, correlate_one_vs_all, population_summary, two_group_test, feature_summary, zscore
    from epifolio.tcpa.figures import correlation_figure, heatmap_figure, volcano_frame, explain_empty_volcano
    return correlate, correlate_one_vs_all, population_summary, two_group_test, feature_summary, zscore, correlation_figure, heatmap_figure, volcano_frame, explain_empty_volcano


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Cancer Cell Lines

    Explore protein measurements from RPPA500 CCLE (Cancer Cell Line Encyclopedia).
    Compare tissues, inspect individual cell lines and find proteins that vary together.
    RPPA antibody assays measure proteins or specific protein modifications;
    values retain the source's units and normalization.

    Tissue comparisons use Welch tests with BH correction across eligible assays.
    Heatmaps can standardize each assay within the stated reference population
    using sample standard deviation. Counts and protein annotations are in the About the Data tab.
    """)
    return


@app.cell
def _(mo):
    # Remembers the open tab: the tabs cell re-runs (and would reset) whenever a control changes.
    get_tab, set_tab = mo.state("All Cell Lines")
    return get_tab, set_tab


@app.cell(hide_code=True)
def _(mo, get_tab, set_tab, one_protein_tab, one_cell_line_tab, relationships_tab, compare_tissues_tab, drug_sensitivity_tab, ranked_tab, heatmap_tab, all_cell_lines_tab, about_tab):
    sections = mo.ui.tabs(
        {
            "All Cell Lines": all_cell_lines_tab,
            "One Protein": one_protein_tab,
            "One Cell Line": one_cell_line_tab,
            "Protein Relationships": relationships_tab,
            "Compare Tissues": compare_tissues_tab,
            "Drug Sensitivity": drug_sensitivity_tab,
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


@app.cell(hide_code=True)
def _(label_options, feature_labels, mo, protein_list):
    # === Single Protein Explorer — controls ===
    # Menus are ordered by readable label and search assay names; values remain assay IDs.
    _assays = label_options(protein_list, feature_labels, sort=True)
    protein_dropdown = mo.ui.dropdown(
        options=_assays,
        value=next(iter(_assays)),
        searchable=True, allow_select_none=False,
        label="Protein / antibody assay",
    )
    plot_type_dropdown = mo.ui.dropdown(
        options=['Box Plot by Tissue', 'Histogram', 'Violin Plot', 'Strip Plot'],
        value='Box Plot by Tissue',
        label="Plot Type",
    )
    return plot_type_dropdown, protein_dropdown


@app.cell
def _(tissue_labels, feature_labels, df_expr, df_meta, mo, pd, plot_type_dropdown, protein_dropdown, px, np, feature_summary):
    def _view():
        protein = protein_dropdown.value
        plot_type = plot_type_dropdown.value

        if protein not in df_expr.columns:
            return mo.md(f"**Protein {feature_labels.get(protein, protein)} not found**")

        plot_df = pd.DataFrame({
            'Expression': df_expr[protein],
            'Tissue': df_meta['tissue'].map(tissue_labels),
            'Cell Line': df_meta['display_label']
        }).replace([np.inf, -np.inf], np.nan).dropna()

        if plot_type == 'Box Plot by Tissue':
            medians = plot_df.groupby('Tissue', observed=True)['Expression'].median().sort_values(ascending=False)
            plot_df['Tissue'] = pd.Categorical(plot_df['Tissue'], categories=medians.index, ordered=True)
            fig = px.box(
                plot_df.sort_values('Tissue'),
                x='Tissue', y='Expression', color='Tissue',
                title=f'{feature_labels[protein]} Expression Across Tissue Types',
                hover_data=['Cell Line']
            )
            fig.update_layout(xaxis_tickangle=-45, height=550, showlegend=False)
        elif plot_type == 'Histogram':
            fig = px.histogram(
                plot_df, x='Expression', color='Tissue',
                title=f'{feature_labels[protein]} Expression Distribution',
                nbins=50, barmode='overlay', opacity=0.6
            )
            fig.update_layout(height=500)
        elif plot_type == 'Violin Plot':
            fig = px.violin(
                plot_df, x='Tissue', y='Expression', color='Tissue',
                box=True, title=f'{feature_labels[protein]} Expression (Violin)'
            )
            fig.update_layout(xaxis_tickangle=-45, height=550, showlegend=False)
        else:  # Strip Plot
            fig = px.strip(
                plot_df, x='Tissue', y='Expression', color='Tissue',
                hover_data=['Cell Line'],
                title=f'{feature_labels[protein]} Expression (Individual Cell Lines)'
            )
            fig.update_layout(xaxis_tickangle=-45, height=550, showlegend=False)

        # Transposing per-tissue rows leaves object columns; restore numeric dtypes so the table sorts numerically.
        summary = pd.DataFrame({name: feature_summary(group[['Expression']]).loc['Expression'] for name, group in plot_df.groupby('Tissue', observed=True)}).T.infer_objects()
        summary = summary.sort_values('mean', ascending=False)
        from epifolio.tcpa.text import analysis_status
        summary['status'] = summary['status'].map(analysis_status)
        summary = summary.rename(columns={'n':'Finite n','mean':'Mean','std':'SD','variance':'Variance','range':'Range','min':'Minimum','max':'Maximum','status':'SD / variance status'})
        summary = summary.astype({'Finite n': 'int64'}).rename_axis('Tissue').reset_index().astype({'Tissue': str})
        return mo.vstack([
            fig,
            mo.ui.table(summary, selection=None, pagination=True, page_size=10, label=f"{feature_labels[protein]} summary by tissue",
                format_mapping={column: '{:.3f}'.format for column in ['Mean','SD','Variance','Range','Minimum','Maximum']}),
        ])

    def one_protein_tab():
        return mo.vstack([
            mo.md("Compare measured values across cell lines and tissues. Type an assay name in the menu to search it."),
            mo.hstack([protein_dropdown, plot_type_dropdown], justify="start"),
            _view(),
        ])
    return (one_protein_tab,)


@app.cell(hide_code=True)
def _(label_options, sample_labels, cell_line_names, mo):
    # === Cell Line Profiler — controls ===
    _cell_lines = label_options(cell_line_names, sample_labels, sort=True)
    cell_line_dropdown = mo.ui.dropdown(
        options=_cell_lines,
        value=next(iter(_cell_lines)),
        searchable=True, allow_select_none=False,
        label="Cell Line",
    )
    profile_top_n = mo.ui.slider(start=10, stop=50, step=5, value=30, label="Proteins shown", debounce=True)
    return cell_line_dropdown, profile_top_n


@app.cell
def _(sample_labels, zscore, display_labels, feature_labels, assay_table, cell_line_dropdown, df_expr, df_meta, feature_summary, np, go, make_subplots, mo, profile_top_n, interaction_view):
    def _view():
        cell_line = cell_line_dropdown.value
        top_n = profile_top_n.value

        if cell_line not in df_expr.index:
            return mo.md(f"**Cell line {cell_line} not found**")

        profile = df_expr.loc[cell_line].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        top_proteins = profile.head(top_n)
        bottom_proteins = profile.tail(top_n)

        fig = make_subplots(rows=1, cols=2, subplot_titles=['Highest Expression', 'Lowest Expression'])
        fig.add_trace(go.Bar(
            x=top_proteins.values, y=display_labels(top_proteins.index, feature_labels),
            orientation='h', marker_color='red', name='High'
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=bottom_proteins.values, y=display_labels(bottom_proteins.index, feature_labels),
            orientation='h', marker_color='blue', name='Low'
        ), row=1, col=2)

        tissue = df_meta.loc[cell_line, 'tissue']
        fig.update_layout(
            title=f'{sample_labels[cell_line]} Protein Profile ({tissue})',
            height=max(500, top_n * 18),
            showlegend=False
        )

        z_profile = zscore(df_expr, axis=0, ddof=1).loc[cell_line]
        shown_ids = top_proteins.index.union(bottom_proteins.index, sort=False)
        shown_profile = profile.loc[shown_ids].rename('Expression').to_frame()
        shown_profile['Profile panel'] = [
            'Highest and lowest' if key in top_proteins.index and key in bottom_proteins.index
            else 'Highest' if key in top_proteins.index else 'Lowest'
            for key in shown_ids
        ]
        shown_profile['Z-score'] = z_profile.reindex(shown_ids)

        profile_summary = feature_summary(profile.to_frame("profile")).loc["profile"]
        summary = [{
            "Cell line": sample_labels[cell_line], "Cell line ID": cell_line,
            "Tissue": tissue, "Proteins measured": int(profile.notna().sum()),
            "Mean expression": profile_summary['mean'],
        }]
        return mo.vstack([
            fig,
            mo.ui.table(summary, selection=None, pagination=False, label='Cell line summary',
                        format_mapping={'Mean expression': '{:.3f}'.format}),
            mo.md(f"The table lists the assays shown in the bars: up to {top_n} highest and {top_n} lowest measured values. Overlapping assays appear once. Z-scores use all CCLE cells as reference, with sample SD."),
            mo.ui.table(assay_table(shown_profile, feature_labels, {'Profile panel': 'Profile panel', 'Expression': 'Expression', 'Z-score': 'Z-score'}), selection=None,
                        pagination=True, label='Displayed protein assays',
                        format_mapping={'Expression': '{:.3f}'.format, 'Z-score': '{:.2f}'.format}),
        ])

    def one_cell_line_tab():
        return mo.vstack([
            mo.md("See the highest and lowest measured assay values for one cell line. Type a cell-line name in the menu to search it."),
            mo.hstack([cell_line_dropdown, profile_top_n], justify="start"),
            _view(),
        ])
    return (one_cell_line_tab,)


@app.cell(hide_code=True)
def _(label_options, feature_labels, mo, protein_list):
    # === Protein-Protein Correlation — controls ===
    protein_a_dropdown = mo.ui.dropdown(
        options=label_options(protein_list, feature_labels, sort=True), value=feature_labels[sorted(protein_list)[0]], searchable=True, allow_select_none=False, label="Protein / antibody assay A",
    )
    protein_b_dropdown = mo.ui.dropdown(
        options=label_options(protein_list, feature_labels, sort=True), value=feature_labels[sorted(protein_list)[1]], searchable=True, allow_select_none=False, label="Protein / antibody assay B",
    )
    corr_method = mo.ui.dropdown(options=['pearson', 'spearman'], value='pearson', label="Method")
    color_by_tissue = mo.ui.checkbox(value=True, label="Color by tissue")
    return color_by_tissue, corr_method, protein_a_dropdown, protein_b_dropdown


@app.cell
def _(df_expr, protein_a_dropdown, corr_method, correlate_one_vs_all):
    correlation_family = correlate_one_vs_all(df_expr, protein_a_dropdown.value,
        method=corr_method.value, inference=True)
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
def _(sample_labels, feature_labels, tissue_labels, color_by_tissue, corr_method, correlate, correlation_figure, df_expr, df_meta, mo, protein_a_dropdown, protein_b_dropdown, correlation_results):
    from epifolio.tcpa.text import analysis_status
    def _view():
        a = protein_a_dropdown.value
        _selected = correlation_results.value
        b = _selected.iloc[0]["Assay ID"] if len(_selected) else protein_b_dropdown.value
        if a not in df_expr or b not in df_expr:
            return mo.md("Select two available protein assays.")
        result = correlate(df_expr[a], df_expr[b], method=corr_method.value)
        fig = correlation_figure(result, labels=sample_labels, groups=df_meta['tissue'].map(tissue_labels) if color_by_tissue.value else None, x_label=feature_labels[a], y_label=feature_labels[b])
        return mo.vstack([fig, mo.md(f"{analysis_status(result['status'])}. Paired finite measurements: {result['n']}. This exploratory pair comparison does not establish significance; interactive pair selection has no correction for multiple comparisons.")])

    def relationships_tab():
        return mo.vstack([
            mo.md("Compare two assays using samples with finite measurements for both. Type an assay name in a menu to search it."),
            mo.hstack([protein_a_dropdown, protein_b_dropdown, corr_method, color_by_tissue], justify="start"),
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
def _(mo, df_expr, df_meta, feature_labels, protein_a_dropdown, corr_method,
      interaction_source, interaction_provenance, interaction_human_population,
      interaction_network, interaction_figure, network_search, network_scope, network_limit):
    def interaction_view():
        human, reason = interaction_human_population(df_expr.attrs['dataset_id'], df_meta)
        nodes, relations = interaction_network(df_expr, df_expr.attrs['feature_metadata'], interaction_source,
            human_population=human,
            reference=protein_a_dropdown.value if network_scope.value == 'Reference assay neighborhood' else None,
            search=network_search.value, method=corr_method.value)
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
def _(label_options, tissue_labels, mo, tissue_list):
    # === Tissue Comparison (Differential Expression) — controls ===
    tissue_a_dropdown = mo.ui.dropdown(
        options=label_options(sorted(tissue_list), tissue_labels), value=tissue_labels[sorted(tissue_list)[0]], label="Tissue A",
    )
    tissue_b_dropdown = mo.ui.dropdown(
        options=label_options(sorted(tissue_list), tissue_labels), value=tissue_labels[sorted(tissue_list)[1]], label="Tissue B",
    )
    fc_thresh = mo.ui.slider(start=0.1, stop=1.0, step=0.1, value=0.3, label="|mean difference| cutoff", debounce=True)
    return fc_thresh, tissue_a_dropdown, tissue_b_dropdown


@app.cell
def _(feature_labels, tissue_labels, df_expr, df_meta, two_group_test, volcano_frame, explain_empty_volcano, fc_thresh, tissue_a_dropdown, tissue_b_dropdown, mo, np, px):
    def _build_comparison():
        a, b = tissue_a_dropdown.value, tissue_b_dropdown.value
        if a == b:
            return None, None, {}, mo.md("Select different groups for comparison.")
        if a not in set(df_meta['tissue'].dropna()) or b not in set(df_meta['tissue'].dropna()):
            return None, None, {}, mo.md("Selected group has no samples.")
        results = two_group_test(df_expr, df_meta['tissue'], a, b)
        plotted = volcano_frame(results, alpha=0.05, min_effect=fc_thresh.value)
        from epifolio.tcpa.text import analysis_status
        plotted['display_protein'] = plotted.index.map(feature_labels)
        plotted['result_label'] = plotted['status'].map(analysis_status)
        plotted['comparison_label'] = plotted['classification'].map({'higher_a': 'Higher in A', 'higher_b': 'Higher in B', 'not_selected': 'Below cutoffs', 'invalid': 'Not testable'})
        plotted['assay_id'] = plotted.index
        fig = px.scatter(plotted, x='mean_diff', y='neg_log10_q', color='comparison_label', color_discrete_map={'Higher in A':'red','Higher in B':'blue','Below cutoffs':'gray','Not testable':'lightgray'}, hover_name='display_protein', custom_data=['assay_id'], hover_data=['mean_a','mean_b','n_a','n_b','p_value','q_value','result_label'], title=f"Welch comparison: {tissue_labels.get(a,a)} vs {tissue_labels.get(b,b)}", labels={'mean_diff':'Mean difference in existing units (A − B)','neg_log10_q':'−log₁₀(BH q-value)', 'comparison_label':'Comparison', 'result_label':'Result', 'mean_a':'Mean A', 'mean_b':'Mean B', 'n_a':'Finite n, A', 'n_b':'Finite n, B', 'p_value':'p-value', 'q_value':'BH q-value'})
        fig.add_hline(y=-np.log10(0.05), line_dash='dash')
        fig.add_vline(x=fc_thresh.value, line_dash='dash')
        fig.add_vline(x=-fc_thresh.value, line_dash='dash')
        explain_empty_volcano(fig, results)
        fig.update_layout(dragmode='lasso')
        point_ids = {(curve, point): row[0] for curve, trace in enumerate(fig.data)
                     for point, row in enumerate(trace.customdata)}
        return mo.ui.plotly(fig), plotted, point_ids, None

    comparison_plot, comparison_frame, comparison_point_ids, comparison_message = _build_comparison()
    return comparison_plot, comparison_frame, comparison_point_ids, comparison_message


@app.cell
def _(assay_table, feature_labels, mo, comparison_plot, comparison_frame, comparison_point_ids, comparison_message, tissue_a_dropdown, tissue_b_dropdown, fc_thresh):
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

    def compare_tissues_tab():
        return mo.vstack([
            mo.md("Compare mean protein measurements between two tissue groups."),
            mo.hstack([tissue_a_dropdown, tissue_b_dropdown, fc_thresh], justify="start"),
            _view(),
        ])
    return (compare_tissues_tab,)


@app.cell(hide_code=True)
def _(mo):
    # === Published Drug Sensitivity — controls ===
    from epifolio.tcpa.loaders import load_ccle_drug_sensitivity, load_ccle_drug_correlations
    from epifolio.tcpa.stats import published_family, published_family_counts
    from epifolio.tcpa.figures import published_volcano_figure
    drug_catalog = load_ccle_drug_sensitivity(anchor=globals().get("__file__"))
    drug_dataset = mo.ui.dropdown(options={'CTRPv2': 'ctrpv2', 'GDSC': 'gdsc'}, value='CTRPv2', label="Drug dataset")
    drug_search_by = mo.ui.dropdown(options=['Drug', 'Protein'], value='Drug', label="Search by")
    return drug_catalog, drug_dataset, drug_search_by, load_ccle_drug_correlations, published_family, published_family_counts, published_volcano_figure


@app.cell(hide_code=True)
def _(mo, drug_catalog, drug_dataset):
    # Lineage groups are the portal's own; a group without published pairs cannot be selected.
    drug_lineages = drug_catalog['datasets'][drug_dataset.value]['lineages']
    drug_lineage = mo.ui.dropdown(
        options={'All lineages' if key == 'all' else key.replace('_', ' ').capitalize(): key
                 for key in drug_lineages.index[drug_lineages.pairs.gt(0)]},
        value='All lineages', searchable=True, label="Portal lineage",
    )
    return drug_lineage, drug_lineages


@app.cell
def _(mo, feature_labels, drug_dataset, drug_lineage, drug_search_by, load_ccle_drug_correlations, published_family_counts):
    drug_rho, drug_p, drug_n = load_ccle_drug_correlations(drug_dataset.value, drug_lineage.value, anchor=globals().get("__file__"))
    _by_drug = drug_search_by.value == 'Drug'
    # One BH family per drug across assays, or per assay across drugs: the portal's rule.
    _counts = published_family_counts(drug_rho if _by_drug else drug_rho.T, drug_p if _by_drug else drug_p.T, alpha=0.1)
    _counts = _counts.loc[_counts.reported.gt(0)].sort_values('significant', ascending=False, kind='stable')
    _counts = _counts.rename(columns={'reported': 'Published pairs', 'positive': 'Significant positive', 'negative': 'Significant negative', 'significant': 'Significant'})
    if _by_drug:
        _counts = _counts.rename_axis('Drug').reset_index()
    else:
        _counts.insert(0, 'Assay ID', _counts.index)
        _counts.insert(0, 'Protein / assay', _counts.index.map(feature_labels))
        _counts = _counts.reset_index(drop=True)
    drug_families = mo.ui.table(_counts, selection='single', initial_selection=[0] if len(_counts) else None,
        pagination=True, page_size=10, label='Published correlation families')
    return drug_rho, drug_p, drug_n, drug_families


@app.cell
def _(assay_table, feature_labels, mo, drug_search_by, drug_rho, drug_p, drug_n, drug_families, published_family, published_volcano_figure):
    def _build_drug_plot():
        _selected = drug_families.value
        if not len(_selected):
            return None, {}, None, None, None
        by_drug = drug_search_by.value == 'Drug'
        key = _selected.iloc[0]['Drug' if by_drug else 'Assay ID']
        family = published_family(drug_rho[key] if by_drug else drug_rho.loc[key], drug_p[key] if by_drug else drug_p.loc[key], alpha=0.1)
        family['n'] = drug_n[key] if by_drug else drug_n.loc[key]
        family = family.loc[family.status.eq('ok')]
        family['result'] = family.classification.map({'positive': 'Positive', 'negative': 'Negative', 'not_selected': 'Below cutoff'})
        name = key if by_drug else feature_labels[key]
        fig = published_volcano_figure(family, labels=feature_labels if by_drug else None, hover=['n'],
            effect_label="Spearman's rho (assay vs drug AUC)", title=f"Published correlations: {name}")
        fig.update_xaxes(range=[-1, 1])
        columns = {'effect': "Spearman's rho", 'p_value': 'p-value', 'q_value': 'BH q-value', 'n': 'Cell lines', 'result': 'Result'}
        ranked = family.sort_values('p_value', kind='stable')
        members = (assay_table(ranked, feature_labels, columns) if by_drug
                   else ranked.loc[:, list(columns)].rename(columns=columns).rename_axis('Drug').reset_index())
        fig.update_layout(dragmode='lasso')
        point_ids = {(curve, point): row[0] for curve, trace in enumerate(fig.data)
                     for point, row in enumerate(trace.customdata)}
        return mo.ui.plotly(fig), point_ids, family, members, name

    drug_plot, drug_point_ids, drug_family, drug_members, drug_name = _build_drug_plot()
    return drug_plot, drug_point_ids, drug_family, drug_members, drug_name


@app.cell
def _(mo, drug_catalog, drug_dataset, drug_lineage, drug_lineages, drug_search_by, drug_families, feature_labels, drug_plot, drug_point_ids, drug_family, drug_members, drug_name):
    def _view():
        if drug_family is None:
            return mo.md("Select a row above to plot its published correlations.")
        # Marimo sends trace-local point indices, not customdata, in selection events.
        selected_ids = {drug_point_ids[(point['curveNumber'], point.get('pointIndex', point.get('pointNumber')))]
                        for point in drug_plot.value}
        identity = 'Assay ID' if drug_search_by.value == 'Drug' else 'Drug'
        members = (drug_members.loc[drug_members[identity].isin(selected_ids)] if selected_ids else drug_members).reset_index(drop=True)
        return mo.vstack([
            drug_plot,
            mo.md(f"{len(drug_family)} published {'assays' if drug_search_by.value == 'Drug' else 'drugs'} for **{drug_name}**; {int(drug_family.classification.eq('positive').sum())} significant positive and {int(drug_family.classification.eq('negative').sum())} significant negative at BH q ≤ 0.1 within this family. Cell lines per pair: {int(drug_family.n.min())}–{int(drug_family.n.max())}."),
            mo.md(f"Lasso or box-select points to filter the table. Use the plot's Reset state control to show all rows. Showing {len(members)} of {len(drug_members)} published correlations; selection does not recalculate q-values."),
            mo.ui.table(members, selection=None, pagination=True, label='Published protein–drug correlations',
                format_mapping={"Spearman's rho": '{:.3f}'.format, 'p-value': '{:.3g}'.format, 'BH q-value': '{:.3g}'.format}),
        ])

    def drug_sensitivity_tab():
        _features = drug_catalog['features']
        _empty = [key for key in drug_lineages.index[drug_lineages.pairs.eq(0)]]
        _release = ', '.join(sorted(set(drug_lineages.data_release)))
        return mo.vstack([
            mo.md("Find proteins whose level tracks drug sensitivity across cell lines. These are the TCPA portal's published results, shown unchanged."),
            mo.hstack([drug_dataset, drug_lineage, drug_search_by], justify="start"),
            mo.md(f"Each row is one {'drug' if drug_search_by.value == 'Drug' else 'assay'} with its count of significant correlations. Select a row to plot it."),
            drug_families,
            _view(),
            mo.accordion({'How to read these results': mo.md(
                f"Spearman's rho compares one RPPA assay with one drug's AUC across cell lines (portal release {_release}). "
                "AUC is the area under the dose–response curve, where a lower value conventionally means greater sensitivity, so a negative rho means higher protein accompanies greater sensitivity; the portal does not restate this direction. "
                "Significance follows the portal: Benjamini–Hochberg within the selected drug across assays, or within the selected assay across drugs, at q ≤ 0.1. Rho, p-values and cell-line counts are published values; only the q-values are calculated here, from the published p-values.\n\n"
                f"The portal publishes results, not the AUC measurements. These correlations therefore cannot be recalculated for the tissue selections used in other tabs, and the lineage groups are the portal's own. "
                f"{len(_features)} of {len(feature_labels)} local assays have published results; pairs the portal did not report are omitted."
                + (f" No {dict(ctrpv2='CTRPv2', gdsc='GDSC')[drug_dataset.value]} results are published for: {', '.join(k.replace('_', ' ').capitalize() for k in _empty)}." if _empty else ""))}),
        ])
    return (drug_sensitivity_tab,)


@app.cell(hide_code=True)
def _(mo):
    # === Protein Ranking — controls ===
    rank_by_dropdown = mo.ui.dropdown(
        options=['variance', 'range', 'mean'], value='variance', label="Rank by",
    )
    rank_top_n = mo.ui.slider(start=10, stop=50, step=5, value=25, label="Proteins shown", debounce=True)
    return rank_by_dropdown, rank_top_n


@app.cell
def _(assay_table, feature_labels, df_expr, mo, feature_summary, px, rank_by_dropdown, rank_top_n):
    def _view():
        by = rank_by_dropdown.value
        rankings = feature_summary(df_expr).sort_values(by, ascending=False).head(rank_top_n.value)
        from epifolio.tcpa.text import analysis_status
        rankings['display_protein'] = rankings.index.map(feature_labels)
        rankings['status'] = rankings['status'].map(analysis_status)
        fig = px.bar(rankings, x='display_protein', y=by, color=by, hover_data=['n','mean','std','variance','range','status'], title=f"Proteins ranked by {by}", labels={'display_protein':'Protein / assay','n':'Finite measurements','std':'Standard deviation','status':'SD / variance status'})
        return mo.vstack([fig, mo.ui.table(assay_table(rankings, feature_labels, {'n':'Finite measurements','mean':'Mean','std':'Standard deviation','variance':'Variance','range':'Range','min':'Minimum','max':'Maximum','status':'SD / variance status'}, rank=True), selection=None, pagination=True, label='Ranked protein assays')])

    def ranked_tab():
        return mo.vstack([
            mo.md("Find assays with the largest variance, range or mean across all cell lines."),
            mo.hstack([rank_by_dropdown, rank_top_n], justify="start"),
            _view(),
        ])
    return (ranked_tab,)


@app.cell(hide_code=True)
def _(label_options, tissue_labels, mo, tissue_list):
    # === Expression Heatmap — controls ===
    hm_n_proteins = mo.ui.slider(start=10, stop=100, step=10, value=40, label="Proteins shown", debounce=True)
    hm_n_cells = mo.ui.slider(start=50, stop=300, step=50, value=100, label="Cell lines shown", debounce=True)
    hm_colorscale = mo.ui.dropdown(
        options=['RdBu_r', 'Viridis', 'Plasma', 'Picnic', 'Spectral'], value='RdBu_r', label="Colors",
    )
    hm_filter_tissue = mo.ui.dropdown(
        options={'All': 'All', **label_options(sorted(tissue_list), tissue_labels)}, value='All', label="Tissue",
    )
    return hm_colorscale, hm_filter_tissue, hm_n_cells, hm_n_proteins


@app.cell
def _(sample_labels, feature_labels, df_expr, df_meta, feature_summary, zscore, heatmap_figure, hm_colorscale, hm_filter_tissue, hm_n_cells, hm_n_proteins, mo, hm_order, hm_cut):
    def _view():
        reference = df_expr if hm_filter_tissue.value == 'All' else df_expr.loc[df_meta['tissue'].eq(hm_filter_tissue.value)]
        features = feature_summary(reference)['variance'].dropna().sort_values(ascending=False).head(hm_n_proteins.value).index
        standardized = zscore(reference.loc[:, features], axis=0, ddof=1)
        displayed = standardized.sample(min(len(standardized), hm_n_cells.value), random_state=42)
        fig = heatmap_figure(displayed, cluster_mode=hm_order.value, cluster_cut=hm_cut.value, raw_data=reference.loc[displayed.index, features], feature_labels=feature_labels, sample_labels=sample_labels, colorscale=hm_colorscale.value, robust=True, title='Protein patterns across cell lines', colorbar_title='Z-score')
        return mo.vstack([fig, mo.md(f"Colors show z-scores relative to all {len(reference)} cell lines in the selected tissue scope, before display sampling (using sample standard deviation). Color limits use the 1st and 99th percentiles; hover shows the unchanged values.")])

    def heatmap_tab():
        return mo.vstack([
            mo.md("Compare the most variable assays across a repeatable sample of cell lines in the selected tissue."),
            mo.hstack([hm_n_proteins, hm_n_cells, hm_colorscale, hm_filter_tissue], justify="start"),
            mo.accordion({'Optional local clustering': mo.vstack([mo.hstack([hm_order,hm_cut],justify='start'),mo.md("Clusters use the displayed matrix with Euclidean distance and average linkage. Only complete entities are clustered; incomplete entities remain at the end with no cluster. No missing values are filled. The cut requests at most this many clusters; ties can give fewer. This local method is separate from the portal's unspecified clustering.")])}),
            _view(),
        ])
    return (heatmap_tab,)


@app.cell(hide_code=True)
def _(mo):
    # === Full Expression Heatmap (All Data) — controls ===
    full_colorscale = mo.ui.dropdown(
        options=['tealrose', 'RdBu_r', 'Viridis', 'Plasma', 'Picnic'], value='tealrose', label="Colors",
    )
    normalize_checkbox = mo.ui.checkbox(value=False, label="Z-score normalize")
    return full_colorscale, normalize_checkbox


@app.cell
def _(feature_labels, sample_labels, df_expr, full_colorscale, normalize_checkbox, zscore, heatmap_figure, mo, hm_order, hm_cut):
    def _view():
        data = zscore(df_expr, axis=0, ddof=1) if normalize_checkbox.value else df_expr
        fig_full = heatmap_figure(data, cluster_mode=hm_order.value, cluster_cut=hm_cut.value, raw_data=df_expr if normalize_checkbox.value else None, compact_hover=True, feature_labels=feature_labels, sample_labels=sample_labels, colorscale=full_colorscale.value, robust=True, title='All cell lines and protein assays', colorbar_title='Z-score' if normalize_checkbox.value else 'Expression')
        return mo.vstack([fig_full, mo.md(f"All {len(df_expr)} cell lines are shown. Optional z-scores use each assay’s mean and sample SD across all cell lines. Color limits use the 1st and 99th percentiles; hover shows the unchanged values.")])

    def all_cell_lines_tab():
        return mo.vstack([
            mo.md("View every cell line and measured assay; the Protein Heatmap tab's tissue filter does not apply here."),
            mo.hstack([full_colorscale, normalize_checkbox], justify="start"),
            mo.accordion({'Optional local clustering': mo.vstack([mo.hstack([hm_order,hm_cut],justify='start'),mo.md("Clusters use the displayed matrix with Euclidean distance and average linkage. Only complete entities are clustered; incomplete entities remain at the end with no cluster. No missing values are filled. The cut requests at most this many clusters; ties can give fewer. This local method is separate from the portal's unspecified clustering.")])}),
            _view(),
        ])
    return (all_cell_lines_tab,)



@app.cell(hide_code=True)
def _(mo, dataset_description, df_expr, df_meta, population_summary):
    from epifolio.tcpa.text import assay_display

    from epifolio.tcpa.text import phenotype_input_gates, phenotype_gate_description

    def about_tab():
        _counts, _finite = population_summary(df_expr, df_meta['tissue'])
        return mo.vstack([mo.md(dataset_description(df_expr)),
            mo.accordion({'Analyses requiring additional inputs': mo.vstack([mo.md(phenotype_gate_description('cell_line')), mo.ui.table(phenotype_input_gates('cell_line'), selection=None, pagination=False, label='Phenotype input gates')])}),
            mo.accordion({"Selected lineage summary": mo.vstack([mo.md("Counts describe the selected observations; finite n counts measurements, not whether an assay was performed."),mo.ui.table(_counts),mo.ui.table(_finite)]), "Protein and Antibody Details": mo.ui.table(assay_display(df_expr.attrs['panel_metadata']))})])
    return (about_tab,)


@app.cell(hide_code=True)
def _(mo):
    hm_order = mo.ui.dropdown(options={'Current order':'source','Local row clustering':'rows','Local assay clustering':'columns','Local rows and assays':'both'}, value='Current order', label='Heatmap order')
    hm_cut = mo.ui.slider(start=1,stop=10,value=3,label='Maximum local clusters',debounce=True)
    return hm_order, hm_cut


if __name__ == "__main__":
    app.run()
