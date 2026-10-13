"""Generic result-to-plot adapters; no data loading or notebook state."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def regimen_connectivity_figure(edges, *, feature_labels=None, edge_limit=50):
    """Single full-regimen star; strongest |Δp| drawing, deterministic ID ties.

    The returned draw table is a display restriction. The complete matching edge
    table stays with the caller, preserving all source context and response values.
    """
    if isinstance(edge_limit,bool) or not isinstance(edge_limit,int) or not 1 <= edge_limit <= 100:
        raise ValueError('Connectivity draw limit must be an integer from 1 to 100')
    if not edges.assay_id.is_unique or (not edges.empty and edges.regimen_id.nunique()!=1):
        raise ValueError('Connectivity requires one full regimen and unique assay edges')
    if not np.isfinite(edges.published_delta_p.to_numpy(dtype=float)).all():
        raise ValueError('Connectivity edges require finite published responses')
    ranked = edges.assign(abs_delta=edges.published_delta_p.abs()).sort_values(
        ['abs_delta','assay_id'],ascending=[False,True],kind='stable')
    drawn = ranked.iloc[:edge_limit].drop(columns='abs_delta').reset_index(drop=True)
    fig = go.Figure()
    meta = dict(kind='within_condition_regimen_connectivity',matching_edges=len(edges),
        drawn_edges=len(drawn),edge_limit=edge_limit,order='absolute delta p descending; assay ID ties',
        aggregation='none; one complete regimen',layout='sorted assay-ID star')
    if not drawn.empty:
        first = drawn.iloc[0]
        meta.update(condition_uid=first.UID,set_id=first['set'],regimen_id=first.regimen_id)
        ids = sorted(drawn.assay_id)
        angles = np.linspace(0,2*np.pi,len(ids),endpoint=False)
        positions = {key:(float(np.cos(angle)),float(np.sin(angle))) for key,angle in zip(ids,angles)}
        bound = float(drawn.published_delta_p.abs().max()) or 1.
        arrows = []; data = []; colors = []
        for row in drawn.itertuples(index=False):
            x,y = positions[row.assay_id]
            color = '#d62728' if row.published_delta_p>0 else '#1f77b4' if row.published_delta_p<0 else '#777777'
            width = 1+3*abs(row.published_delta_p)/bound
            fig.add_trace(go.Scatter(x=[0,x],y=[0,y],mode='lines',line=dict(color=color,width=width),hoverinfo='skip',showlegend=False))
            arrows.append(dict(x=.8*x,y=.8*y,ax=.7*x,ay=.7*y,xref='x',yref='y',axref='x',ayref='y',
                showarrow=True,arrowhead=2,arrowwidth=width,arrowcolor=color))
            data.append([row.assay_id,row.published_delta_p,row.UID,row.set,row.cell_line_name,row.response_sign])
            colors.append(color)
        node_ids = drawn.assay_id.tolist()
        fig.add_trace(go.Scatter(x=[positions[key][0] for key in node_ids],y=[positions[key][1] for key in node_ids],
            mode='markers+text',text=_aligned_labels(pd.Index(node_ids),feature_labels),textposition='top center',
            marker=dict(size=11,color=colors),customdata=np.array(data,dtype=object),showlegend=False,
            hovertemplate='Assay ID: %{customdata[0]}<br>Published Δp: %{customdata[1]}<br>Condition: %{customdata[2]}<br>Set: %{customdata[3]}<br>Cell line: %{customdata[4]}<br>Sign: %{customdata[5]}<extra></extra>'))
        context = {field:first[field] for field in edges if field not in {
            'regimen_id','assay_id','published_delta_p','response_sign','raw_name','assay_label','gene_symbols',
            'candidate_gene_symbols','annotation_status','target_annotation_status','modification','residue_site'}}
        fig.add_trace(go.Scatter(x=[0],y=[0],mode='markers+text',text=['Complete regimen'],textposition='bottom center',
            marker=dict(size=18,color='#636efa'),customdata=np.array([[first.regimen_id,pd.Series(context,dtype=object).to_json(double_precision=15)]],dtype=object),
            hovertemplate='%{customdata[0]}<br>Full source condition: %{customdata[1]}<extra></extra>',showlegend=False))
        fig.update_layout(annotations=arrows)
    fig.update_layout(title='Within-condition treatment connectivity',height=500,template='plotly_white',meta=meta,
        margin=dict(l=25,r=25,t=55,b=25),xaxis=dict(visible=False,range=[-1.4,1.4]),
        yaxis=dict(visible=False,range=[-1.4,1.4],scaleanchor='x'))
    return fig, drawn


def interaction_figure(nodes, edges, *, labels=None, edge_limit=200):
    """Sorted-ID circle; source-directed arrows with descriptive correlation color.

    Source records are combined only for drawing duplicate directed assay pairs.
    The caller retains the complete source-relation table, including invalid r.
    """
    if isinstance(edge_limit, bool) or not isinstance(edge_limit, int) or not 1 <= edge_limit <= 500:
        raise ValueError('Network edge limit must be an integer between 1 and 500')
    if not nodes.index.is_unique:
        raise ValueError('Network assay identities must be unique')
    pairs = []
    for (a,b), records in edges.groupby(['assay_a','assay_b'],sort=True):
        if a not in nodes.index or b not in nodes.index:
            raise ValueError('Network relation references an absent assay')
        row = records.sort_values('SIGNOR_ID',kind='stable').iloc[0]
        pairs.append(dict(assay_a=a, assay_b=b, r=row.r, finite_n=int(row.finite_n),
            correlation_status=row.correlation_status,
            source_records='; '.join(sorted(records.SIGNOR_ID)),
            source_record_count=len(records)))
    complete = pd.DataFrame(pairs,columns=['assay_a','assay_b','r','finite_n',
                                          'correlation_status','source_records','source_record_count'])
    drawn = complete.iloc[:edge_limit].copy()
    ids = sorted(set(drawn.assay_a)|set(drawn.assay_b))
    angles = np.linspace(0,2*np.pi,len(ids),endpoint=False)
    positions = {key:(float(np.cos(angle)),float(np.sin(angle))) for key,angle in zip(ids,angles)}
    fig = go.Figure(); annotations = []; hover_x = []; hover_y = []; hover_data = []
    for row in drawn.itertuples(index=False):
        x0,y0 = positions[row.assay_a]; x1,y1 = positions[row.assay_b]
        color = '#777777' if not np.isfinite(row.r) else '#d62728' if row.r >= 0 else '#1f77b4'
        width = 1 if not np.isfinite(row.r) else 1+3*abs(row.r)
        fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode='lines',line=dict(color=color,width=width),
                                hoverinfo='skip',showlegend=False))
        annotations.append(dict(x=x0+.8*(x1-x0),y=y0+.8*(y1-y0),ax=x0+.7*(x1-x0),ay=y0+.7*(y1-y0),
            xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=2,arrowcolor=color,arrowwidth=width))
        hover_x.append((x0+x1)/2);hover_y.append((y0+y1)/2)
        hover_data.append([row.assay_a,row.assay_b,row.r,row.finite_n,row.correlation_status,row.source_records])
    fig.add_trace(go.Scatter(x=hover_x,y=hover_y,mode='markers',marker=dict(size=9,color='#777777',opacity=.5),
        customdata=np.array(hover_data,dtype=object).reshape(-1,6),showlegend=False,
        hovertemplate='%{customdata[0]} → %{customdata[1]}<br>Measured r: %{customdata[2]:.4f}<br>Finite n: %{customdata[3]}<br>Status: %{customdata[4]}<br>Source records: %{customdata[5]}<extra></extra>'))
    node_data = [[key,nodes.loc[key,'mapping_status'],nodes.loc[key,'gene_symbols'],
                  nodes.loc[key].get('residue_site','unknown')] for key in ids]
    fig.add_trace(go.Scatter(x=[positions[key][0] for key in ids],y=[positions[key][1] for key in ids],
        mode='markers+text',text=_aligned_labels(pd.Index(ids),labels),textposition='top center',
        marker=dict(size=12,color='#636efa'),customdata=np.array(node_data,dtype=object).reshape(-1,4),
        hovertemplate='%{customdata[0]}<br>Mapping: %{customdata[1]}<br>Targets: %{customdata[2]}<br>Assay site: %{customdata[3]}<extra></extra>',showlegend=False))
    fig.update_layout(height=500,template='plotly_white',annotations=annotations,
        margin=dict(l=20,r=20,t=30,b=20),xaxis=dict(visible=False,range=[-1.3,1.3]),
        yaxis=dict(visible=False,range=[-1.3,1.3],scaleanchor='x'),
        meta=dict(total_pairs=len(complete),drawn_pairs=len(drawn),edge_limit=edge_limit,
                  layout='sorted assay-ID circle',color='red nonnegative r; blue negative r; gray unavailable',
                  direction='curated causal source; not inferred from correlation'))
    return fig, drawn


def volcano_frame(results, alpha=0.05, min_effect=0.0):
    """Classify the complete BH family; clamp zero q only for display."""
    if not 0 < alpha < 1 or not np.isfinite(min_effect) or min_effect < 0:
        raise ValueError("Require 0 < alpha < 1 and a finite nonnegative effect cutoff")
    out = results.copy(deep=True)
    q = out['q_value']
    effect = out['mean_diff']
    valid = out['status'].eq('ok') & np.isfinite(q) & q.between(0, 1) & np.isfinite(effect)
    out['neg_log10_q'] = -np.log10(q.where(valid).clip(lower=np.nextafter(0., 1.)))
    out['classification'] = 'invalid'
    out.loc[valid, 'classification'] = 'not_selected'
    out.loc[valid & (q < alpha) & (effect > min_effect), 'classification'] = 'higher_a'
    out.loc[valid & (q < alpha) & (effect < -min_effect), 'classification'] = 'higher_b'
    return out


def published_volcano_figure(family, *, labels=None, effect_label, title, hover=()):
    """Published effect against −log₁₀ published p; color follows the family classification."""
    frame = family.loc[family.status.eq('ok')].copy()
    frame['name'] = _aligned_labels(frame.index, labels) if labels else frame.index.astype(str)
    frame['result'] = frame.classification.map({'positive': 'Positive', 'negative': 'Negative', 'not_selected': 'Below cutoff'})
    frame['identity'] = frame.index
    fig = px.scatter(frame, x='effect', y='neg_log10_p', color='result', hover_name='name', custom_data=['identity'],
        color_discrete_map={'Positive': 'red', 'Negative': 'blue', 'Below cutoff': 'gray'},
        category_orders={'result': ['Positive', 'Negative', 'Below cutoff']},
        hover_data={'result': False, 'identity': False, 'p_value': True, 'q_value': bool(frame.q_value.notna().any()),
                    **{name: True for name in hover}},
        labels={'effect': effect_label, 'neg_log10_p': '−log₁₀(published p-value)', 'result': 'Result',
                'p_value': 'p-value', 'q_value': 'Adjusted p-value', 'n': 'Cell lines'}, title=title)
    if frame.empty:
        fig.add_annotation(text='No published results for this selection', showarrow=False, xref='paper', yref='paper', x=0.5, y=0.5)
    return fig


def published_box_figure(row, groups, *, title, y_label):
    """Boxes drawn from published five-number summaries; no observations are available."""
    fig = go.Figure()
    for key, label in groups.items():
        fig.add_trace(go.Box(name=label, x=[label], lowerfence=[row[f'{key}_min']], q1=[row[f'{key}_q1']],
            median=[row[f'{key}_median']], q3=[row[f'{key}_q3']], upperfence=[row[f'{key}_max']]))
    fig.update_layout(title=title, yaxis_title=y_label, showlegend=False, height=420)
    return fig


def _aligned_labels(index, labels):
    if labels is None:
        return [str(key) for key in index]
    if isinstance(labels, pd.Series) and not labels.index.is_unique:
        raise ValueError('Display label identities must be unique')
    return [str(labels.get(key, key)) for key in index]


def correlation_figure(result, labels=None, groups=None, x_label='x', y_label='y'):
    """Plot exactly the statistical pairs, with a single overall OLS line."""
    x, y = result['x'], result['y']
    if not x.index.is_unique or not x.index.equals(y.index):
        raise ValueError('Correlation pairs require identical unique ordered IDs')
    if len(x) != result['n'] or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError('Plot rows must equal the finite statistical sample')
    frame = pd.DataFrame({'x': x, 'y': y})
    frame['label'] = _aligned_labels(frame.index, labels)
    if groups is None:
        frame['group'] = 'Samples'
    else:
        if not isinstance(groups, pd.Series) or not groups.index.is_unique:
            raise ValueError('Color groups must be uniquely keyed Series')
        frame['group'] = groups.reindex(frame.index).astype(object).where(lambda s: s.notna(), 'Unknown').astype(str)
    fig = go.Figure()
    for group in frame['group'].unique():
        part = frame.loc[frame['group'].eq(group)]
        fig.add_trace(go.Scatter(x=part.x, y=part.y, mode='markers', name=group,
            customdata=np.column_stack([part.index.astype(str), part.label, part.group]),
            hovertemplate='%{customdata[1]}<br>ID: %{customdata[0]}<br>Group: %{customdata[2]}<br>x: %{x}<br>y: %{y}<extra></extra>'))
    if result['status'] == 'ok' and len(x) >= 2 and np.ptp(x.to_numpy()) > 0:
        # OLS with intercept, across all displayed points; independent of color.
        xc = x.to_numpy() - x.mean()
        slope = np.dot(xc, y.to_numpy() - y.mean()) / np.dot(xc, xc)
        line_x = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(x=line_x, y=y.mean() + slope * (line_x - x.mean()),
            mode='lines', name='Overall trend', hoverinfo='skip'))
    from .text import analysis_status
    annotation = f"{result.get('method', 'correlation').capitalize()} correlation<br>{result['n']} paired measurements"
    if result['status'] != 'ok':
        annotation += "<br>" + analysis_status(result['status'])
    if result['status'] == 'ok':
        annotation += f"<br>r = {result['r']:.4f}"
        if np.isfinite(result['p_value']):
            annotation += f"<br>p = {result['p_value']:.2e} (single comparison)"
        if np.isfinite(result.get('ci_low', np.nan)):
            annotation += f"<br>95% confidence interval [{result['ci_low']:.3f}, {result['ci_high']:.3f}]"
        else:
            annotation += '<br>Confidence interval not estimated'
        if result.get('method') == 'spearman' and np.isfinite(result['p_value']):
            annotation += '<br>Asymptotic p; unreliable for small samples'
    fig.add_annotation(text=annotation, xref='paper', yref='paper', x=0.01, y=0.99, showarrow=False,
        xanchor='left', yanchor='top', bgcolor='rgba(255,255,255,0.8)')
    fig.update_layout(title=f'Correlation: {x_label} vs {y_label}', xaxis_title=x_label,
        yaxis_title=y_label, template='plotly_white')
    return fig


def heatmap_figure(data, feature_labels=None, sample_labels=None, colorscale='RdBu_r',
                   robust=False, title='Heatmap', colorbar_title='Value', raw_data=None, compact_hover=False, cluster_mode='source', cluster_cut=3, row_id_label='Sample ID'):
    """Zero-centered colors; optional 1st/99th percentile color-only clipping.

    z and hover both retain the supplied values. Standardization, if desired,
    must happen explicitly in the caller with a declared reference population.
    """
    if not data.index.is_unique or not data.columns.is_unique:
        raise ValueError('Heatmap identities must be unique')
    if raw_data is not None and (not data.index.equals(raw_data.index) or not data.columns.equals(raw_data.columns)):
        raise ValueError('Raw hover data must have identical ordered identities')
    if cluster_mode != 'source':
        from .stats import cluster_heatmap
        data, row_clusters, column_clusters = cluster_heatmap(data, cluster_mode, cluster_cut)
        if raw_data is not None:
            raw_data = raw_data.loc[data.index, data.columns]
        def cluster_labels(keys, labels, clusters):
            return {key: f"{label} | Local cluster {clusters[key] if pd.notna(clusters[key]) else 'unassigned'}"
                    for key,label in zip(keys,_aligned_labels(keys,labels))}
        if cluster_mode in ('rows','both'):
            sample_labels = cluster_labels(data.index,sample_labels,row_clusters)
        if cluster_mode in ('columns','both'):
            feature_labels = cluster_labels(data.columns,feature_labels,column_clusters)
    values = data.to_numpy(dtype=float, copy=True)
    values[~np.isfinite(values)] = np.nan
    finite = values[np.isfinite(values)]
    fig = go.Figure()
    if not finite.size:
        fig.add_annotation(text='No finite values in this selection', showarrow=False)
        fig.update_layout(title=title)
        return fig
    limits = np.quantile(finite, [.01, .99]) if robust else np.array([finite.min(), finite.max()])
    bound = float(np.max(np.abs(limits))) or 1.0
    hover = values.copy() if raw_data is None else np.stack([values, raw_data.to_numpy(dtype=float, copy=True)], axis=-1)
    template = '%{y}<br>%{x}<br>Value: %{customdata}<extra></extra>' if raw_data is None else '%{y}<br>%{x}<br>Displayed value: %{customdata[0]}<br>Raw value: %{customdata[1]}<extra></extra>'
    if compact_hover:
        hover = None if raw_data is None else raw_data.to_numpy(dtype=float, copy=True)
        template = '%{y}<br>%{x}<br>Value: %{z}<extra></extra>' if raw_data is None else '%{y}<br>%{x}<br>Displayed value: %{z}<br>Raw value: %{customdata}<extra></extra>'
    x_labels = _aligned_labels(data.columns, feature_labels)
    y_labels = _aligned_labels(data.index, sample_labels)
    feature_metadata = data.attrs.get('feature_metadata')
    raw_names = feature_metadata.raw_name.to_dict() if feature_metadata is not None else {}
    # Axis categories encode identity once per row/column, not once per matrix cell.
    x_hover = [f'{label}<br>Assay ID: {key}<br>Raw assay: {raw_names.get(key, key)}'
               for key, label in zip(data.columns, x_labels)]
    y_hover = [f'{label}<br>{row_id_label}: {key}' for key, label in zip(data.index, y_labels)]
    fig.add_trace(go.Heatmap(z=values, x=x_hover, y=y_hover, customdata=hover, colorscale=colorscale,
        zmid=0, zmin=-bound, zmax=bound, colorbar_title=colorbar_title,
        hovertemplate=template))
    x_ticks = list(range(0, len(x_hover), max(1, int(np.ceil(len(x_hover) / 30)))))
    y_ticks = list(range(0, len(y_hover), max(1, int(np.ceil(len(y_hover) / 30)))))
    fig.update_layout(title=title, template='plotly_white',
        xaxis=dict(tickmode='array', tickvals=[x_hover[i] for i in x_ticks], ticktext=[x_labels[i] for i in x_ticks]),
        yaxis=dict(tickmode='array', tickvals=[y_hover[i] for i in y_ticks], ticktext=[y_labels[i] for i in y_ticks]))
    if cluster_mode != 'source':
        fig.add_annotation(text=f'Local clustering: displayed values, Euclidean, average linkage; max {cluster_cut} clusters. Incomplete entities unassigned; no imputation.',
            xref='paper',yref='paper',x=0,y=-.18,showarrow=False)
    if robust:
        fig.add_annotation(text='Colors clipped at symmetric 1st–99th percentile limits; hover retains values',
            xref='paper', yref='paper', x=0, y=1.08, showarrow=False)
    return fig


def explain_empty_volcano(fig, results):
    """Explain zero eligible tests on the plot, including finite denominators."""
    eligible = results['status'].eq('ok') & results['q_value'].notna()
    if not eligible.any():
        def span(column):
            values = results[column]
            return f'{int(values.min())}–{int(values.max())}' if len(values) else '0'
        from .text import analysis_status
        reasons = '; '.join(f'{count} {analysis_status(status).lower()}' for status, count in results['status'].value_counts().items())
        fig.add_annotation(text=f'No eligible comparisons: {reasons}<br>'
            f'Finite samples per assay: A={span("n_a")}; B={span("n_b")}. Welch requires ≥3 per group.',
            xref='paper', yref='paper', x=.5, y=.5, showarrow=False)
    return fig


def response_curve_figure(values, feature_labels=None, highlight=None):
    """Full finite published single-condition Δp, stable descending order."""
    from .stats import feature_summary
    if not isinstance(values, pd.Series) or not values.index.is_unique:
        raise ValueError('Responses require uniquely keyed assay values')
    summary = feature_summary(values.to_frame().T)
    ranked = summary.loc[summary.n.gt(0)].sort_values('mean', ascending=False, kind='stable')
    table = pd.DataFrame({'Assay ID':ranked.index, 'Protein / assay':_aligned_labels(ranked.index,feature_labels),
                          'Published Δp':ranked['mean'].to_numpy(), 'Finite n':ranked.n.to_numpy()})
    table.insert(0, 'Rank', np.arange(1,len(table)+1))
    fig = go.Figure(go.Scatter(x=table['Rank'], y=table['Published Δp'], mode='lines+markers',
        name='Published Δp', customdata=table[['Assay ID','Protein / assay']].to_numpy(),
        hovertemplate='Rank: %{x}<br>%{customdata[1]}<br>Assay ID: %{customdata[0]}<br>Published Δp: %{y}<extra></extra>'))
    selected = table.loc[table['Assay ID'].eq(highlight)]
    if len(selected):
        fig.add_trace(go.Scatter(x=selected['Rank'],y=selected['Published Δp'],mode='markers',
            marker=dict(size=12,color='orange'),name='Selected assay',
            customdata=selected[['Assay ID','Protein / assay']].to_numpy(),
            hovertemplate='%{customdata[1]}<br>Assay ID: %{customdata[0]}<br>Published Δp: %{y}<extra></extra>'))
    fig.update_layout(title=f'Full published response curve · {values.name}',xaxis_title='Rank by signed Δp',
                      yaxis_title='Published Δp',height=500,template='plotly_white')
    fig.add_hline(y=0,line_dash='dash',line_color='gray')
    return fig, table
