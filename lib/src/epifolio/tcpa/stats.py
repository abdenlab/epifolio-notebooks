"""Identity-aligned, finite-only statistical operations with explicit validity.

P-values and BH families are calculated before caller display filtering. All
operations return new objects and retain the original feature ordering.
"""
import numbers

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

MIN_N = 3
SUMMARY_COLUMNS = ['n', 'mean', 'std', 'variance', 'min', 'max', 'range', 'status']
CORRELATION_COLUMNS = ['r', 'p_value', 'q_value', 'n', 'status', 'ci_low', 'ci_high', 'ci_method']
GROUP_COLUMNS = ['n_a', 'n_b', 'mean_a', 'mean_b', 'mean_diff', 'statistic', 'p_value', 'q_value', 'status']


def _identity(index):
    # to_frame also supports MultiIndex, where Index.hasnans is unavailable.
    if not index.is_unique or index.to_frame(index=False).isna().any().any():
        raise ValueError('Identities must be unique and nonmissing')


def _numeric(data):
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError('Expected a pandas Series or DataFrame')
    _identity(data.index)
    if isinstance(data, pd.DataFrame):
        _identity(data.columns)
        dtypes = data.dtypes
    else:
        dtypes = [data.dtype]
    if any(not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_complex_dtype(dtype) for dtype in dtypes):
        raise TypeError('Values must be real numeric data')
    return data.astype(float).replace([np.inf, -np.inf], np.nan)


def _dataframe(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError('Expected a pandas DataFrame')
    return _numeric(data)


def _minimum(min_n):
    if isinstance(min_n, bool) or not isinstance(min_n, numbers.Integral) or min_n < MIN_N:
        raise ValueError('min_n must be an integer >= 3')


def _method(method):
    if method not in ('pearson', 'spearman'):
        raise ValueError('method must be pearson or spearman')


def _table(rows, features, columns):
    return pd.DataFrame(rows, index=pd.Index(features, name='feature'), columns=columns)


def _adjust(table):
    valid = table.status.eq('ok') & np.isfinite(table.p_value.astype(float))
    table['q_value'] = np.nan
    if valid.any():
        table.loc[valid, 'q_value'] = scipy_stats.false_discovery_control(
            table.loc[valid, 'p_value'].to_numpy(dtype=float), method='bh')
    return table


def published_family(effect, p_value, *, alpha, adjusted='bh'):
    """Classify one published result family; effects and p-values are never recomputed.

    ``adjusted`` is 'bh' (BH over every finite p in the family, the portal's
    own rule), a Series of published adjusted values, or None for the
    unadjusted p-value. Significance is inclusive, matching the portal.
    """
    if not 0 < alpha < 1:
        raise ValueError('Require 0 < alpha < 1')
    _identity(effect.index)
    if not effect.index.equals(p_value.index):
        raise ValueError('Published effects and p-values must share identities')
    table = pd.DataFrame({'effect': effect.astype(float), 'p_value': p_value.astype(float)})
    valid = np.isfinite(table.effect) & table.p_value.between(0, 1)
    table['status'] = np.where(valid, 'ok', 'not_reported')
    if isinstance(adjusted, str):
        if adjusted != 'bh':
            raise ValueError("adjusted must be 'bh', a Series or None")
        table = _adjust(table)
    else:
        table['q_value'] = np.nan if adjusted is None else adjusted.reindex(table.index).astype(float).where(valid)
    decisive = table.p_value if adjusted is None else table.q_value
    table['neg_log10_p'] = -np.log10(table.p_value.where(valid).clip(lower=np.nextafter(0., 1.)))
    table['classification'] = np.where(valid, 'not_selected', 'invalid')
    table.loc[valid & (decisive <= alpha) & (table.effect > 0), 'classification'] = 'positive'
    table.loc[valid & (decisive <= alpha) & (table.effect < 0), 'classification'] = 'negative'
    return table


def published_family_counts(effect, p_value, *, alpha):
    """Count significant published results per column; each column is one BH family."""
    if not 0 < alpha < 1:
        raise ValueError('Require 0 < alpha < 1')
    _identity(effect.index); _identity(effect.columns)
    if not effect.index.equals(p_value.index) or not effect.columns.equals(p_value.columns):
        raise ValueError('Published effects and p-values must share identities')
    r, p = effect.to_numpy(dtype=float), p_value.to_numpy(dtype=float)
    valid = np.isfinite(r) & (p >= 0) & (p <= 1)
    q = np.full(p.shape, np.nan)
    for column in range(p.shape[1]):
        rows = valid[:, column]
        if rows.any():
            q[rows, column] = scipy_stats.false_discovery_control(p[rows, column], method='bh')
    selected = valid & (q <= alpha)
    table = pd.DataFrame({'reported': valid.sum(axis=0), 'positive': (selected & (r > 0)).sum(axis=0),
                          'negative': (selected & (r < 0)).sum(axis=0)}, index=effect.columns)
    table['significant'] = table.positive + table.negative
    return table


def feature_summary(data):
    """Finite descriptive summaries with sample variance (ddof=1)."""
    clean = _dataframe(data)
    n = clean.count()
    result = pd.DataFrame(dict(n=n, mean=clean.mean(), std=clean.std(ddof=1),
                               variance=clean.var(ddof=1), min=clean.min(), max=clean.max(),
                               range=clean.max()-clean.min(),
                               status=np.where(n >= 2, 'ok', 'insufficient_n')))
    result.index = result.index.rename('feature')
    return result.loc[:, SUMMARY_COLUMNS]


def regimen_response_edges(values, condition, metadata, *, set_id):
    """One complete source condition to finite measured assay responses.

    This is published Δp, without pooling, constituent attribution, target
    inference, or reconstruction of unresolved sample baselines. Raw condition
    context is repeated unchanged on each edge for inspectable provenance.
    """
    if not isinstance(values, pd.Series):
        raise TypeError('Regimen responses must be an assay-indexed Series')
    clean = _numeric(values)
    if (not isinstance(condition, pd.Series) or not condition.index.is_unique
        or not {'UID','set','cell_line_name'}.issubset(condition.index)
        or not isinstance(condition.UID, str) or not condition.UID
        or values.name != condition.UID):
        raise ValueError('Response values and source condition UID must agree')
    if not isinstance(set_id,str) or condition['set'] != set_id or not condition.UID.startswith(set_id+'_'):
        raise ValueError('Source condition set differs from declared set')
    if (not isinstance(metadata,pd.DataFrame) or not metadata.index.is_unique
        or not clean.index.isin(metadata.index).all()):
        raise ValueError('Assay metadata must cover every response identity')
    clean = clean.loc[clean.notna()]
    table = pd.DataFrame({'regimen_id':'CPPA-'+set_id+':'+condition.UID,
        'assay_id':clean.index, 'published_delta_p':clean.to_numpy(),
        'response_sign':np.where(clean.gt(0),'positive',np.where(clean.lt(0),'negative','zero'))})
    for field,value in condition.items():
        if field in table:
            raise ValueError('Condition field collides with response edge contract')
        table[field] = [value]*len(table)
    for field in ['raw_name','display_label','gene_symbols','candidate_gene_symbols',
                  'annotation_status','target_annotation_status','modification','residue_site']:
        column = 'assay_label' if field == 'display_label' else field
        if column in table:
            raise ValueError('Condition field collides with assay metadata contract')
        table[column] = metadata.loc[clean.index,field].to_numpy() if field in metadata else None
    return table


def correlate(x, y, method='pearson', min_n=MIN_N, inference=True):
    """Correlate finite pairs aligned by identity; Pearson Fisher CI is 95%."""
    _method(method)
    _minimum(min_n)
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        raise TypeError('x and y must be pandas Series')
    x, y = _numeric(x), _numeric(y)
    shared = x.index[x.index.isin(y.index)]
    x, y = x.loc[shared], y.reindex(shared)
    finite = x.notna() & y.notna()
    x, y = x.loc[finite], y.loc[finite]
    n = len(x)
    result = dict(r=np.nan, p_value=np.nan, n=n, status='insufficient_n',
                  ci_low=np.nan, ci_high=np.nan,
                  ci_method='unavailable' if inference else 'descriptive',
                  x=x, y=y, method=method)
    if n < min_n:
        return result
    if x.nunique() < 2 or y.nunique() < 2:
        result['status'] = 'constant'
        return result
    test = (scipy_stats.pearsonr if method == 'pearson' else scipy_stats.spearmanr)(x, y)
    if not np.isfinite(test.statistic) or (inference and not np.isfinite(test.pvalue)):
        raise ValueError('Correlation calculation produced nonfinite results for finite, varying inputs')
    result.update(r=float(test.statistic), status='ok')
    if not inference:
        return result
    result['p_value'] = float(test.pvalue)
    if method == 'pearson' and n > 3:
        r = result['r']
        if abs(r) == 1:
            low = high = r
        else:
            delta = scipy_stats.norm.ppf(.975) / np.sqrt(n - 3)
            low, high = np.tanh(np.arctanh(r) + np.array([-delta, delta]))
        result.update(ci_low=float(low), ci_high=float(high), ci_method='fisher_z_95')
    return result


def correlate_one_vs_all(data, target, method='pearson', min_n=MIN_N, inference=True):
    """Target-excluded correlation family with BH over all eligible features."""
    clean = _dataframe(data)
    _method(method)
    _minimum(min_n)
    if target not in clean.columns:
        raise ValueError('Target feature is absent')
    # Statistical pairs need values and identities, not copied display/provenance
    # dictionaries on every Series operation. The caller metadata is untouched.
    clean.attrs = {}
    features = clean.columns[clean.columns != target]
    rows = [correlate(clean[target], clean[feature], method, min_n, inference) for feature in features]
    result = _table(rows, features, CORRELATION_COLUMNS)
    return _adjust(result)


def interaction_network(data, metadata, source, *, human_population, reference=None,
                        search='', method='pearson'):
    """Assay-preserving projection of a validated gene-level causal scaffold.

    Search/reference restrict the displayed relations only. Every source record
    remains separate; multiple targets generate every supported assay pair. The
    correlation is descriptive on exactly the passed population, never a causal
    effect, an interaction score, or a corrected hypothesis-test family.
    """
    import json
    clean = _dataframe(data); clean.attrs = {}
    _method(method)
    if not isinstance(human_population, bool):
        raise TypeError('human_population must be an explicit bool')
    if (not isinstance(metadata, pd.DataFrame) or not metadata.index.is_unique
        or not clean.columns.isin(metadata.index).all()
        or not {'gene_symbols','annotation_status'}.issubset(metadata.columns)):
        raise ValueError('Interaction metadata must cover each assay identity')
    required = ['SIGNOR_ID','ENTITYA','ENTITYB']
    if not set(required).issubset(source.columns) or not source.SIGNOR_ID.is_unique:
        raise ValueError('Interaction source record identities are invalid')
    nodes = metadata.reindex(clean.columns).copy()
    nodes.index = nodes.index.rename('assay_id')
    source_genes = set(source.ENTITYA)|set(source.ENTITYB)
    targets = {}; statuses = []; matches = []; target_counts = []
    for assay, row in nodes.iterrows():
        genes = json.loads(row.gene_symbols)
        if (not isinstance(genes, list) or any(not isinstance(g, str) or not g for g in genes)
            or len(genes) != len(set(genes))):
            raise ValueError('Assay gene_symbols must be a unique JSON string list')
        mapped = [g for g in genes if g in source_genes]
        status = ('unsupported_population' if not human_population else
                  str(row.annotation_status) if row.annotation_status != 'verified' else
                  'verified_unmapped' if not mapped else
                  'verified_multiple_targets' if len(genes) > 1 else 'verified_single_target')
        statuses.append(status); matches.append(json.dumps(mapped)); target_counts.append(len(genes))
        if human_population and row.annotation_status == 'verified':
            for gene in mapped: targets.setdefault(gene, []).append(assay)
    nodes['mapping_status'] = statuses
    nodes['source_gene_matches'] = matches
    nodes['target_count'] = target_counts
    seed = None
    if search.strip():
        query = search.strip().casefold()
        seed = {assay for assay, row in nodes.iterrows()
                if query in ' '.join([str(assay), *[str(row.get(field,'')) for field in
                    ['raw_name','display_label','gene_symbols','modification','residue_site']]]).casefold()}
    rows = []; correlations = {}
    for record in source.to_dict('records'):
        for a in targets.get(record['ENTITYA'], []):
            for b in targets.get(record['ENTITYB'], []):
                if a == b or (reference is not None and reference not in {a,b}): continue
                if seed is not None and not ({a,b} & seed): continue
                key = tuple(sorted((a,b)))
                if key not in correlations:
                    result = correlate(clean[a], clean[b], method=method, inference=False)
                    correlations[key] = dict(r=result['r'], finite_n=result['n'],
                        correlation_status=result['status'],
                        paired_ids=json.dumps(result['x'].index.tolist()), correlation_method=method)
                rows.append(dict(record, assay_a=a, assay_b=b, **correlations[key]))
    columns = [*source.columns, 'assay_a','assay_b','r','finite_n',
               'correlation_status','paired_ids','correlation_method']
    edges = pd.DataFrame(rows, columns=columns)
    if not edges.empty:
        edges = edges.sort_values(['assay_a','assay_b','SIGNOR_ID'],kind='stable').reset_index(drop=True)
    return nodes, edges


def two_group_test(data, groups, group_a, group_b, min_n=MIN_N):
    """Two-sided Welch tests; effect is mean_a minus mean_b on input scale."""
    clean = _dataframe(data)
    _minimum(min_n)
    if not isinstance(groups, pd.Series):
        raise TypeError('groups must be a pandas Series')
    _identity(groups.index)
    if groups.isna().any() or len(groups) != len(clean) or not clean.index.isin(groups.index).all():
        raise ValueError('Groups must provide one nonmissing label per data identity')
    if group_a == group_b or group_a not in groups.values or group_b not in groups.values:
        raise ValueError('Two distinct present group labels are required')
    groups = groups.reindex(clean.index)
    a = clean.loc[groups.eq(group_a)]
    b = clean.loc[groups.eq(group_b)]
    n_a, n_b = a.count(), b.count()
    mean_a, mean_b = a.mean(), b.mean()
    std_a, std_b = a.std(ddof=1), b.std(ddof=1)
    enough = (n_a >= min_n) & (n_b >= min_n)
    constant = a.nunique().lt(2) & b.nunique().lt(2)
    status = pd.Series('insufficient_n', index=clean.columns)
    status.loc[enough & constant] = 'constant'
    eligible = enough & ~constant
    result = pd.DataFrame(dict(n_a=n_a, n_b=n_b, mean_a=mean_a, mean_b=mean_b,
                               mean_diff=mean_a-mean_b, statistic=np.nan,
                               p_value=np.nan, q_value=np.nan, status=status))
    if eligible.any():
        # Vectorized sample moments preserve per-feature finite denominators.
        test = scipy_stats.ttest_ind_from_stats(
            mean_a[eligible].to_numpy(), std_a[eligible].to_numpy(), n_a[eligible].to_numpy(),
            mean_b[eligible].to_numpy(), std_b[eligible].to_numpy(), n_b[eligible].to_numpy(),
            equal_var=False)
        finite = np.isfinite(test.statistic) & np.isfinite(test.pvalue)
        if not finite.all():
            raise ValueError('Welch calculation produced nonfinite results for finite, varying inputs')
        result.loc[eligible, 'statistic'] = test.statistic
        result.loc[eligible, 'p_value'] = test.pvalue
        result.loc[eligible, 'status'] = 'ok'
    result.index = result.index.rename('feature')
    return _adjust(result.loc[:, GROUP_COLUMNS])


def zscore(data, axis=0, ddof=1):
    """Standardize finite values per feature (axis 0) or per row (axis 1)."""
    if isinstance(axis, bool) or not isinstance(axis, numbers.Integral) or axis not in (0, 1):
        raise ValueError('axis must be 0 or 1')
    if isinstance(ddof, bool) or not isinstance(ddof, numbers.Integral) or ddof < 0:
        raise ValueError('ddof must be a nonnegative integer')
    clean = _dataframe(data)
    mean = clean.mean(axis=axis)
    std = clean.std(axis=axis, ddof=ddof).replace(0, np.nan)
    return clean.sub(mean, axis=1-axis).div(std, axis=1-axis)


def cluster_heatmap(data, mode='source', cut=3):
    """Local average-linkage Euclidean ordering on complete displayed entities.

    Each axis clusters only entities finite across its entire opposite axis.
    Incomplete entities stay at the end in input order with a missing cluster.
    Values are neither transformed nor imputed here. ``cut`` is the maximum
    clusters requested from scipy fcluster(maxclust); ties may yield fewer.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    clean = _dataframe(data)
    if mode not in ('source', 'rows', 'columns', 'both'):
        raise ValueError('mode must be source, rows, columns or both')
    if isinstance(cut, bool) or not isinstance(cut, numbers.Integral) or cut < 1:
        raise ValueError('cut must be a positive integer')
    def axis_order(frame, enabled):
        clusters = pd.Series(pd.NA, index=frame.index, dtype='Int64')
        if not enabled or frame.shape[1] == 0:
            return frame.index, clusters
        complete = frame.notna().all(axis=1)
        keys = frame.index[complete]
        if len(keys) < 2:
            if len(keys): clusters.loc[keys] = 1
            return keys.append(frame.index[~complete]), clusters
        tree = linkage(frame.loc[keys].to_numpy(), method='average', metric='euclidean')
        clusters.loc[keys] = fcluster(tree, t=cut, criterion='maxclust')
        return keys[leaves_list(tree)].append(frame.index[~complete]), clusters
    rows, row_clusters = axis_order(clean, mode in ('rows','both'))
    columns, column_clusters = axis_order(clean.T, mode in ('columns','both'))
    return clean.loc[rows, columns], row_clusters, column_clusters


def population_summary(data, groups):
    """Selected observation counts and finite assay denominators by exact ID."""
    clean = _dataframe(data)
    if not isinstance(groups, pd.Series):
        raise TypeError('groups must be a pandas Series')
    _identity(groups.index)
    if set(groups.index) != set(clean.index):
        raise ValueError('Summary groups must match selected observation IDs')
    aligned = groups.reindex(clean.index).astype(object).where(lambda x: x.notna(), 'Unreported').astype(str)
    rows = []
    for group in aligned.unique():
        part = clean.loc[aligned.eq(group)]
        rows.append({'Group':group, 'Observations':len(part),
                     'Assays with finite measurements':int(part.notna().any().sum()),
                     'Finite measurements':int(part.notna().sum().sum())})
    counts = pd.DataFrame(rows, columns=['Group','Observations','Assays with finite measurements','Finite measurements'])
    finite = clean.count().rename('Finite n').rename_axis('Assay ID').reset_index()
    return counts, finite
