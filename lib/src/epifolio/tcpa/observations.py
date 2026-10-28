"""Source observation metadata and shared cohort selection, without inferred samples."""
from copy import deepcopy
import numpy as np
import pandas as pd

FIELDS = {
    'cancer_group': ('Cancer / tissue', ('Tumor',)),
    'cancer_subtype': ('Subtype (source)', ('cancer_subtype',)),
    'cell_line': ('Cell line', ('Cell line', 'cell_line')),
    'species': ('Species', ('species', 'Species')),
    'drug': ('Drug', ('Drug',)),
    'concentration': ('Concentration (source units)', ('Conc',)),
    'time': ('Time (source units)', ('Time',)),
    'replicate': ('Replicate (source label)', ('Replicate',)),
    'sample_type': ('Sample type', ('sample_type', 'Sample Type')),
    'tcga_sample_type': ('TCGA sample type', ('tcga_sample_type',)),
    'region': ('Region', ('region',)),
    'tumour_id': ('Tumor / participant ID', ('tumour_id', 'participant_id')),
    'batch': ('Batch', ('batch',)),
}
UNREPORTED = {'', 'na', 'nan', 'none', '<na>', 'unknown', 'unreported'}
# Explicit lexical synonyms only. Broader disease categories remain separate.
CANCER_LABELS = {'ovary':'Ovarian', 'ovarian':'Ovarian', 'cervix':'Cervical',
                 'cervical':'Cervical', 'oesophagus':'Oesophagus'}


def _text(value):
    if pd.isna(value) or str(value).strip().casefold() in UNREPORTED:
        return 'Unreported'
    return str(value).strip()


def observation_metadata(obs):
    """Add a presentation view; the original columns and identifiers are untouched."""
    result = pd.DataFrame(index=obs.index)
    for key, (_, sources) in FIELDS.items():
        source = next((name for name in sources if name in obs), None)
        if source is not None:
            result[key] = obs[source].astype(object).map(
                (lambda value: 'Unreported' if pd.isna(value) else str(value))
                if key == 'cancer_subtype' else _text)
    if 'cancer_group' in result:
        result['cancer_group'] = result.cancer_group.map(lambda x: CANCER_LABELS.get(x.casefold(), x))
    return result


def observation_options(metadata):
    return {key: FIELDS[key][0] for key in metadata
            if metadata[key].ne('Unreported').any()}


def select_observations(expr, obs, filters, *, group_by='cancer_group'):
    """Intersect source-value filters by stable row ID and refresh selection metadata."""
    if not expr.index.is_unique or not obs.index.is_unique or set(expr.index) != set(obs.index):
        raise ValueError('Expression/observation identities disagree')
    obs = obs.loc[expr.index]
    metadata = observation_metadata(obs)
    if group_by not in observation_options(metadata):
        raise ValueError('Unknown observation grouping: '+group_by)
    mask = pd.Series(True, index=expr.index)
    for field, values in filters.items():
        if field not in observation_options(metadata):
            raise ValueError('Unknown observation filter: '+field)
        if not set(values) <= set(metadata[field]):
            raise ValueError('Unknown source value for observation filter: '+field)
        if values:
            mask &= metadata[field].isin(values)
    rows = obs.loc[mask].copy()
    selected = expr.loc[mask].copy()
    selected.attrs = deepcopy(expr.attrs)
    selected = selected.loc[:,selected.notna().any(axis=0)].copy()
    attrs = selected.attrs
    panel = attrs['panel_metadata'].copy()
    panel['finite_observations'] = selected.notna().sum().reindex(panel.index, fill_value=0)
    panel['measured_in_selection'] = panel.finite_observations.gt(0)
    attrs['panel_metadata'] = panel
    attrs['feature_metadata'] = panel.loc[selected.columns].copy()
    attrs['feature_labels'] = {key:expr.attrs['feature_labels'][key] for key in selected}
    attrs['coverage'] = {'total_features':len(panel),'measured_features':len(selected.columns),
                         'unmeasured_features':len(panel)-len(selected.columns)}
    attrs['selected_rows'] = len(rows)
    attrs['source_rows'] = len(expr)
    attrs['observation_filters'] = deepcopy(filters)
    attrs['observation_grouping'] = group_by
    attrs['annotation_coverage'] = panel.annotation_status.value_counts().to_dict()
    for column, attribute in [('biological_identity_status','biological_identity_status'),
                              ('species','species_counts')]:
        if column in rows:
            attrs[attribute] = rows[column].astype(str).value_counts().to_dict()
    sample_field = ('tcga_sample_type' if 'tcga_sample_type' in metadata and metadata.tcga_sample_type.ne('Unreported').any()
                    else 'sample_type')
    if sample_field in rows:
        attrs['sample_type_counts'] = rows[sample_field].astype(str).value_counts().to_dict()
    # A new grouping variable does not acquire the original tumor-comparison contract.
    if group_by != 'cancer_group':
        attrs['dataset_contract']['analysis_policy'] = 'descriptive'
    return selected, rows, metadata.loc[mask].copy()


# CPPA rows are response conditions, not cohort observations. Slots describe one
# complete regimen; filtering a slot never splits or attributes that regimen.
CPPA_FIELDS = {
    'condition_uid': 'Condition ID', 'cell_line_name': 'Cell line',
    **{f'compound_name_{slot}': f'Compound {slot}' for slot in range(1, 5)},
    **{f'dosage_{slot}': f'Dose {slot} (source units)' for slot in range(1, 5)},
    'time': 'Exposure time (source units)', 'resource': 'Resource',
    '2D_3D': 'Culture', 'stimuli': 'Stimulus',
}


def cppa_condition_metadata(treatment):
    """Present declared design values on stable condition UIDs; preserve raw rows."""
    if 'UID' not in treatment or treatment.UID.isna().any() or treatment.UID.duplicated().any():
        raise ValueError('CPPA condition identities must be unique and nonmissing')
    rows = treatment.set_index('UID', drop=False)
    result = pd.DataFrame(index=rows.index)
    for field in CPPA_FIELDS:
        source = 'UID' if field == 'condition_uid' else field
        if source in rows:
            result[field] = rows[source].astype(object).map(_text)
    return result


def cppa_condition_options(metadata):
    return {field: CPPA_FIELDS[field] for field in metadata
            if metadata[field].ne('Unreported').any()}


def select_cppa_conditions(response, treatment, filters):
    """One UID-aligned intersection for published Δp and complete treatment rows."""
    metadata = cppa_condition_metadata(treatment)
    if not response.index.is_unique or set(response.index) != set(metadata.index):
        raise ValueError('CPPA response/treatment identities disagree')
    metadata = metadata.loc[response.index]
    fields = cppa_condition_options(metadata)
    mask = pd.Series(True, index=response.index)
    for field, values in filters.items():
        if field not in fields or not set(values) <= set(metadata[field]):
            raise ValueError('Unknown CPPA condition filter or source value: '+field)
        if values:
            mask &= metadata[field].isin(values)
    rows = treatment.set_index('UID', drop=False).loc[response.index[mask]].reset_index(drop=True)
    selected = response.loc[mask].copy()
    selected.attrs = deepcopy(response.attrs)
    finite = np.isfinite(selected)
    selected = selected.loc[:,finite.any(axis=0)].copy()
    attrs = selected.attrs
    panel = attrs['panel_metadata'].copy()
    panel['finite_observations'] = finite.sum().reindex(panel.index, fill_value=0)
    panel['measured_in_selection'] = panel.finite_observations.gt(0)
    attrs['panel_metadata'] = panel
    attrs['feature_metadata'] = panel.loc[selected.columns].copy()
    attrs['feature_labels'] = {key:response.attrs['feature_labels'][key] for key in selected}
    attrs['condition_labels'] = {uid:response.attrs['condition_labels'][uid] for uid in selected.index}
    attrs['coverage'] = {'total_features':len(panel),'measured_features':len(selected.columns),
                         'unmeasured_features':len(panel)-len(selected.columns)}
    attrs['selected_rows'], attrs['source_rows'] = len(rows), len(response)
    attrs['selected_condition_uids'] = selected.index.tolist()
    attrs['condition_filters'] = deepcopy(filters)
    if 'reconstruction_status' in rows:
        attrs['reconstruction_coverage'] = {status:int(rows.reconstruction_status.eq(status).sum()) for status in ['verified','unresolved']}
    return selected, rows, metadata.loc[mask].copy()


CPPA_ATLAS_FACETS = {'experiment': 'Study', 'lab': 'Lab', 'cell_line': 'Cell line',
                     'tissue': 'Tissue (CCLE match)', 'compound': 'Compound', 'dose': 'Compound and dose',
                     'regimen': 'Given alone or in combination', 'time': 'Exposure time',
                     'culture': 'Culture', 'stimulus': 'Stimulus'}
NOT_MAPPED = 'Not yet mapped'


def _name_key(value):
    return ''.join(ch for ch in str(value).upper() if ch.isalnum())


def cppa_tissue_labels(cell_lines, ccle):
    """Exact normalized-name matches to CCLE cell lines; ambiguous or absent names stay unmapped."""
    tissues = {}
    for line, tissue in zip(ccle.cell_line.astype(str), ccle.tissue.astype(str)):
        tissues.setdefault(_name_key(line), set()).add(tissue)
    def label(name):
        found = tissues.get(_name_key(name), set()) if _name_key(name) else set()
        return next(iter(found)).replace('_', ' ').capitalize() if len(found) == 1 else NOT_MAPPED
    return {name: label(name) for name in cell_lines}


def cppa_atlas_conditions(summary, tissues):
    """One design row per condition UID across experiments; published values are not altered."""
    rows = summary.set_index('UID')
    slots = [slot for slot in range(1, 5) if f'compound_name_{slot}' in rows]
    def regimen(row):
        parts = [(_text(row[f'compound_name_{slot}']), _text(row.get(f'dosage_{slot}'))) for slot in slots]
        return [(name, dose) for name, dose in parts if name != 'Unreported']
    regimens = rows.apply(regimen, axis=1)
    doses = regimens.map(lambda parts: tuple(f'{name} (dose unreported)' if dose == 'Unreported' else f'{name} {dose}'
                                             for name, dose in parts))
    atlas = pd.DataFrame({
        'experiment': rows.dataset_id.astype(str), 'lab': rows.resource.map(_text),
        'cell_line': rows.cell_line_name.map(_text),
        'tissue': rows.cell_line_name.map(lambda name: tissues.get(name, NOT_MAPPED)),
        'compounds': regimens.map(lambda parts: tuple(name for name, _ in parts)),
        'regimen': regimens.map(lambda parts: 'Combination' if len(parts) > 1 else 'Single compound'),
        'doses': doses, 'regimen_label': doses.map(' + '.join),
        'time': rows.time.map(_text), 'culture': rows['2D_3D'].map(_text), 'stimulus': rows.stimuli.map(_text),
        'treated_n': rows.treated_n, 'control_n': rows.control_n,
        'response_check': rows.reconstruction_status.map(_text) if 'reconstruction_status' in rows else 'Unreported',
    }, index=rows.index)
    atlas.index.name = 'UID'
    return atlas


def cppa_atlas_values(atlas, field):
    if field in _TUPLES:
        return sorted({name for names in atlas[_TUPLES[field]] for name in names})
    return sorted(atlas[field].unique())


# Compound and dose facets match any regimen slot; their atlas columns hold one tuple per condition.
_TUPLES = {'compound': 'compounds', 'dose': 'doses'}


def cppa_atlas_mask(atlas, facets, *, exclude=None):
    """Values within a facet are alternatives; facets intersect. A compound matches any regimen slot."""
    mask = pd.Series(True, index=atlas.index)
    for field, values in facets.items():
        if field not in CPPA_ATLAS_FACETS:
            raise ValueError('Unknown CPPA atlas facet: ' + field)
        if field == exclude or not values:
            continue
        chosen = set(values)
        mask &= (atlas[_TUPLES[field]].map(lambda names: bool(chosen & set(names))) if field in _TUPLES
                 else atlas[field].isin(chosen))
    return mask


def _examples(values, limit=4):
    values = sorted(set(values))
    return ', '.join(values[:limit]) + (f' and {len(values) - limit} more' if len(values) > limit else '')


def _count(n, word, plural=None):
    return f"{n} {word if n == 1 else plural or word + 's'}"


def cppa_study_labels(atlas):
    """Readable study names: the source set number and the lab that ran it."""
    labs = atlas.groupby('experiment').lab.agg(lambda v: ', '.join(sorted(set(v))))
    return {study: f"{study.removeprefix('CPPA-')} · {lab}" for study, lab in labs.items()}


def cppa_entry_options(atlas, field, published=()):
    """Starting-point choices, each labelled with what the atlas holds for it."""
    rows = atlas.explode('compounds') if field == 'compound' else atlas
    groups = rows.groupby('compounds' if field == 'compound' else field)
    counts = pd.DataFrame({'conditions': groups.size(), 'lines': groups.cell_line.nunique(),
                           'studies': groups.experiment.nunique()})
    if field == 'cell_line':
        counts['tissue'] = groups.tissue.first()
        counts['compounds'] = groups.compounds.agg(lambda v: len({name for names in v for name in names}))
    def label(name, row):
        if field == 'compound':
            return (f"{name} — {_count(row.conditions, 'condition')} · {_count(row.lines, 'cell line')} · "
                    f"{_count(row.studies, 'study', 'studies')}" + (' · published analysis' if name in published else ''))
        return (f"{name} ({row.tissue}) — {_count(row.conditions, 'condition')} · {_count(row.compounds, 'compound')} · "
                f"{_count(row.studies, 'study', 'studies')}")
    return {label(name, row): name for name, row in counts.sort_index().iterrows()}


def cppa_profile(atlas, study_labels, compounds=()):
    """What the atlas records for a set of conditions, as source values; nothing is inferred."""
    if atlas.empty:
        return pd.DataFrame(columns=['Recorded', 'In these conditions'])
    def counted(values, limit=None):
        counts = pd.Series(list(values), dtype=object).value_counts()
        shown = ', '.join(f'{value} ({n})' for value, n in counts.head(limit).items())
        return (shown + (f' and {len(counts) - limit} more' if limit and len(counts) > limit else '')) or 'None'
    # Slot-aligned pairs keep each dose with its own compound.
    pairs = [(name, dose) for names, doses in zip(atlas.compounds, atlas.doses) for name, dose in zip(names, doses)]
    rows = {
        'Conditions': f"{len(atlas):,} ({int(atlas.regimen.eq('Combination').sum())} in combination)",
        'Studies': counted(atlas.experiment.map(study_labels)),
        'Cell lines': f"{atlas.cell_line.nunique()}: {_examples(atlas.cell_line, 12)}",
        'Tissue (CCLE name match)': counted(atlas.drop_duplicates('cell_line').tissue),
    }
    for compound in compounds:
        rows[f'{compound} doses as reported'] = counted(dose.removeprefix(compound).strip().replace('(dose unreported)', 'Unreported')
                                                         for name, dose in pairs if name == compound)
        rows[f'{compound} given with'] = counted(other for names in atlas.compounds if compound in names
                                                 for other in names if other != compound)
    if not compounds:
        rows['Compounds'] = f"{len({name for name, _ in pairs})}: " + counted((name for name, _ in pairs), 12)
    rows.update({
        'Exposure times as reported': counted(atlas.time),
        'Culture': counted(atlas.culture),
        'Stimulus': counted(atlas.stimulus),
        'Response check': counted(atlas.response_check.replace({'verified': 'Reproduced', 'unresolved': 'Unresolved'})),
    })
    return pd.DataFrame({'Recorded': list(rows), 'In these conditions': list(rows.values())})


def cppa_study_choices(atlas, study_labels, compounds=()):
    """Studies in a selection, largest first; studies holding every anchored compound lead."""
    groups = atlas.groupby('experiment')
    found = pd.DataFrame({'n': groups.size(), 'lines': groups.cell_line.agg(lambda v: _examples(v, 3)),
                          'all': groups.compounds.agg(lambda v: set(compounds) <= {name for names in v for name in names})})
    found = found.sort_values(['all', 'n'], ascending=False, kind='stable')
    return {f"{study_labels[study]} — {_count(row.n, 'condition')} · {row.lines}": study for study, row in found.iterrows()}
