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
