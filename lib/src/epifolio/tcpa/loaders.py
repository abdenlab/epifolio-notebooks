"""Read validated Phase 2 artifacts with explicit identities and notebook scope.

No inference is performed here. Generic analytical functions live in stats.
"""
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from .paths import resolve_data_dir
from .storage import read_matrix, read_table, analysis_group, LAYOUT, named_layer_names

RPPA500_TCGA_DATASET = 'RPPA500-TCGA'
SCHEMA_VERSION = '3.0.0'
SIGNOR_SNAPSHOT_SHA256 = '5be15571d1282ff791fa2b9c05bae9dd7cc4cedbba7d41ee1aa1236d752f6ae6'


def interaction_human_population(dataset_id, metadata):
    """Authorize human scaffold use from source-declared population context."""
    if metadata.empty:
        return False, 'No selected observations'
    values = (metadata['species'].fillna('').astype(str).str.strip().str.casefold()
              if 'species' in metadata else pd.Series('',index=metadata.index))
    humans = {'human','homo sapiens'}
    missing = {'','unknown','unreported','na','none','nan','not specified'}
    # These source collections are explicitly human; contributed experiments
    # otherwise require populated species on every selected observation.
    declared_human = dataset_id in {'CCLE','RPPA500-TCGA','CONTRIB-TCPA00000002'}
    if values.isin(humans).all() or (declared_human and values.isin(humans|missing).all()):
        return True, 'Human population declared by the source'
    return False, 'Human-only scaffold: select observations with source-declared human species'


def load_interaction_reference(*, anchor=None):
    """Read only the pinned SIGNOR reference; never download during analysis."""
    import hashlib
    import json
    root = resolve_data_dir(anchor=anchor,
        required=('interaction_source.json', 'interaction_edges.parquet'))
    manifest = json.loads((root/'interaction_source.json').read_text())
    source = manifest['source']; artifact = manifest['artifact']
    declared = dict(TYPEA='protein', TYPEB='protein', DIRECT='t', TAX_ID='9606',
                    DATABASEA='UNIPROT', DATABASEB='UNIPROT')
    if (manifest['schema_version'] != '1.0.0' or manifest['provider'] != 'SIGNOR'
        or source['sha256'] != SIGNOR_SNAPSHOT_SHA256 or source['license'] != 'CC-BY-4.0'
        or manifest['slice'] != declared or artifact['path'] != 'interaction_edges.parquet'):
        raise ValueError('Interaction reference differs from pinned source contract')
    path = root/artifact['path']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Interaction artifact escapes data root')
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact['sha256']:
        raise ValueError('Interaction artifact hash mismatch')
    records = pd.read_parquet(path)
    if (list(records.columns) != manifest['columns'] or len(records) != artifact['records']
        or len(records) != manifest['coverage']['selected_records']
        or not records.SIGNOR_ID.is_unique or records.SIGNOR_ID.eq('').any()
        or not all(records[field].eq(value).all() for field, value in declared.items())
        or records[['ENTITYA','ENTITYB','IDA','IDB','EFFECT','PMID']].isna().any().any()
        or records[['ENTITYA','ENTITYB','IDA','IDB','EFFECT','PMID']].eq('').any().any()):
        raise ValueError('Interaction artifact violates record/provenance contract')
    return records, manifest

TCGA_DATASET = 'CONTRIB-TCPA00000002'
TUMOR_LABELS = {
    'BLCA':'Bladder cancer', 'BRCA':'Breast cancer', 'COAD':'Colon adenocarcinoma',
    'READ':'Rectal adenocarcinoma', 'GBM':'Glioblastoma', 'HNSC':'Head and neck squamous cell carcinoma',
    'KIRC':'Kidney renal clear cell carcinoma', 'LUAD':'Lung adenocarcinoma',
    'LUSC':'Lung squamous cell carcinoma', 'OVCA':'Ovarian cancer', 'OV':'Ovarian cancer',
    'UCEC':'Uterine corpus endometrial carcinoma',
}


def _matrix(path, dataset_id=None):
    print(f'Loading validated artifact from {path}...')
    data = read_matrix(path, dataset_id)
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError(f'Duplicate sample/feature identities in {path.name}')
    if 'feature_id' not in data.var or list(data.var.feature_id.astype(str)) != list(data.var_names):
        raise ValueError('Feature identity column differs from matrix index')
    for frame in (data.obs, data.var):
        if len({str(c).casefold() for c in frame}) != len(frame.columns):
            raise ValueError('Metadata columns collide on case-insensitive filesystems')
    if 'raw_name' not in data.var or 'display_label' not in data.var:
        raise ValueError('Missing source/display feature metadata; rebuild Phase 2 artifacts')
    if np.isinf(np.asarray(data.X)).any():
        raise ValueError('Infinite measurement in artifact')
    return data


def _labels(var):
    display = var['display_label'].astype(object).fillna('').astype(str)
    display = display.where(display.str.strip().ne(''), var['raw_name'].astype(str))
    result = display.to_dict()
    duplicated = display.duplicated(keep=False)
    for key in var.index[duplicated]:
        result[key] = f'{result[key]} [{key}]'
    return result


def load_ccle(*, anchor=None):
    root = resolve_data_dir(anchor=anchor, required=('ccle_rppa.zarr',))
    entry = _entry(root, 'CCLE')
    data = _matrix(_artifact(root, entry, 'matrix_path', 'ccle_rppa.zarr'))
    _ownership(data, 'CCLE', entry)
    if 'sample_id' not in data.obs or list(data.obs.sample_id) != list(data.obs_names):
        raise ValueError('CCLE sample identities differ from matrix index')
    expr, meta = data.to_df(), data.obs.copy()
    # The source identifier defines the split once: everything after it is tissue.
    expected = pd.Series([s.split('_', 1)[1] if '_' in s else 'Unknown' for s in expr.index], index=expr.index)
    if 'tissue' not in meta or not meta['tissue'].astype(str).equals(expected):
        raise ValueError('CCLE tissue metadata disagrees with source sample identifiers')
    _attach(expr, data.var, _entry(root, 'CCLE'), 'CCLE')
    expr.attrs['gene_symbols_encoding'] = data.uns.get('gene_symbols_encoding', 'not recorded')
    expr.attrs['tissue_labels'] = {key: key.replace('_', ' ').capitalize() for key in sorted(meta.tissue.unique())}
    expr.attrs['sample_labels'] = _observation_labels(meta)
    return expr.index.tolist(), expr, meta, expr.columns.tolist(), sorted(meta.tissue.unique()), meta.tissue.tolist()


def _ccle_drug_group(anchor):
    root = resolve_data_dir(anchor=anchor, required=('ccle_rppa.zarr',))
    return analysis_group(_artifact(root, _entry(root, 'CCLE'), 'matrix_path', 'ccle_rppa.zarr'), 'drug_sensitivity')


def load_ccle_drug_sensitivity(*, anchor=None):
    """Catalog of the portal's published protein–drug correlations; values stay on disk."""
    group = _ccle_drug_group(anchor)
    features = ad.io.read_elem(group['features'])
    if not features.index.is_unique or not features.portal_protein.is_unique:
        raise ValueError('Published drug-correlation assay identities are ambiguous')
    datasets = {}
    for name in group['datasets']:
        node = group['datasets'][name]
        drugs, lineages = list(ad.io.read_elem(node['drugs'])), ad.io.read_elem(node['lineages'])
        if len(set(drugs)) != len(drugs) or node['rho'].shape != (len(lineages), len(features), len(drugs)):
            raise ValueError(f'Published drug-correlation axes disagree: {name}')
        datasets[name] = {'drugs': drugs, 'lineages': lineages}
    return {'features': features, 'datasets': datasets, 'contract': json.loads(ad.io.read_elem(group['contract_json']))}


def load_ccle_drug_correlations(drug_dataset, lineage, *, anchor=None):
    """One lineage's assay × drug rho, p-value and n; unpublished pairs are NaN / 0."""
    group = _ccle_drug_group(anchor)
    if drug_dataset not in group['datasets']:
        raise ValueError(f'Unknown published drug dataset: {drug_dataset}')
    node = group['datasets'][drug_dataset]
    lineages = list(ad.io.read_elem(node['lineages']).index)
    if lineage not in lineages:
        raise ValueError(f'Lineage has no published {drug_dataset} results: {lineage}')
    layer = lineages.index(lineage)
    index = ad.io.read_elem(group['features']).index
    columns = pd.Index(ad.io.read_elem(node['drugs']), name='drug')
    return tuple(pd.DataFrame(node[field][layer], index=index, columns=columns) for field in ('rho', 'p_value', 'n'))


def load_cppa_portal_analyses(*, anchor=None):
    """Portal-wide CPPA drug tables; they are pooled across sets by the provider."""
    root = resolve_data_dir(anchor=anchor, required=('manifest.parquet',))
    catalog = pd.read_parquet(root / 'manifest.parquet')
    names = catalog.loc[catalog.resource_family.eq('CPPA') & catalog.artifact_status.eq('validated'), 'matrix_path'].unique()
    if len(names) != 1:
        raise ValueError('CPPA portal analyses require one consolidated response store')
    group = analysis_group(_artifact(root, {'matrix_path': names[0]}, 'matrix_path', 'cppa_rppa.zarr'))
    tables = {name: ad.io.read_elem(group[name]).reset_index(drop=True)
              for name in ('protein_response', 'predictive_markers', 'sensitive_resistant')}
    tables['contract'] = json.loads(ad.io.read_elem(group['contract_json']))
    return tables


def select_contrib(expr, obs, dataset_id=TCGA_DATASET, *, var=None):
    if not expr.index.is_unique or not obs.index.is_unique or set(expr.index) != set(obs.index):
        raise ValueError('CONTRIB expression/metadata identities disagree')
    if 'dataset_id' not in obs or 'Tumor' not in obs:
        raise ValueError('Missing CONTRIB scope metadata')
    selected = obs.loc[obs.dataset_id.eq(dataset_id)].copy()
    if selected.empty:
        raise ValueError(f'Requested CONTRIB dataset is absent: {dataset_id}')
    matrix = expr.loc[selected.index].copy()
    # Ownership precedes measured coverage; foreign-panel values are corruption.
    if var is not None:
        if 'dataset_id' not in var or not var.index.equals(expr.columns):
            raise ValueError('CONTRIB feature ownership metadata disagree')
        owned = var.dataset_id.eq(dataset_id)
        if matrix.loc[:, ~owned].notna().any().any():
            raise ValueError('Foreign-panel measurements in selected CONTRIB dataset')
        matrix = matrix.loc[:, owned]
    matrix = matrix.loc[:, matrix.notna().any(axis=0)]
    return matrix, selected


def _cohort_result(data, dataset_id, entry):
    if 'sample_id' not in data.obs or list(data.obs.sample_id) != list(data.obs_names):
        raise ValueError('CONTRIB source sample keys differ from matrix index')
    obs = data.obs.copy()
    if 'Tumor' not in obs:
        obs['Tumor'] = 'Unknown source group'
    if obs['Tumor'].isna().any() or obs['Tumor'].astype(str).str.strip().eq('').any():
        obs['Tumor'] = obs['Tumor'].astype(object).fillna('Unknown source group').replace('', 'Unknown source group')
    expr, obs = select_contrib(data.to_df(), obs, dataset_id, var=data.var)
    panel = data.var.loc[data.var.dataset_id.eq(dataset_id)]
    _attach(expr, panel, entry, dataset_id)
    expr.attrs['tumor_labels'] = {code: f'{TUMOR_LABELS.get(code, code)} ({code})' for code in obs.Tumor.unique()}
    expr.attrs['sample_labels'] = _observation_labels(obs)
    expr.attrs['scope'] = dataset_id
    expr.attrs['sample_type_counts'] = obs['tcga_sample_type'].value_counts(dropna=False).to_dict() if 'tcga_sample_type' in obs else (obs['sample_type'].value_counts(dropna=False).to_dict() if 'sample_type' in obs else {})
    expr.attrs['biological_identity_status'] = obs['biological_identity_status'].value_counts(dropna=False).to_dict() if 'biological_identity_status' in obs else {'source_declared': len(obs)}
    expr.attrs['gene_symbols_encoding'] = data.uns.get('gene_symbols_encoding', 'not recorded')
    print(f'{dataset_id}: {len(expr)} samples; {len(expr.columns)} measured antibodies')
    return obs.Tumor.unique().tolist(), expr, obs, expr.columns.tolist()


def load_contrib(*, anchor=None, dataset_id=TCGA_DATASET):
    root = resolve_data_dir(anchor=anchor, required=())
    entry = _entry(root, dataset_id)
    data = _matrix(_artifact(root, entry, 'matrix_path', 'contrib_patient_rppa.zarr'), dataset_id)
    _ownership(data, dataset_id, entry)
    return _cohort_result(data, dataset_id, entry)


def load_tcga_rppa500(*, anchor=None):
    root = resolve_data_dir(anchor=anchor, required=())
    entry = _entry(root, RPPA500_TCGA_DATASET)
    if entry is None:
        raise ValueError('RPPA500-TCGA is not registered; PanCan11 is not a fallback')
    data = _matrix(_artifact(root, entry, 'matrix_path', 'tcga_rppa500.zarr'))
    _ownership(data, RPPA500_TCGA_DATASET, entry)
    return _cohort_result(data, RPPA500_TCGA_DATASET, entry)


def delta_wide(long):
    required = {'UID', 'feature_id', 'modality', 'value'}
    if not required <= set(long):
        raise ValueError(f'Missing CPPA columns: {required - set(long)}')
    delta = long.loc[long.modality.eq('delta_p')].copy()
    if delta.empty or delta[['UID','feature_id']].isna().any().any() or delta[['UID','feature_id']].eq('').any().any():
        raise ValueError('Missing CPPA delta identity keys')
    if delta.duplicated(['UID','feature_id']).any():
        raise ValueError('Duplicate CPPA condition/feature keys; averaging is forbidden')
    full = delta.pivot(index='UID', columns='feature_id', values='value')
    measured = full.notna().any(axis=0)
    return full.loc[:, measured], {'total_features':len(full.columns),
        'measured_features':int(measured.sum()), 'unmeasured_features':int((~measured).sum())}


def load_cppa(*, anchor=None, set_id='set120'):
    root = resolve_data_dir(anchor=anchor, required=())
    entry = _entry(root, f'CPPA-{set_id}')
    if entry is None and set_id != 'set120':
        raise ValueError(f'CPPA {set_id} is not registered')
    data = _matrix(_artifact(root, entry, 'matrix_path', 'cppa_rppa.zarr'), f'CPPA-{set_id}')
    _ownership(data, f'CPPA-{set_id}', entry)
    long = _dataset_table(root, entry, 'long_path', 'cppa_long.parquet')
    metadata = _dataset_table(root, entry, 'summary_path', 'cppa_treatment_summary.parquet')
    wide, coverage = delta_wide(long)
    mandatory = ['UID','set','cell_line_name','compound_name_1','resource','treated_n','control_n','display_label']
    if (not set(mandatory) <= set(metadata)
            or metadata[mandatory].isna().any().any()
            or metadata[mandatory].astype(str).apply(lambda col: col.str.strip().isin(['', 'NA', 'unknown'])).any().any()):
        raise ValueError('Missing mandatory CPPA condition/design metadata')
    if metadata.UID.duplicated().any() or set(metadata.UID) != set(wide.index) or set(data.obs_names) != set(wide.index):
        raise ValueError('CPPA matrix/metadata identities disagree')
    if set(named_layer_names(data)) != {'delta_p'}:
        raise ValueError('CPPA condition matrix must contain only the delta_p layer')
    if not np.array_equal(np.asarray(data.layers['delta_p']), np.asarray(data.X), equal_nan=True):
        raise ValueError('CPPA delta_p layer and X disagree')
    if not metadata['set'].eq(set_id).all():
        raise ValueError(f'CPPA condition metadata must belong to {set_id}')
    design = ['time', '2D_3D', 'stimuli', 'dosage_1', 'compound_name_2', 'dosage_2',
              'compound_name_3', 'dosage_3', 'compound_name_4', 'dosage_4', 'modality']
    if not set(design) <= set(metadata) or not set(metadata.columns) <= set(data.obs.columns):
        raise ValueError('Missing CPPA matrix/summary design fields')
    if (not metadata.modality.eq('delta_p').all() or 'UID' not in data.obs
            or list(data.obs.UID) != list(data.obs_names)):
        raise ValueError('CPPA matrix/summary modality or UID identities disagree')
    summary_by_uid = metadata.set_index('UID', drop=False).loc[wide.index]
    matrix_by_uid = data.obs.loc[wide.index, metadata.columns]
    # Compare every persisted summary field by UID, including blanks, counts and
    # raw source fields; categorical versus string storage is immaterial.
    if not summary_by_uid.astype(object).equals(matrix_by_uid.astype(object)):
        raise ValueError('CPPA matrix/summary design values disagree')
    if set(long.loc[long.modality.eq('delta_p'), 'feature_id']) != set(data.var_names):
        raise ValueError('CPPA long/wide feature identities disagree')
    source = data.to_df().loc[wide.index, wide.columns]
    if not np.array_equal(source.to_numpy(), wide.to_numpy(), equal_nan=True):
        raise ValueError('CPPA long/wide values disagree')
    metadata = metadata.set_index('UID', drop=False).loc[wide.index].reset_index(drop=True)
    reconstruction = _cppa_reconstruction(metadata, entry)
    _attach(wide, data.var, entry, f'CPPA-{set_id}')
    wide.attrs['gene_symbols_encoding'] = data.uns.get('gene_symbols_encoding', 'not recorded')
    wide.attrs['condition_labels'] = _observation_labels(metadata.set_index('UID'))
    wide.attrs['reconstruction_coverage'] = reconstruction
    if 'reconstruction_status' in metadata:
        for uid in metadata.loc[metadata.reconstruction_status.eq('unresolved'), 'UID']:
            wide.attrs['condition_labels'][uid] += ' [reconstruction unresolved]'
    wide.attrs['coverage'] = coverage
    print(f'CPPA {set_id}: {len(wide)} conditions; {coverage["measured_features"]} measured / '
          f'{coverage["total_features"]} cataloged antibodies; {coverage["unmeasured_features"]} unmeasured')
    def values(column):
        return [x for x in metadata[column].dropna().unique().tolist() if str(x).strip() and str(x) not in {'NA','unknown'}]
    return values('cell_line_name'), values('compound_name_1'), wide, metadata, values('resource'), values('time'), long, wide.columns.tolist()


def _cppa_reconstruction(metadata, entry):
    """Validate diagnostic disclosure without filtering published responses."""
    fields = {'reconstruction_status', 'reconstruction_reason'}
    if not fields.intersection(metadata):
        if entry and entry.get('source_values_status') == 'verified':
            raise ValueError('Missing CPPA reconstruction disclosure')
        return {'not_recorded': len(metadata)}
    if not fields.issubset(metadata) or metadata[list(fields)].isna().any().any():
        raise ValueError('Missing CPPA reconstruction disclosure')
    status = metadata.reconstruction_status
    reason = metadata.reconstruction_reason.astype(str).str.strip()
    if not status.isin(['verified', 'unresolved']).all() or not reason.ne('').equals(status.eq('unresolved')):
        raise ValueError('Invalid CPPA reconstruction status/reason')
    treated = pd.to_numeric(metadata.treated_n, errors='raise')
    controls = pd.to_numeric(metadata.control_n, errors='raise')
    if (not (treated.ge(1) & treated.mod(1).eq(0)).all()
            or not ((controls.ge(1) | controls.eq(-1)) & controls.mod(1).eq(0)).all()
            or (status.eq('verified') & controls.lt(1)).any()):
        raise ValueError('Invalid CPPA reconstruction replicate counts')
    counts = {key: int(status.eq(key).sum()) for key in ['verified', 'unresolved']}
    if entry and entry.get('source_values_status') == 'verified':
        for key, count in counts.items():
            if entry.get(f'reconstruction_{key}_conditions') != count:
                raise ValueError('CPPA catalog reconstruction counts disagree')
        expected = 'unresolved' if counts['unresolved'] else 'verified'
        if entry.get('reconstruction_status') != expected:
            raise ValueError('CPPA catalog reconstruction status disagrees')
    return counts


def load_manifest(*, anchor=None):
    root = resolve_data_dir(anchor=anchor, required=('manifest.parquet',))
    print(f'Loading source catalog from {root / "manifest.parquet"}...')
    return pd.read_parquet(root / 'manifest.parquet')


def _legacy_catalog(frame):
    """Describe the legacy inventory without claiming unbuilt datasets are ready."""
    rows = []
    for source in frame.to_dict('records'):
        key = source['dataset_id']
        family = 'CPPA' if key.startswith('CPPA-') else 'CCLE' if key == 'CCLE' else 'CONTRIB'
        ready = bool(source.get('measured_in_patient_artifact', False)) or key in {'CCLE', 'CPPA-set120'}
        row = dict(source, resource_family=family, cohort_name='Historical TCGA PanCan11 RBN' if key == TCGA_DATASET else key,
            release=source.get('TCPA_Release', 'unknown'), source_status='available',
            artifact_status='validated' if ready else 'not_built', schema_version='2.0.0',
            analysis_policy='exploratory' if key in {'CCLE', TCGA_DATASET} else 'descriptive',
            block_reason='' if ready else 'Source is inventoried but has no validated local matrix',
            normalization=source.get('Normalization_Method') or 'unknown', units='source units',
            normalization_evidence=source.get('Normalization_Description') or 'not recorded',
            specimen_type=source.get('Sample_Type') or 'unknown', source_rows=source.get('actual_samples'),
            artifact_rows=source.get('expression_samples', 0), assays_total=source.get('actual_antibodies'))
        row['matrix_path'] = 'ccle_rppa.zarr' if key == 'CCLE' else 'cppa_rppa.zarr' if family == 'CPPA' else 'contrib_patient_rppa.zarr' if ready else None
        if family == 'CPPA':
            row.update(long_path='cppa_long.parquet', summary_path='cppa_treatment_summary.parquet')
        rows.append(row)
    rows.append(dict(dataset_id=RPPA500_TCGA_DATASET, resource_family='RPPA500-TCGA', cohort_name='RPPA500 TCGA',
        source_status='missing', artifact_status='not_built', analysis_policy='blocked', schema_version='2.0.0',
        block_reason='RPPA500 TCGA is not present in this legacy artifact release; PanCan11 is separate'))
    return pd.DataFrame(rows)


def load_dataset_catalog(*, anchor=None):
    frame = load_manifest(anchor=anchor)
    if 'artifact_status' not in frame:
        frame = _legacy_catalog(frame)
    required = {'dataset_id', 'resource_family', 'artifact_status', 'analysis_policy', 'block_reason', 'matrix_path'}
    if not required <= set(frame) or frame.dataset_id.isna().any() or frame.dataset_id.duplicated().any():
        raise ValueError('Invalid dataset catalog identities or status contract')
    if not frame.artifact_status.isin(['validated', 'blocked', 'not_built']).all():
        raise ValueError('Unknown dataset artifact status')
    leading = [name for name in ['dataset_id', 'cohort_name', 'artifact_status', 'block_reason', 'resource_family', 'analysis_policy', 'source_rows', 'artifact_rows', 'assays_total', 'assays_measured', 'annotation_verified', 'annotation_unresolved'] if name in frame]
    return frame.loc[:, leading + [name for name in frame if name not in leading]]


def _entry(root, dataset_id):
    # Unit-level legacy adapters may supply a matrix without a catalog.
    if not (root / 'manifest.parquet').exists():
        return None
    frame = pd.read_parquet(root / 'manifest.parquet')
    if 'artifact_status' not in frame:
        frame = _legacy_catalog(frame)
    selected = frame.loc[frame.dataset_id.eq(dataset_id)]
    if len(selected) != 1:
        raise ValueError(f'Dataset catalog must contain exactly one entry: {dataset_id}')
    entry = selected.iloc[0].to_dict()
    if str(entry.get('schema_version', SCHEMA_VERSION)) not in {'2.0.0', SCHEMA_VERSION}:
        raise ValueError('Incompatible dataset catalog schema')
    if entry['artifact_status'] != 'validated':
        raise ValueError(f'{dataset_id}: {entry.get("block_reason") or entry["artifact_status"]}')
    return entry


def _dataset_table(root, entry, field, legacy):
    return read_table(_artifact(root, entry, field, legacy),
                      entry.get('dataset_id') if entry else None,
                      consolidated=bool(entry and entry.get('storage_layout') == LAYOUT),
                      columns=entry.get(field+'_columns') if entry else None)


def _artifact(root, entry, field, legacy):
    name = entry.get(field) if entry else legacy
    if not isinstance(name, str) or not name or Path(name).name != name or name in {'.', '..'}:
        raise ValueError(f'Invalid dataset artifact path: {field}={name!r}')
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Dataset artifact escapes data root')
    return path


def _ownership(data, dataset_id, entry):
    version = data.uns.get('schema_version')
    is_v3 = entry is not None and str(entry.get('schema_version', SCHEMA_VERSION)) == SCHEMA_VERSION
    if version is not None and version not in {'2.0.0', SCHEMA_VERSION}:
        raise ValueError(f'Incompatible artifact schema: {version}')
    if is_v3 or version == SCHEMA_VERSION:
        if version != SCHEMA_VERSION or data.uns.get('dataset_id') != dataset_id:
            raise ValueError('Artifact schema or embedded dataset identity disagrees with selection')
        if 'dataset_id' not in data.var or not data.var.dataset_id.eq(dataset_id).all():
            raise ValueError('Foreign-panel feature ownership in selected dataset')
        if 'dataset_id' in data.obs and not data.obs.dataset_id.eq(dataset_id).all():
            raise ValueError('Foreign dataset observations in selected matrix')


def _attach(expr, var, entry, dataset_id):
    measured = var.loc[expr.columns]
    expr.attrs['feature_labels'] = _labels(measured)
    expr.attrs['feature_metadata'] = measured.copy()
    panel = var.copy()
    panel['measured_in_selection'] = panel.index.isin(expr.columns)
    panel['finite_observations'] = expr.notna().sum().reindex(panel.index, fill_value=0)
    expr.attrs['panel_metadata'] = panel
    expr.attrs['dataset_id'] = dataset_id
    expr.attrs['dataset_contract'] = entry or {'dataset_id': dataset_id, 'analysis_policy': 'descriptive'}
    expr.attrs['coverage'] = {'total_features': len(var), 'measured_features': len(expr.columns),
        'unmeasured_features': len(var) - len(expr.columns)}
    if 'annotation_status' in var:
        expr.attrs['annotation_coverage'] = var.annotation_status.value_counts(dropna=False).to_dict()
    else:
        expr.attrs['annotation_coverage'] = {'unresolved': len(var)}


def _observation_labels(obs):
    display = obs.display_label.astype(object).fillna('').astype(str)
    display = display.where(display.str.strip().ne(''), pd.Series(obs.index.astype(str), index=obs.index))
    duplicated = display.duplicated(keep=False)
    return {key: f'{value} [{key}]' if duplicated.loc[key] else value for key, value in display.items()}


def load_cppa_sample_detail(condition_id, feature_id, *, anchor=None, set_id='set120'):
    """Source sample measurements with persisted membership, never fitted controls.

    Unresolved controls are shown individually as candidates. No candidate mean
    or reconstructed difference is calculated by this display adapter.
    """
    root = resolve_data_dir(anchor=anchor, required=())
    dataset_id = f'CPPA-{set_id}'
    entry = _entry(root, dataset_id)
    membership = _dataset_table(root, entry, 'membership_path', 'cppa_membership.parquet')
    summary = _dataset_table(root, entry, 'summary_path', 'cppa_treatment_summary.parquet')
    if summary.UID.duplicated().any() or condition_id not in set(summary.UID):
        raise ValueError('Condition must belong to the selected experiment')
    _cppa_reconstruction(summary, entry)
    condition = summary.set_index('UID').loc[condition_id].copy()
    links = membership.loc[membership.condition_id.eq(condition_id)].copy()
    if (links.empty or links[['sample_id','role']].isna().any().any()
            or links.sample_id.duplicated().any() or not links.role.isin(['treated','control']).all()):
        raise ValueError('Invalid persisted CPPA sample membership')
    data = read_matrix(_artifact(root, entry, 'samples_path', 'cppa_samples.zarr'), dataset_id,
                       obs_ids=links.sample_id.tolist(), var_ids=[feature_id])
    _ownership(data, dataset_id, entry)
    if (int(links.role.eq('treated').sum()) != int(condition.treated_n)
            or (condition.control_n >= 0 and int(links.role.eq('control').sum()) != int(condition.control_n))):
        raise ValueError('CPPA membership counts disagree with diagnostics')
    if 'p0_p1' not in data.layers or not np.array_equal(data.X,data.layers['p0_p1'],equal_nan=True):
        raise ValueError('CPPA individual sample modality disagrees')
    table = links.set_index('sample_id').join(data.obs, rsuffix='_source')
    table['Assay ID'] = feature_id
    table['Sample measurement'] = data.to_df()[feature_id]
    unresolved = condition.reconstruction_status != 'verified'
    table['Membership'] = table.role.map({'treated':'Treated','control':'Candidate control' if unresolved else 'Control'})
    table['Reconstruction status'] = condition.reconstruction_status
    table['Reconstruction reason'] = condition.reconstruction_reason
    return table.reset_index(), condition
