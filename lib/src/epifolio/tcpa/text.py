"""Shared presentation strings; no calculation, file access or UI imports."""
from math import isfinite


def phenotype_input_gates(surface):
    """Declared unavailable capabilities; these rows never activate analyses."""
    gates = {
        'patient': [
            ('Reactive patient survival',
             'Per-participant endpoint time/event data, endpoint definition, time units and censoring semantics.',
             'Validate release and exact participant joins, uniqueness, coverage and missingness. Specify the selected population, cutoff and survival method.'),
            ('Patient ecDNA',
             'Sourced ecDNA calls, release and sample-to-participant crosswalk.',
             'Validate sample identity, call definition and coverage; distinguish negative calls from missing or unassessed samples.'),
        ],
        'cell_line': [
            ('Cell-line ecDNA',
             'Sourced ecDNA calls, release and model crosswalk.',
             'Validate model identity, call definition and coverage; distinguish negative calls from missing or unassessed models.'),
            ('Cell-line mutation',
             'Mutation data, release, model/gene joins and assay coverage.',
             'Define altered, wild-type and missing separately; wild-type requires adequate coverage. Validate joins and the comparison method.'),
            ('Cell-line dependency',
             'Dependency scores, release and model/gene joins.',
             'Validate score meaning, scale, coverage, duplicate mappings and missingness; specify the association method.'),
            ('Recalculated cell-line drug association',
             'Per-cell-line sensitivity outcomes with endpoint definition, release and cell/drug identity mapping. The portal publishes correlation results only.',
             'Validate outcome units, design compatibility, coverage and missingness; specify the association method. Published correlations cannot be re-derived for another selection.'),
        ],
        'cppa': [
            ('Recalculated CPPA predictive markers',
             'Per-cell-line sensitivity endpoint and an explicit P0, P1 or published Δp predictor definition. The portal publishes correlation results only.',
             'Validate release and compatible cell/drug/design joins, units, coverage and missingness; specify the predictive method. Response correlation alone is insufficient.'),
            ('Recalculated CPPA sensitive/resistant',
             'Source-defined or explicitly specified sensitivity classes, their outcome source and classification rule. The portal publishes group summaries only.',
             'Validate cell/drug/design joins, class coverage and missingness; specify the comparison method. Two-condition subtraction does not establish sensitivity classes.'),
        ],
    }
    if surface not in gates:
        raise ValueError(f'Unknown phenotype surface: {surface}')
    return [{'Module': module, 'Status': 'Unavailable — input contract required',
             'Required input': required, 'Activation checks': checks}
            for module, required, checks in gates[surface]]


def phenotype_gate_description(surface):
    """Short scope note beside the declared input gates."""
    if surface == 'patient':
        return ('These analyses are unavailable for the selected cohort. The 64 cached ACC1/MTOR documents across 32 cancers contain fixed published survival curves, not per-participant endpoint records. They cannot recompute survival after patient filtering, arbitrary assay selection or cutoff changes. A published-result viewer would require separate validation of result semantics.')
    if surface == 'cell_line':
        return ('These analyses are unavailable. RPPA expression and lineage labels supply neither ecDNA calls, mutation coverage, dependency scores nor per-cell-line drug-sensitivity outcomes. The Drug Sensitivity tab shows the portal\'s published protein–drug correlations unchanged; recalculating them for a local selection needs the sensitivity outcomes. Each module needs its own sourced input and validated identity contract.')
    if surface == 'cppa':
        return ('These analyses are unavailable. Published Δp, individual sample measurements and treatment context describe protein responses. They do not provide per-cell-line sensitivity outcomes. The Published Drug Analyses tab shows the portal\'s own predictive-marker and sensitive/resistant results unchanged; recalculating them for a local selection needs those outcomes. Response correlations and two-condition comparisons retain their descriptive meaning.')
    raise ValueError(f'Unknown phenotype surface: {surface}')


def fenced_text(lines):
    """Preserve the notebooks' plain fenced summary formatting."""
    return "```\n" + "\n".join(lines) + "\n```"


def label_options(ids, labels, *, sort=False):
    """Display unique readable labels while widget values retain stable identities.

    ``sort`` orders the menu by label (case-insensitive, identity as tie-break);
    it never reorders the data the identities select.
    """
    ids = list(ids)
    options = {labels.get(key, str(key)): key for key in ids}
    if len(options) != len(ids):
        raise ValueError("Display labels must be unique; identities cannot be merged")
    if sort:
        options = dict(sorted(options.items(), key=lambda item: (item[0].casefold(), str(item[1]))))
    return options


def assay_table(frame, labels, columns, *, rank=False):
    """Display copy of an assay-indexed result: readable label first, exact ID retained.

    ``columns`` maps result column to heading, in display order. Row order,
    values, dtypes and missingness are unchanged; the input is not modified.
    """
    reserved = {'Protein / assay', 'Assay ID'} | ({'Rank'} if rank else set())
    headings = list(columns.values())
    if len(set(headings)) != len(headings) or reserved & set(headings):
        raise ValueError("Display headings must be unique and distinct from the identity columns")
    display = frame.loc[:, list(columns)].rename(columns=columns)
    ids = list(frame.index)
    display.insert(0, 'Assay ID', ids)
    display.insert(0, 'Protein / assay', [labels.get(key, str(key)) for key in ids])
    if rank:
        display.insert(0, 'Rank', range(1, len(display) + 1))
    return display.reset_index(drop=True)


def display_labels(ids, labels):
    """Resolve labels by explicit stable key, preserving selection order."""
    return [labels.get(key, str(key)) for key in ids]


def analysis_status(value):
    """Translate result codes only at the presentation boundary."""
    return {'ok': 'Available', 'insufficient_n': 'Too few measurements',
            'constant': 'No variation', 'verified': 'Checked',
            'unresolved': 'Unconfirmed', 'conflicting': 'Conflicting annotations',
            'descriptive': 'Descriptive only'}.get(str(value), str(value).replace('_', ' '))


def _reported(value):
    return isinstance(value, str) and value.strip().lower() not in {'', 'na', 'nan', 'unknown', 'unreported', 'none', '<na>'}


def dataset_metadata(row):
    """Read both legacy catalog fields and versioned source metadata."""
    import json
    source = row.get('source_metadata', {})
    if isinstance(source, str):
        try:
            source = json.loads(source)
        except (ValueError, TypeError):
            source = {}
    if not isinstance(source, dict):
        source = {}
    fields = {'description': ('Dataset_Description',), 'tissue': ('Cancer_Type',),
              'specimen_type': ('Sample_Type', 'specimen_type'), 'species': ('Species',),
              'normalization': ('Normalization_Description', 'Normalization_Method', 'normalization'),
              'reference': ('Reference',), 'pmid': ('PMID', 'publication'), 'comments': ('Comments',)}
    return {name: next((value.strip() for key in keys for value in (row.get(key), source.get(key))
                        if _reported(value)), 'Unreported') for name, keys in fields.items()}


def dataset_name(row):
    """Source-backed display name; the accession remains the selection key."""
    key = row.get('dataset_id', 'Unreported')
    cohort = row.get('cohort_name')
    metadata = dataset_metadata(row)
    if _reported(cohort) and cohort != key:
        name = cohort
    else:
        name = next((metadata[field] for field in ('description', 'tissue')
                     if _reported(metadata[field])), key)
    specimen = metadata['specimen_type']
    return f'{name} · {specimen}' if _reported(specimen) else name


def dataset_options(catalog):
    """Readable availability labels; option values retain source identities."""
    names = {'validated': 'Available', 'blocked': 'Unavailable', 'not_built': 'Not loaded'}
    return {f"{dataset_name(row)} [{row['dataset_id']}; {names.get(row['artifact_status'], 'Unavailable')}]": row['dataset_id']
            for row in catalog.to_dict('records')}


def dataset_description(frame):
    """Public source and coverage notes, without build terminology."""
    c = frame.attrs['dataset_contract']
    coverage = frame.attrs['coverage']
    annotation = frame.attrs['annotation_coverage']
    lines = ['### About these data', f"**Selected dataset: {dataset_name(c)}**"]
    if frame.attrs.get('biological_identity_status', {}).get('unresolved_source_record_only', 0):
        lines.append('Rows are source records; some may describe the same specimen. Counts do not represent independent specimens.')
    if frame.attrs.get('sample_type_counts'):
        types = {'01': 'primary', '06': 'metastatic', '02': 'recurrent'}
        lines.append('Samples: ' + ', '.join(f"{n:,} {types.get(code, code)}" for code, n in frame.attrs['sample_type_counts'].items()) + '.')
    if frame.attrs.get('species_counts'):
        lines.append('Species in this selection: ' + ', '.join(f'{n:,} {name}' for name,n in frame.attrs['species_counts'].items()) + '.')
    lines.append(f"- Source ID: {frame.attrs['dataset_id']}; release {c.get('release', 'unreported')}.")
    metadata = dataset_metadata(c)
    for field, label in [('description', 'Study description'), ('tissue', 'Tissue / cancer type'),
                         ('specimen_type', 'Sample type'), ('species', 'Species'),
                         ('reference', 'Reference'), ('pmid', 'Publication ID'), ('comments', 'Source notes')]:
        if _reported(metadata[field]):
            lines.append(f'- {label}: {metadata[field]}.')
    lines.append(f"- Measurements: {c.get('normalization', 'normalization not reported')}. Units: {c.get('units', 'not reported')}.")
    unit = "response conditions" if c.get("resource_family") == "CPPA" else "source observations"
    lines.append(f"- Coverage: {len(frame):,} {unit}; {len(frame.columns):,} measured assays of {coverage['total_features']:,}. {coverage['unmeasured_features']:,} assays have no finite values in this selection and remain in the assay guide.")
    lines.append(f"- Missing values: {int(frame.isna().sum().sum()):,} among the measured assays. Values are not filled in; each comparison uses its available measurements.")
    if isinstance(c.get('source_missing_values'), (int, float)) and isfinite(c['source_missing_values']):
        lines.append(f"- Missing values across the full source panel: {int(c['source_missing_values']):,}.")
    lines.append(f"- Panel annotations: {annotation.get('verified', 0):,} source matched; {annotation.get('unresolved', 0):,} unresolved; {annotation.get('conflicting', 0):,} conflicting.")
    panel = frame.attrs.get('panel_metadata')
    if panel is not None and 'target_annotation_status' in panel:
        targets = panel.target_annotation_status.eq('source_supported')
        partial = targets & panel.annotation_status.ne('verified')
        candidates = panel.candidate_gene_symbols.ne('[]') if 'candidate_gene_symbols' in panel else targets & False
        lines.append(f'- Target annotations: {int(targets.sum()):,} source supported, including {int(partial.sum()):,} with an unresolved reagent. {int(candidates.sum()):,} assays have unverified cross-panel candidates for review.')
    lines.append('Panel matching is distinct from experimental validation of an antibody. Target-only annotations and candidate matches do not establish reagent equivalence. Assays and source values remain separate.')
    return '\n'.join(lines)


def available_groupings(metadata):
    """Only offer populated design fields; retain missing members explicitly."""
    return [column for column in ['compound_name_1', 'cell_line_name', 'time', 'dosage_1']
            if column in metadata and metadata[column].fillna('').astype(str).str.strip().replace({'NA':'', 'unknown':''}).ne('').any()] + ['None']


def source_group_label(value):
    text = '' if value is None else str(value).strip()
    return 'Unreported' if text in {'', 'NA', 'unknown', 'nan', '<NA>'} else text


def cppa_reconstruction_description(frame):
    """Explain source-response checks without hiding uncertainty."""
    counts = frame.attrs['reconstruction_coverage']
    if 'not_recorded' in counts:
        return 'Published TCPA responses are shown. A per-condition treatment/control check is not included in this data collection.'
    return (
        f"**All {len(frame):,} published response conditions are shown.** "
        f"Treatment/control checks: {counts['verified']:,} reproduced; {counts['unresolved']:,} unresolved. "
        'Unresolved responses stay visible and are marked. These checks use the available sample information and an inferred rounding model; published values are unchanged.'
    )


def reconstruction_reason(value):
    """Human-readable explanations; the exact diagnostic remains in the data."""
    value = str(value)
    if not value: return ''
    if 'Ambiguous' in value:
        return 'More than one control group matches; the original group is unconfirmed.'
    if 'membership' in value or 'no controls' in value:
        return 'No matching control group could be established from the published sample information.'
    if 'rounding tolerance' in value:
        return 'The published response differs from the available treatment/control calculation beyond the inferred rounding bounds.'
    if 'missingness' in value:
        return 'The response and individual samples have different missing measurements.'
    if 'precision' in value:
        return 'The available precision is insufficient to check this response.'
    return value


def condition_display(frame):
    names = {'UID': 'Condition ID', 'set': 'Experiment', 'display_label': 'Condition',
             'cell_line_name': 'Cell line', 'compound_name_1': 'Primary compound',
             'compound_name_2': 'Compound 2', 'compound_name_3': 'Compound 3', 'compound_name_4': 'Compound 4',
             'dosage_1': 'Primary dose', 'dosage_2': 'Dose 2', 'dosage_3': 'Dose 3', 'dosage_4': 'Dose 4',
             'time': 'Exposure time', '2D_3D': 'Culture', 'stimuli': 'Stimulus', 'resource': 'Source',
             'treated_n': 'Treated samples', 'control_n': 'Candidate controls',
             'reconstruction_status': 'Response check', 'reconstruction_reason': 'Check notes'}
    frame = frame.copy()
    names['dataset_id'] = 'Dataset ID'
    display = frame[[c for c in names if c in frame]].copy()
    if 'control_n' in display: display['control_n'] = display.control_n.astype(object).replace(-1, 'Unreported')
    if 'reconstruction_status' in display: display['reconstruction_status'] = display.reconstruction_status.astype(object).replace({'verified':'Reproduced', 'unresolved':'Unresolved'})
    if 'reconstruction_reason' in display: display['reconstruction_reason'] = display.reconstruction_reason.map(reconstruction_reason)
    return display.rename(columns=names)


def assay_display(frame):
    import json
    names = {'display_label':'Assay', 'raw_name':'Published name', 'gene_symbols':'Gene targets',
             'protein_name':'Protein', 'modification':'Modification', 'residue_site':'Site',
             'annotation_status':'Panel annotation', 'target_annotation_status':'Target annotation',
             'target_annotation_evidence':'Target evidence', 'candidate_gene_symbols':'Candidate gene targets (unverified)',
             'annotation_resolution_reason':'Annotation notes', 'annotation_locator':'Source location',
             'reagent_origin':'Reagent origin', 'reagent_supplier':'Reagent supplier',
             'reagent_catalog_number':'Reagent catalog number', 'reagent_rrid':'Reagent RRID',
             'antibody_id_source':'Source reagent ID', 'validation_status':'Source validation status',
             'measured_in_selection':'Finite in selection',
             'finite_observations':'Finite measurements', 'feature_id':'Assay ID',
             'annotation_source':'Annotation source'}
    if 'gene_symbols' not in frame and 'gene' in frame:
        names['gene'] = 'Gene targets'
    frame = frame.copy()
    names['dataset_id'] = 'Dataset ID'
    display = frame[[c for c in names if c in frame]].copy()
    if 'annotation_status' in display:
        display['annotation_status'] = display.annotation_status.astype(object).replace({'verified':'Source matched','unresolved':'Unresolved','conflicting':'Source conflict'})
    if 'target_annotation_status' in display:
        display['target_annotation_status'] = display.target_annotation_status.astype(object).replace({
            'source_supported':'Source supported','unresolved':'Unresolved','unreported_in_source':'Unreported in source'})
    if 'gene_symbols' in display:
        def genes(value):
            if isinstance(value, (list, tuple)):
                return ', '.join(str(item) for item in value) or 'Unreported'
            if isinstance(value, str) and value.startswith('['):
                values = json.loads(value)
                return ', '.join(values) or 'Unreported'
            return value if isinstance(value, str) and value.strip() else 'Unreported'
        display['gene_symbols'] = display.gene_symbols.map(genes)
        if 'candidate_gene_symbols' in display:
            display['candidate_gene_symbols'] = display.candidate_gene_symbols.map(genes)
    return display.rename(columns=names)


def catalog_display(frame):
    """Public dataset guide; source values and identifiers stay unchanged."""
    names = {'cohort_name':'Dataset', 'dataset_id':'Source ID', 'resource_family':'Collection',
             'release':'Release', 'specimen_type':'Sample type', 'artifact_status':'Availability',
             'source_rows':'Published observations', 'artifact_rows':'Available observations',
             'assays_total':'Assays', 'assays_measured':'Measured assays', 'assays_unmeasured':'Unmeasured assays',
             'annotation_verified':'Source matched annotations', 'annotation_unresolved':'Unresolved annotations',
             'annotation_conflicting':'Conflicting annotations',
             'reconstruction_verified_conditions':'Responses reproduced',
             'reconstruction_unresolved_conditions':'Responses unresolved', 'block_reason':'Availability notes'}
    frame = frame.copy()
    rows = frame.to_dict('records')
    frame['cohort_name'] = [dataset_name(row) for row in rows]
    metadata = [dataset_metadata(row) for row in rows]
    extra = {'description': 'Study description', 'tissue': 'Tissue / cancer type',
             'specimen_type': 'Sample type', 'species': 'Species', 'normalization': 'Normalization',
             'reference': 'Reference', 'pmid': 'Publication ID', 'comments': 'Source notes'}
    for field in extra:
        frame[field] = [row[field] for row in metadata]
    names.update(extra)
    display = frame[[c for c in names if c in frame]].copy()
    if 'artifact_status' in display:
        display['artifact_status'] = display.artifact_status.replace({'validated':'Available','blocked':'Unavailable','not_built':'Not loaded'})
    if 'block_reason' in display: display['block_reason'] = display.block_reason.map(reconstruction_reason)
    for name in display:
        values = display[name]
        if values.dtype.kind in 'iuf' and len(values) and values.notna().all() and values.eq(0).all():
            display[name] = '0'
    return display.rename(columns=names)
