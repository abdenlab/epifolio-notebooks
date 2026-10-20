"""Read a source dataset from either a standalone or consolidated AnnData store."""
import json
import anndata as ad
import numpy as np
import pandas as pd
import zarr

LAYOUT = 'collection-anndata-v1'
# Published portal result tables; read on request, never with the measurements.
ANALYSES = 'portal_analyses'


def named_layer_names(data):
    """Return stored modality names, excluding AnnData 0.13's implicit X key."""
    return tuple(name for name in data.layers if name is not None)


def _standalone(path, group):
    if ANALYSES not in group['uns']:
        return ad.read_zarr(path)
    parts = {name: ad.io.read_elem(group[name])
             for name in ('X', 'obs', 'var', 'layers', 'obsm', 'varm', 'obsp', 'varp') if name in group}
    parts['uns'] = {name: ad.io.read_elem(group['uns'][name]) for name in group['uns'] if name != ANALYSES}
    return ad.AnnData(**parts)


def analysis_group(path, *names):
    """Open one published analysis group; arrays stay on disk until sliced."""
    node = zarr.open_group(str(path), mode='r')['uns']
    for name in (ANALYSES, *names):
        if name not in node:
            raise ValueError(f'Published analysis absent from {path}: {"/".join(names) or ANALYSES}')
        node = node[name]
    return node


def read_matrix(path, dataset_id=None, *, obs_ids=None, var_ids=None):
    group = zarr.open_group(str(path), mode='r')
    uns = group['uns']
    if 'storage_layout' not in uns or ad.io.read_elem(uns['storage_layout']) != LAYOUT:
        return _subset(_standalone(path, group), obs_ids, var_ids)
    if dataset_id is None or dataset_id not in uns['datasets']:
        raise ValueError(f'Dataset absent from consolidated store: {dataset_id}')
    record = uns['datasets'][dataset_id]
    bounds = ad.io.read_elem(record['bounds'])
    r0,r1,c0,c1 = map(int,bounds)
    if not (0 <= r0 < r1 <= group['X'].shape[0] and 0 <= c0 < c1 <= group['X'].shape[1]):
        raise ValueError('Consolidated slice bounds invalid')
    obs, var = ad.io.read_elem(record['obs']), ad.io.read_elem(record['var'])
    if len(obs)!=r1-r0 or len(var)!=c1-c0:
        raise ValueError('Consolidated slice dimensions disagree')
    # Check the collection axes as well as the retained source metadata.
    obs_index = ad.io.read_elem(group['obs'][group['obs'].attrs['_index']])[r0:r1]
    var_index = ad.io.read_elem(group['var'][group['var'].attrs['_index']])[c0:c1]
    if list(obs.index)!=list(obs_index) or list(var.index)!=list(var_index):
        raise ValueError('Consolidated slice identities disagree')
    if not obs.dataset_id.eq(dataset_id).all() or not var.dataset_id.eq(dataset_id).all():
        raise ValueError('Consolidated slice ownership disagrees')
    result = ad.AnnData(np.asarray(group['X'][r0:r1,c0:c1]),obs=obs,var=var,
                        uns=ad.io.read_elem(record['uns']))
    for name in group['layers']:
        result.layers[name] = np.asarray(group['layers'][name][r0:r1,c0:c1])
    return _subset(result, obs_ids, var_ids)


def _subset(data, obs_ids, var_ids):
    if obs_ids is None and var_ids is None:
        return data
    rows = data.obs_names if obs_ids is None else pd.Index(obs_ids)
    columns = data.var_names if var_ids is None else pd.Index(var_ids)
    if (not rows.is_unique or not columns.is_unique or not rows.isin(data.obs_names).all()
            or not columns.isin(data.var_names).all()):
        raise ValueError("Requested measurement identities must be unique and present")
    return data[rows, columns].copy()


def read_table(path, dataset_id=None, *, consolidated=False, columns=None):
    if isinstance(columns, str): columns = json.loads(columns)
    else: columns = None
    return pd.read_parquet(path, filters=[('dataset_id','==',dataset_id)] if consolidated else None,
                           columns=columns)

