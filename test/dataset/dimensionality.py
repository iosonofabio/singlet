#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import platform
import numpy as np


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')

    # FIXME: these test assertions need to be checked!
    print('Test Dataset PCA')
    ds.counts = ds.counts.iloc[:200]
    d = ds.dimensionality.pca(
            n_dims=2,
            transform='log10',
            robust=False,
            random_state=0)
    assert(np.allclose(d['vs'].values[0], [-2.677194, -5.129792]))
    print('Done!')

    print('Test cache for PCA')
    ds.dimensionality._pca_cache['func_kwargs']['n_dims'] = 'none'
    d = ds.dimensionality.pca(
            n_dims='none',
            transform='log10',
            robust=False,
            random_state=0)
    print('Done!')

    print('Test Dataset robust PCA')
    ds.counts = ds.counts.iloc[:200]
    d = ds.dimensionality.pca(
            n_dims=2,
            transform='log10',
            robust=True)
    assert(np.allclose(d['vs'].values[0, :1], [-4.38351382]))
    print('Done!')

    print('Test Dataset t-SNE')
    ds.counts = ds.counts.iloc[:200]
    ds.counts.log(inplace=True)
    vs = ds.dimensionality.tsne(
            n_dims=2,
            theta=0.5,
            perplexity=0.8)
    assert(np.allclose(vs.values[0], [-19.164444, 1229.9626]))
    print('Done!')

    print('Test cache for t-SNE')
    ds.dimensionality._tsne_cache['func_kwargs']['n_dims'] = 'none'
    vs = ds.dimensionality.tsne(
            n_dims='none',
            theta=0.5,
            perplexity=0.8)
    print('Done!')

    print('Test Dataset UMAP')
    ds.counts = ds.counts.iloc[:200]
    vs = ds.dimensionality.umap(
            n_dims=2,
            n_neighbors=3)
    assert(np.allclose(vs.values[0], [12.637338, -6.560592]))
    print('Done!')

    print('Test cache for UMAP')
    ds.dimensionality._umap_cache['func_kwargs']['n_dims'] = 'none'
    vs = ds.dimensionality.umap(
            n_dims='none',
            n_neighbors=3)
    print('Done!')
