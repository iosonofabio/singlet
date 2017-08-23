#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
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
    ds.counts = ds.counts.iloc[:20]
    d = ds.dimensionality.pca(
            n_dims=4,
            transform='log10',
            robust=False,
            random_state=0)
    print(d)
    print(d['eigenvalues'])
    assert(tuple(d['eigenvalues'].astype(int)) == (5, 3))
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
    assert(tuple(d['eigenvalues'].astype(int)) == (11, 1))
    print('Done!')

    print('Test Dataset t-SNE')
    ds.counts = ds.counts.iloc[:200]
    vs = ds.dimensionality.tsne(
            n_dims=2,
            transform='log10',
            theta=0.5,
            perplexity=0.8)
    assert(tuple(vs.values[0].astype(int)) == (512, 42))
    print('Done!')

    print('Test cache for t-SNE')
    ds.dimensionality._tsne_cache['func_kwargs']['n_dims'] = 'none'
    vs = ds.dimensionality.tsne(
            n_dims='none',
            transform='log10',
            theta=0.5,
            perplexity=0.8)
    print('Done!')
