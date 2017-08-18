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
            n_dims=2,
            transform='log10',
            robust=False)
    assert(tuple(d['lambdas'].astype(int)) == (5, 3))
    print('Done!')

    print('Test Dataset robust PCA')
    ds.counts = ds.counts.iloc[:200]
    d = ds.dimensionality.pca(
            n_dims=2,
            transform='log10',
            robust=True)
    assert(tuple(d['lambdas'].astype(int)) == (11, 1))
    print('Done!')
