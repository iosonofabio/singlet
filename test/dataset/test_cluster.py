#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import numpy as np
import scipy as sp
import pytest


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')


def test_hierarchical_samples(ds):
    print('Hierarchical clustering of samples')
    d = ds.cluster.hierarchical(
            'samples',
            optimal_ordering=False,
            log_features=True)
    assert(tuple(d['leaves']) == ('second_sample', 'third_sample',
                                  'test_pipeline', 'first_sample'))
    print('Done!')


def test_hierarchical_features(ds):
    print('Hierarchical clustering of features')
    ds.counts = ds.counts.iloc[:200]
    d = ds.cluster.hierarchical(
            'features',
            optimal_ordering=False,
            log_features=True)
    assert(tuple(d['leaves'])[:3] == ('PNPLA4', 'RHBDF1', 'ITGAL'))
    print('Done!')


def test_hierarchical_features_phenotypes(ds):
    print('Hierarchical clustering of features and phenotypes')
    ds.counts = ds.counts.iloc[:200]
    d = ds.cluster.hierarchical(
            axis='features',
            phenotypes=('quantitative_phenotype_1_[A.U.]',),
            optimal_ordering=True,
            log_features=True)
    #FIXME: ordering seems to be slightly nondeterministic?
    assert('quantitative_phenotype_1_[A.U.]' in d['leaves'])
    print('Done!')


def test_affinitypropagation(ds):
    print('Affinity propagation (precomputed)')
    from scipy.spatial.distance import pdist, squareform
    distance = squareform(pdist(ds.counts.values.T))
    labels = ds.cluster.affinity_propagation(axis='samples', metric=distance)
    assert(labels.tolist() == [0, 0, 0, 1])
    print('Done!')


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    ds = ds()
    test_hierarchical_samples(ds)
    test_hierarchical_features(ds)
    test_hierarchical_features_phenotypes(ds)
    test_affinitypropagation(ds)
