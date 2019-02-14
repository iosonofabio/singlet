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


def test_kmeans_samples(ds):
    ds2 = ds.copy()
    ph = 'quantitative_phenotype_1_[A.U.]'
    ds2.samplesheet[ph] = ds2.samplesheet[ph].fillna(0)
    d = ds2.cluster.kmeans(
            n_clusters=2,
            axis='samples',
            phenotypes=(ph,))
    assert(d.tolist() == [0, 0, 1, 0])


def test_kmeans_features(ds):
    ds2 = ds.query_features_by_name(['TSPAN6', 'GAPDH', 'ACTB', 'ACTG1'])
    ph = 'quantitative_phenotype_1_[A.U.]'
    ds2.samplesheet[ph] = ds2.samplesheet[ph].fillna(0)
    d = ds2.cluster.kmeans(
            n_clusters=2,
            axis='features',
            phenotypes=(ph,))
    assert(d.tolist() == [0, 1, 1, 0, 0])


# FIXME: improve this test
def test_dbscan_samples(ds):
    ds2 = ds.query_features_by_name(ds.featurenames[:100])
    ph = 'quantitative_phenotype_1_[A.U.]'
    ds2.samplesheet[ph] = ds2.samplesheet[ph].fillna(0)
    ds2.counts.log(inplace=True)
    d = ds2.cluster.dbscan(
            axis='samples',
            phenotypes=(ph,))
    assert(d.tolist() == [-1, -1, -1, -1])


def test_dbscan_features(ds):
    ds2 = ds.query_features_by_name(ds.featurenames[:100])
    ph = 'quantitative_phenotype_1_[A.U.]'
    ds2.samplesheet[ph] = ds2.samplesheet[ph].fillna(0)
    d = ds2.cluster.dbscan(
            axis='features',
            phenotypes=(ph,))
    assert(d.tolist()[:6] == [-1, 0, -1, -1, -1, 0])


def test_hierarchical_samples(ds):
    d = ds.cluster.hierarchical(
            'samples',
            optimal_ordering=False,
            log_features=True)
    assert(tuple(d['leaves']) == ('second_sample', 'third_sample',
                                  'test_pipeline', 'first_sample'))


def test_hierarchical_features(ds):
    from scipy.spatial.distance import pdist, squareform

    ds2 = ds.copy()
    ds2.counts = ds2.counts.iloc[:200]
    Y = squareform(pdist(ds2.counts.values, metric='euclidean'))
    d = ds2.cluster.hierarchical(
            'features',
            metric=Y,
            optimal_ordering=False,
            log_features=True)
    assert(tuple(d['leaves'])[:3] == ('PSMB1', 'SLC25A5', 'KDM1A'))


def test_hierarchical_features_phenotypes(ds):
    print('Hierarchical clustering of features and phenotypes')
    ds2 = ds.copy()
    ds2.counts = ds2.counts.iloc[:200]
    d = ds2.cluster.hierarchical(
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
    labels = ds.cluster.affinity_propagation(
            axis='samples',
            metric=distance)
    assert(labels.iloc[[0, 3]].tolist() == [1, 1])
    print('Done!')


def test_affinitypropagation_log_features(ds):
    print('Affinity propagation (precomputed)')
    from scipy.spatial.distance import pdist, squareform
    ds2 = ds.query_features_by_name(['TSPAN6', 'GAPDH', 'ACTB', 'ACTG1'])
    distance = squareform(pdist(ds2.counts.values))
    labels = ds2.cluster.affinity_propagation(
            axis='features',
            metric=distance,
            log_features=True)
    assert(labels.tolist() == [1, 0, 0, 1])
    print('Done!')
