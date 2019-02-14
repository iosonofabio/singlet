#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')


def test_features_phenotypes(ds):
    r = ds.correlation.correlate_features_phenotypes(
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            features=['TSPAN6', 'DPM1'])
    assert(np.isclose(r.values[0, 0], -0.8, rtol=1e-1, atol=1e-1))


def test_features_phenotype(ds):
    r = ds.correlation.correlate_features_phenotypes(
            phenotypes='quantitative_phenotype_1_[A.U.]',
            features=['TSPAN6', 'DPM1'])
    assert(np.isclose(r.values[0], -0.8, rtol=1e-1, atol=1e-1))


def test_feature_phenotypes(ds):
    r = ds.correlation.correlate_features_phenotypes(
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            features='TSPAN6')
    assert(np.isclose(r.values[0], -0.8, rtol=1e-1, atol=1e-1))


def test_feature_phenotype(ds):
    r = ds.correlation.correlate_features_phenotypes(
            phenotypes='quantitative_phenotype_1_[A.U.]',
            features='TSPAN6')
    assert(np.isclose(r, -0.8, rtol=1e-1, atol=1e-1))


def test_features_phenotypes_pearson(ds):
    r = ds.correlation.correlate_features_phenotypes(
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            features=['TSPAN6', 'DPM1'],
            method='pearson',
            fillna=0)
    assert(np.isclose(r.values[1, 0], -0.6, rtol=1e-1, atol=1e-1))


def test_features_phenotypes_fillna(ds):
    r = ds.correlation.correlate_features_phenotypes(
            phenotypes='quantitative_phenotype_1_[A.U.]',
            features=['TSPAN6', 'DPM1'],
            method='pearson',
            fillna={'quantitative_phenotype_1_[A.U.]': 0})
    assert(np.isclose(r.values[1], -0.6, rtol=1e-1, atol=1e-1))


def test_features_phenotypes_pearson_all(ds):
    ds2 = ds.query_features_by_name(['TSPAN6', 'DPM1'])
    r = ds2.correlation.correlate_features_phenotypes(
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            features='all',
            method='pearson',
            fillna=0)
    assert(np.isclose(r.values[1, 0], -0.6, rtol=1e-1, atol=1e-1))


def test_features_phenotypes_pearson_fillna(ds):
    r = ds.correlation.correlate_phenotypes_phenotypes(
            phenotypes='quantitative_phenotype_1_[A.U.]',
            phenotypes2='quantitative_phenotype_1_[A.U.]',
            method='pearson',
            fillna={'quantitative_phenotype_1_[A.U.]': 0},
            fillna2={'quantitative_phenotype_1_[A.U.]': 0},
            )
    assert(np.isclose(r, 1, rtol=1e-1, atol=1e-1))


def test_features_features(ds):
    r = ds.correlation.correlate_features_features(
            features=['TSPAN6', 'DPM1'],
            features2=['TSPAN6'],
            method='pearson')
    assert(np.isclose(r.values[0, 0], 1, rtol=1e-1, atol=1e-1))


def test_features_features_all(ds):
    ds2 = ds.query_features_by_name(['TSPAN6', 'DPM1'])
    r = ds2.correlation.correlate_features_features(
            features='all',
            features2='TSPAN6',
            method='pearson')
    assert(np.isclose(r.values[0], 1, rtol=1e-1, atol=1e-1))


def test_samples(ds):
    n = ds.n_samples
    r = ds.correlation.correlate_samples()
    assert(np.allclose(r.values[np.arange(n), np.arange(n)], 1))


def test_samples_2(ds):
    n = ds.n_samples
    sns = ds.samplenames
    r = ds.correlation.correlate_samples(
            samples=sns,
            samples2=sns,
            )
    assert(np.allclose(r.values[np.arange(n), np.arange(n)], 1))


def test_samples_3(ds):
    n = ds.n_samples
    sns = ds.samplenames
    r = ds.correlation.correlate_samples(
            samples=sns,
            samples2='all',
            )
    assert(np.allclose(r.values[np.arange(n), np.arange(n)], 1))


def test_samples_withpheno(ds):
    sns = ds.samplenames
    print(ds.samplesheet.columns)
    r = ds.correlation.correlate_samples(
            samples=sns[0],
            samples2=sns[0],
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            )
    assert(np.isclose(r, 1))
