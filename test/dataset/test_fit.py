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
    return Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')


def test_linear_fit_phenotypes(ds):
    res = ds.fit.fit_single(
            xs=['TSPAN6', 'DPM1'],
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='linear')
    assert(np.allclose(res[0, 0], [-0.005082, 3.548869, 0.599427],
                       rtol=1e-03, atol=1e-03))


def test_linear_fit_phenotypes_spikeins(ds):
    res = ds.fit.fit_single(
            xs='spikeins',
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='linear')
    assert(np.allclose(res[0, 0], [-1.890637e-03,  2.581346,  2.655694],
                       rtol=1e-03, atol=1e-03))


def test_linear_fit_phenotypes_other(ds):
    res = ds.fit.fit_single(
            xs='other',
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='linear')
    assert(np.allclose(res[-1, 0], [-1.127875e-06,  3.664309,  9.685498e-01],
                       rtol=1e-03, atol=1e-03))


def test_linear_fit_phenotypes_mapped(ds):
    ds2 = ds.query_features_by_name(['TSPAN6', 'DPM1'])
    res = ds2.fit.fit_single(
            xs='mapped',
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='linear')
    assert(np.allclose(res[0, 0], [-0.005082, 3.548869, 0.599427],
                       rtol=1e-03, atol=1e-03))


def test_linear_fit_phenotypes_total(ds):
    ds2 = ds.query_features_by_name(['TSPAN6', 'DPM1'])
    res = ds2.fit.fit_single(
            xs='total',
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='linear')
    assert(np.allclose(res[0, 0], [-0.005082, 3.548869, 0.599427],
                       rtol=1e-03, atol=1e-03))


@pytest.mark.xfail(True, reason='Nonlinear fit: assert non implemented yet')
def test_nonlinear_fit_phenotypes(ds):
    print('Test nonlinear fit of phenotypes')
    res = ds.fit.fit_single(
            xs=['TSPAN6', 'DPM1'],
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='threshold-linear')
    # TODO: assert result!
    assert(0 == 1)
    print('Done!')


def test_logistic_fit_phenotypes(ds):
    ds2 = ds.copy()
    ds2.samplesheet['name2'] = ['s1', 's2', 's3', 's4']
    ds2.reindex(axis='samples', column='name2', drop=True, inplace=True)
    ds2 = ds + ds2
    res = ds2.fit.fit_single(
            xs=['TSPAN6', 'DPM1'],
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='logistic')
    assert(np.allclose(res[0, 0], [1, 2.16666667, 1, 1, 5.77333333],
                       rtol=1e-03, atol=1e-03))

