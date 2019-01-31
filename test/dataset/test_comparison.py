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
def ds_ds2():
    from singlet.dataset import Dataset
    ds = Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')
    ds2 = ds.copy()
    ds.samplesheet = ds.samplesheet.iloc[:2]
    ds2.samplesheet = ds2.samplesheet.iloc[2:]
    return (ds, ds2)


def test_mann_whitney(ds_ds2):
    (ds, ds2) = ds_ds2
    print('Test feature comparison (Mann-Whitney U)')
    pvals = ds.compare(
            ds2,
            method='mann-whitney')
    assert(np.isclose(pvals.values.min(), 0.193931))
    print('Done!')


def test_ks(ds_ds2):
    (ds, ds2) = ds_ds2
    print('Test feature comparison (Kolmogorov-Smirnov)')
    pvals = ds.compare(
            ds2,
            method='kolmogorov-smirnov')
    assert(np.isclose(pvals.values.min(), 0.097027))
    print('Done!')


def test_ks_pheno(ds_ds2):
    (ds, ds2) = ds_ds2
    print('Test phenotype comparison (Kolmogorov-Smirnov)')
    pvals = ds.compare(
            ds2,
            features=None,
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            method='kolmogorov-smirnov')
    assert(np.isclose(pvals.values.min(), 0.84382))
    print('Done!')


def test_mann_whitney_mixed(ds_ds2):
    (ds, ds2) = ds_ds2
    print('Test mixed comparison (Mann-Whitney U)')
    pvals = ds.compare(
            ds2,
            features=['TSPAN6', 'DPM1', 'MAT3'],
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            method='mann-whitney')
    assert(np.isclose(pvals.values.min(), 0.245278))
    print('Done!')


def test_custom(ds_ds2):
    (ds, ds2) = ds_ds2
    print('Test custom comparison')
    pvals = ds.compare(
            ds2,
            features=['TSPAN6', 'DPM1', 'MAT3'],
            phenotypes=['quantitative_phenotype_1_[A.U.]'],
            method=lambda x, y: 0.5 + 0.5 * float(x.min() < y.min()))
    assert(np.isclose(pvals.values.min(), 0.5))
    print('Done!')
