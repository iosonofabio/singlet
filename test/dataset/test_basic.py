#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import pytest


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')


def test_str(ds):
    print('Testing Dataset.__str__')
    assert(str(ds) == 'Dataset with 4 samples and 60721 features')
    print('Done!')


def test_repr(ds):
    print('Testing Dataset.__repr__')
    assert(ds.__repr__() == '<Dataset: 4 samples, 60721 features>')
    print('Done!')


def test_copy(ds):
    print('Testing Dataset.copy')
    assert(ds.copy() == ds)
    print('Done!')


def test_copy_changes(ds):
    print('Testing Dataset.copy with modifications')
    dsp = ds.copy()
    dsp._counts.iloc[0, 0] = -5
    assert(dsp != ds)
    print('Done!')


def test_injection_count_table(ds):
    print('Testing injection into counts table')
    dsp = ds.copy()
    dsp.counts.exclude_features(inplace=True)
    assert(dsp.counts.shape[0] == dsp.featuresheet.shape[0])
    print('Done!')
