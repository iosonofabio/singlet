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


def test_str(ds):
    assert(str(ds) == 'Dataset with 4 samples and 60721 features')


def test_repr(ds):
    assert(ds.__repr__() == '<Dataset: 4 samples, 60721 features>')


def test_copy(ds):
    assert(ds.copy() == ds)


def test_copy_changes(ds):
    dsp = ds.copy()
    dsp._counts.iloc[0, 0] = -5
    assert(dsp != ds)


def test_injection_count_table(ds):
    dsp = ds.copy()
    dsp.counts.exclude_features(inplace=True)
    assert(dsp.counts.shape[0] == dsp.featuresheet.shape[0])


def test_add(ds):
    dsi = ds + ds
    assert(np.allclose(dsi.counts.values, 2 * ds.counts.values))


def test_add_renamed(ds):
    n = ds.n_samples
    ds2 = ds.copy()
    ds2.samplesheet['name2'] = [x+'_copy' for x in ds2.samplenames]
    ds2.reindex(axis='samples', column='name2', inplace=True, drop=True)
    dsi = ds + ds2
    assert(np.allclose(dsi.counts.values[:, :n], ds.counts.values))
    assert(np.allclose(dsi.counts.values[:, n:], ds.counts.values))
