#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import sys
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    dset = Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')

    dset.counts.exclude_features(spikeins=True, other=True, inplace=True)
    return dset


def test_unique(ds):
    res = ds.feature_selection.unique()
    assert('TSPAN6' in res)


def test_expression(ds):
    res = ds.feature_selection.expressed(n_samples=1, exp_min=1)
    assert(res[0] == 'TSPAN6')


def test_expression_inplace(ds):
    dsp = ds.copy()
    dsp.feature_selection.expressed(n_samples=1, exp_min=1, inplace=True)
    assert(dsp.featurenames[0] == 'TSPAN6')


def test_overdispersed_strata(ds):
    res = ds.feature_selection.overdispersed_strata()
    assert(res[-1] == 'FTL')


def test_overdispersed_strata_inplace(ds):
    dsp = ds.copy()
    dsp.feature_selection.overdispersed_strata(inplace=True)
    assert(dsp.featurenames[-1] == 'FTL')


@pytest.mark.xfail(True, reason='SAM has fragile deps and APIs')
def test_selfassemblingmanifolds(ds):
    dsp = ds.copy()
    sam = dsp.feature_selection.sam(npcs=3)
    weights = sam.output_vars['gene_weights']
