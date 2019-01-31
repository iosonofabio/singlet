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
    return Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')


@pytest.mark.xfail(True, reason='TODO: check this test')
def test_bootstrap(ds):
    print('Bootstrap')
    dsboot = ds.bootstrap()
    assert('--sampling_' in dsboot.samplenames[0])
    print('Done!')


def test_bootstrap_by_group_string(ds):
    print('Bootstrap by group, string')
    dsboot = ds.bootstrap(groupby='experiment')
    assert('--sampling_' in dsboot.samplenames[0])
    print('Done!')


def test_bootstrap_by_group_list(ds):
    print('Bootstrap by group, list')
    dsboot = ds.bootstrap(groupby=['experiment', 'barcode'])
    assert('--sampling_' in dsboot.samplenames[0])
    print('Done!')
