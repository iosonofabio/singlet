#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import sys
import pytest


def test_initialize():
    from singlet.dataset import Dataset
    return Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')


def test_initialize_fromdataset():
    from singlet.dataset import Dataset
    return Dataset(dataset='example_dataset')


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_initialize_fromdataset_integrated():
    from singlet.dataset import Dataset
    return Dataset(dataset='PBMC_loom')


def test_plugins():
    from singlet.dataset import Dataset, Plugin

    ds = Dataset(
            samplesheet='example_sheet_tsv',
            plugins={'testplugin': Plugin})
    assert(ds.testplugin.dataset == ds)
