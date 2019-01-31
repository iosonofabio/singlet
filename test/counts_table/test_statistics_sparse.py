#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTable class.
'''
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ct():
    from singlet.counts_table import CountsTableSparse
    return CountsTableSparse.from_tablename('example_PBMC_sparse')


def test_statistics(ct):
    print('Test statistics of CountsTable')
    ctn = ct.iloc[-200:]
    assert(np.allclose(ctn.get_statistics(metrics=('min', 'cv')).iloc[0, 1], 1.327366))
    print('Done!')


def test_normalization(ct):
    print('Test normalization of CountsTable')
    ctn = ct.iloc[-200:]
    ctn = ctn.normalize('counts_per_million')
    assert(int(ctn.iloc[-2, -1]) == 17070)
    print('Done!')
