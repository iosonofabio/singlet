#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTable class.
'''
import pytest


@pytest.fixture(scope="module")
def ct():
    from singlet.counts_table import CountsTable
    return CountsTable.from_tablename('example_table_tsv').iloc[:200]


def test_bin(ct):
    print('Test binning of CountsTable')
    ct2 = ct.bin(result='index', inplace=False)
    assert(ct2.values.max() == 4)
    print('Done!')


def test_bin_inplace(ct):
    print('Test binning of CountsTable inplace')
    ct.bin(result='index', inplace=True)
    assert(ct.values.max() == 4)
    print('Done!')
