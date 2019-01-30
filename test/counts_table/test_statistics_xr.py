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
    from singlet.counts_table import CountsTableXR
    return CountsTableXR.from_tablename('example_table_tsv')


def test_statistics(ct):
    print('Test statistics of CountsTable')
    assert(ct.get_statistics(metrics=('min', 'cv')).iloc[0, 0] == 29.0)
    print('Done!')


def test_normalizaiton(ct):
    print('Test normalization of CountsTable')
    ctn = ct.normalize('counts_per_million')
    assert(int(ctn.data[0, 0]) == 147)
    print('Done!')


def test_normalization_inplace(ct):
    print('Test inplace normalization of CountsTable')
    ctn = ct.copy()
    ctn.normalize('counts_per_million', inplace=True)
    assert(int(ctn.data[0, 0]) == 147)
    print('Done!')


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    ct1 = ct()
    test_statistics(ct)
    test_normalizaiton(ct)
    test_normalization_inplace(ct)
