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
    assert(ct.get_statistics(metrics=('min', 'cv', 'fano', 'max', 'var')).iloc[0, 0] == 29.0)
    print('Done!')
