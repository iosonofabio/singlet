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
    return CountsTable.from_tablename('example_table_tsv')


def test_bin(ct):
    print('Test binning of CountsTable')
    ct = ct.iloc[:200]
    ct.bin(result='index', inplace=True)
    assert(ct.values.max() == 4)
    print('Done!')


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    ct1 = ct()
    test_bin(ct1)




