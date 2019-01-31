#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTableSparse class.
'''
import pytest


@pytest.fixture(scope="module")
def ct():
    print('Instantiating CountsTableSparse')
    from singlet.counts_table import CountsTableSparse
    ctable = CountsTableSparse.from_tablename('example_PBMC_sparse')
    print('Done!')
    return ctable


def test_log(ct):
    print('log CountsTableSparse')
    ctlog = ct.log(base=10)
    print('Done!')

    print('unlog CountsTableSparse')
    ctunlog = ctlog.unlog(base=10)
    print('Done!')
