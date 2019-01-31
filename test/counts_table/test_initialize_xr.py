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
    print('Instantiating CountsTableXR')
    from singlet.counts_table import CountsTableXR
    ctable = CountsTableXR.from_tablename('example_table_tsv')
    print('Done!')
    return ctable


def test_log(ct):
    print('log CountsTableXR')
    ctlog = ct.log(base=10)
    print('Done!')

    print('unlog CountsTableXR')
    ctunlog = ctlog.unlog(base=10)
    print('Done!')
