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
    from singlet.counts_table import CountsTable
    return CountsTable.from_tablename('example_table_tsv')


def test_getspikeins(ct):
    print('Get spikeins')
    assert(ct.get_spikeins().index[0] == 'ERCC-00002')
    print('Done!')


def test_getother(ct):
    print('Get spikeins')
    assert(ct.get_other_features().index[0] == 'NIST_ConsensusVector')
    print('Done!')


def test_log(ct):
    print('Log')
    assert(np.isclose(ct.log().iloc[0, 0], 2.274388795550379))
    assert(np.isclose(ct.log().unlog().iloc[0, 0], 188.0))
    print('Done!')


def test_center(ct):
    print('Center')
    assert(np.isclose(ct.center().iloc[0, 0], -63))
    print('Done!')


def test_zscore(ct):
    print('z score')
    assert(np.isclose(ct.z_score().iloc[0, 0], -0.41125992263799543))
    print('Done!')


def test_standard_scale(ct):
    print('Standard scale')
    assert(np.isclose(ct.standard_scale().iloc[0, 0], 0.43561643835616437))
    print('Done!')
