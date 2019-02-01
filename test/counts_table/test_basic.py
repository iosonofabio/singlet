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
    print('Done!')


def test_unlog(ct):
    ctlog = ct.log()
    assert(np.isclose(ctlog.unlog().iloc[0, 0], 188.0))
    ctlog.unlog(inplace=True)
    assert(np.isclose(ctlog.iloc[0, 0], 188.0))
    print('Done!')


def test_center(ct):
    print('Center')
    assert(np.isclose(ct.center().iloc[0, 0], -63))
    print('Done!')


def test_center_inplace(ct):
    print('Center')
    ct2 = ct.copy()
    ct2.center(inplace=True)
    assert(np.isclose(ct2.iloc[0, 0], -63))
    print('Done!')


def test_center_features(ct):
    print('Center')
    assert(np.isclose(ct.center(axis='features').iloc[0, 0], 131.90425058875843))
    print('Done!')


def test_zscore(ct):
    print('z score')
    assert(np.isclose(ct.z_score().iloc[0, 0], -0.41125992263799543))
    print('Done!')


def test_zscore_inplace(ct):
    print('z score')
    ct2 = ct.copy()
    ct2.z_score(inplace=True)
    assert(np.isclose(ct2.iloc[0, 0], -0.41125992263799543))
    print('Done!')


def test_zscore_features(ct):
    print('z score')
    assert(np.isclose(ct.z_score(axis='features').iloc[0, 0], 0.016413014240482873))
    print('Done!')


def test_standard_scale(ct):
    print('Standard scale')
    assert(np.isclose(ct.standard_scale().iloc[0, 0], 0.43561643835616437))
    print('Done!')


def test_standard_scale_inplace(ct):
    print('Standard scale')
    ct2 = ct.copy()
    ct2.standard_scale(inplace=True)
    assert(np.isclose(ct2.iloc[0, 0], 0.43561643835616437))
    print('Done!')


def test_standard_scale_features(ct):
    print('Standard scale')
    assert(np.isclose(ct.standard_scale(axis='features').iloc[0, 0], 9.510897059716292e-05))
    print('Done!')
