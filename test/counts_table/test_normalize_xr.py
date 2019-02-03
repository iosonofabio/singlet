# vim: fdm=indent
'''
author:     Fabio Zanini
date:       31/01/19
content:    Test normalization of counts table.
'''
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ct():
    from singlet.counts_table import CountsTableXR
    return CountsTableXR.from_tablename('example_table_tsv')


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


# FIXME
#def test_normalize_withspikeins(ct):
#    print('Normalize per 1,000 spikeins')
#    assert(ct.normalize(method='counts_per_thousand_spikeins').data[0, 0] == 112.91291291291292)
#    print('Done!')


#def test_normalize_withspikeins_inplace(ct):
#    print('Normalize per 1,000 spikeins inplace')
#    ct2 = ct.copy()
#    ct2.normalize(method='counts_per_thousand_spikeins', inplace=True)
#    assert(ct2.iloc[0, 0] == 112.91291291291292)
#    print('Done!')
#
#
#def test_normalize_thousand_features(ct):
#    print('Normalize per 1,000 features')
#    feas = ct.index[:10]
#    assert(ct.normalize(
#        method='counts_per_thousand_features',
#        features=feas,
#        ).iloc[0, 0] == 191.83673469387756)
#    print('Done!')
#
#
#def test_normalize_thousand_features_inplace(ct):
#    print('Normalize per 1,000 features')
#    feas = ct.index[:10]
#    ct2 = ct.copy()
#    ct2.normalize(
#        method='counts_per_thousand_features',
#        features=feas,
#        inplace=True
#        )
#    assert(ct2.iloc[0, 0] == 191.83673469387756)
#    print('Done!')
