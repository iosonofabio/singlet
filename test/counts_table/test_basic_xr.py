#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTableSparse class.
'''
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ct():
    print('Instantiating CountsTableXR')
    from singlet.counts_table import CountsTableXR
    ctable = CountsTableXR.from_tablename('example_table_tsv')
    print('Done!')
    return ctable


def test_str(ct):
    assert(str(ct)[:42] == "<singlet.CountsTableXR 'example_table_tsv'")


def test_repr(ct):
    assert(ct.__repr__()[:42] == "<singlet.CountsTableXR 'example_table_tsv'")


def test_log(ct):
    ctlog = ct.log(base=10)
    ctunlog = ctlog.unlog(base=10)


def test_eq(ct):
    assert(ct.__eq__(ct))


def test_abs(ct):
    assert(ct.__abs__() == ct)


def test_add(ct):
    assert((ct.__add__(ct)._data == 2 * ct._data).all())


def test_and(ct):
    assert(((ct == 3).__and__((ct == 3)))._data.data.sum() == 1278)


def test_copy(ct):
    assert(ct.__copy__() == ct)


def test_deepcopy(ct):
    assert(ct.__deepcopy__() == ct)


def test_ge(ct):
    assert(ct.__ge__(ct))


def test_gt(ct):
    assert(ct.__gt__(ct)._data.data.sum() == 0)


def test_le(ct):
    assert(ct.__le__(ct))


def test_lt(ct):
    assert(ct.__lt__(ct)._data.data.sum() == 0)


def test_mod(ct):
    assert(ct.__mod__(1) == ct)


def test_mul(ct):
    assert(ct.__mul__(1) == ct)


def test_neg(ct):
    ct2 = ct.__copy__()
    ct2._data = -ct2._data
    assert(ct.__neg__() == ct2)


def test_or(ct):
    assert(((ct == 3).__or__((ct == 3)))._data.data.sum() == 1278)


def test_xor(ct):
    assert(((ct == 3).__xor__((ct == 3)))._data.data.sum() == 0)


def test_all(ct):
    assert(bool(ct.all().data) is False)


def test_any(ct):
    assert(bool(ct.any().data) is True)


def test_getitem(ct):
    assert(ct[0, 0]._data.data == 188.0)


def test_delitem(ct):
    ct2 = ct.__copy__()
    del ct2['gene name']
    assert(list(ct2.coords.keys()) == ['sample name'])


def test_radd(ct):
    assert((ct.__radd__(ct)._data == 2 * ct._data).all())


def test_rand(ct):
    assert(((ct == 3).__rand__((ct == 3)))._data.data.sum() == 1278)


def test_rmod(ct):
    assert(ct.__rmod__(1) == ct)


def test_rmul(ct):
    assert(ct.__rmul__(1) == ct)


def test_ror(ct):
    assert(((ct == 3).__ror__((ct == 3)))._data.data.sum() == 1278)


def test_rtruediv(ct):
    assert(ct.__rtruediv__(1) == ct)


def test_rxor(ct):
    assert(((ct == 3).__rxor__((ct == 3)))._data.data.sum() == 0)


def test_rsub(ct):
    assert((ct.__rsub__(ct)._data == 0).all())


def test_sub(ct):
    assert((ct.__sub__(ct)._data == 0).all())


def test_truediv(ct):
    assert(ct.__truediv__(1) == ct)


def test_dims(ct):
    assert(ct.dims == ('gene name', 'sample name'))


def test_dot(ct):
    assert(np.isclose(float(ct.dot(ct)), 1.08029e+13))


def test_dropna(ct):
    assert(ct.dropna(dim='gene name') == ct)


def test_equals(ct):
    assert(ct.equals(ct))


def test_fillna(ct):
    assert(ct.fillna(0) == ct)


def test_get_axis_num(ct):
    assert(ct.get_axis_num('gene name') == 0)


def test_identical(ct):
    assert(ct.identical(ct))


def test_max(ct):
    assert(ct.max() == 1976680.0)


def test_min(ct):
    assert(ct.min() == 0)


def test_mean(ct):
    assert(np.isclose(ct.mean(), 45.644336))


def test_var(ct):
    assert(np.isclose(ct.var(), 44475514.709833))


def test_median(ct):
    assert(ct.median() == 0)


def test_shape(ct):
    assert(ct.shape == (60721, 4))


# FIXME: how does this work?
#def test_swap_dims(ct):
#    assert(ct.swap_dims(
#        {'gene name': 'sample name',
#         'sample name': 'gene name'}).shape == (4, 60721))


def test_T(ct):
    ct = ct.__copy__()
    assert(ct.T.shape == (4, 60721))


def test_transpose(ct):
    assert(ct.transpose('sample name', 'gene name').shape == (4, 60721))


def test_values(ct):
    assert(ct.values.shape == (60721, 4))


# FIXME
#def test_getspikeins(ct):
#    print('Get spikeins')
#    assert(ct.get_spikeins().index[0] == 'ERCC-00002')
#    print('Done!')
#
#
#def test_getother(ct):
#    print('Get spikeins')
#    assert(ct.get_other_features().index[0] == 'NIST_ConsensusVector')
#    print('Done!')
