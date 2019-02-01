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
    assert(ct.all() == False)


def test_any(ct):
    assert(ct.any() == True)



