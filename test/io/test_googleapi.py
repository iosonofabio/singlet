# vim: fdm=indent
'''
author:     Fabio Zanini
date:       31/01/19
content:    Test Google Sheet API access
'''
import pytest


@pytest.fixture(scope="module")
def ss():
    print('Google Sheet converters')
    from singlet.io.googleapi.samplesheet import SampleSheet
    return SampleSheet


def test_colind_to_A1(ss):
    assert(ss._convert_col_index_to_A1(1) == 'B')


def test_colind_to_A1_2(ss):
    assert(ss._convert_col_index_to_A1(27) == 'AB')


def test_rowcol_to_A1(ss):
    assert(ss.convert_row_col_to_A1(3, 4) == 'E4')


def test_rowcol_to_A24(ss):
    assert(ss.convert_row_col_to_A24(3, 4) == 'D5')


def test_A24_to_rowcol(ss):
    assert(ss.convert_A24_to_row_col('D5') == (3, 4))


def test_hex_to_rgb(ss):
    assert(ss.hex_to_rgb('#ffffff') == (1.0, 1.0, 1.0))


def test_rgb_to_hex(ss):
    assert(ss.rgb_to_hex((1.0, 1.0, 1.0)) == '#ffffff')
