#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTable class.
'''
def test_initialize():
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_tablename('example_table_tsv')


def test_initialize_fromdataset():
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_datasetname('example_dataset')


def test_initialize_128():
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_tablename('example_table_tsv_float128')


def test_initialize_32():
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_tablename('example_table_tsv_float32')


def test_initialize_16():
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_tablename('example_table_tsv_float16')
