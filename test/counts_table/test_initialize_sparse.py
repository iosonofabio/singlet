#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTableSparse class.
'''
def test_initialize():
    from singlet.counts_table import CountsTableSparse
    ctable = CountsTableSparse.from_tablename('example_PBMC_sparse')


def test_initialize_fromdataset():
    from singlet.counts_table import CountsTableSparse
    ctable = CountsTableSparse.from_datasetname('example_PBMC_sparse')
