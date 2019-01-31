#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import pytest


def test_initialize():
    print('Instantiating Dataset')
    from singlet.dataset import Dataset
    ds = Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')
    print('Done!')
