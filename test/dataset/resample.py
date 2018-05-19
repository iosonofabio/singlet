#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import numpy as np


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')

    #print('Bootstrap')
    #dsboot = ds.bootstrap()
    #assert('--sampling_' in dsboot.samplenames[0])
    #print('Done!')

    print('Bootstrap by group, string')
    dsboot = ds.bootstrap(groupby='experiment')
    assert('--sampling_' in dsboot.samplenames[0])
    print('Done!')

    print('Bootstrap by group, list')
    dsboot = ds.bootstrap(groupby=['experiment', 'barcode'])
    assert('--sampling_' in dsboot.samplenames[0])
    print('Done!')
