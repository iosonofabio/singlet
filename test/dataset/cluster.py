#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import numpy as np
import scipy as sp


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')

    print('Hierarchical clustering of samples')
    d = ds.cluster.hierarchical(
            axis='samples',
            optimal_ordering=True)
    assert(tuple(d['leaves']) == ('second_sample', 'test_pipeline',
                                  'first_sample', 'third_sample'))
    print('Done!')

    print('Hierarchical clustering of features')
    ds.counts = ds.counts.iloc[:200]
    d = ds.cluster.hierarchical(
            axis='features',
            optimal_ordering=True)
    assert(tuple(d['leaves'])[:3] == ('PNPLA4', 'RHBDF1', 'HOXA11'))
    print('Done!')

    print('Hierarchical clustering of features and phenotypes')
    ds.counts = ds.counts.iloc[:200]
    d = ds.cluster.hierarchical(
            axis='features',
            phenotypes=('quantitative_phenotype_1_[A.U.]',),
            optimal_ordering=True)
    assert(d['leaves'][23] == 'quantitative_phenotype_1_[A.U.]')
    print('Done!')
