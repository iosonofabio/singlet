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
            'samples',
            optimal_ordering=False,
            log_features=True)
    assert(tuple(d['leaves']) == ('second_sample', 'third_sample',
                                  'test_pipeline', 'first_sample'))
    print('Done!')

    print('Hierarchical clustering of features')
    ds.counts = ds.counts.iloc[:200]
    d = ds.cluster.hierarchical(
            'features',
            optimal_ordering=False,
            log_features=True)
    assert(tuple(d['leaves'])[:3] == ('PNPLA4', 'RHBDF1', 'ITGAL'))
    print('Done!')

    print('Hierarchical clustering of features and phenotypes')
    ds.counts = ds.counts.iloc[:200]
    d = ds.cluster.hierarchical(
            axis='features',
            phenotypes=('quantitative_phenotype_1_[A.U.]',),
            optimal_ordering=True,
            log_features=True)
    #FIXME: ordering seems to be slightly nondeterministic?
    assert('quantitative_phenotype_1_[A.U.]' in d['leaves'])
    print('Done!')
