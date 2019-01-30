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

    print('Test linear fit of phenotypes')
    res = ds.fit.fit_single(
            xs=['TSPAN6', 'DPM1'],
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='linear')
    assert(np.allclose(res[0, 0], [-0.005082, 3.548869, 0.599427],
                       rtol=1e-03, atol=1e-03))
    print('Done!')

    # TODO: assert result!
    print('Test nonlinear fit of phenotypes')
    res = ds.fit.fit_single(
            xs=['TSPAN6', 'DPM1'],
            ys=['quantitative_phenotype_1_[A.U.]'],
            model='threshold-linear')
    print('Done!')
