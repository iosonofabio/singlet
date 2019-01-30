#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    print('Instantiating Dataset')
    from singlet.dataset import Dataset
    ds = Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')
    print('Done!')

    print('Testing Dataset.__str__')
    assert(str(ds) == 'Dataset with 4 samples and 60721 features')
    print('Done!')

    print('Testing Dataset.__repr__')
    assert(ds.__repr__() == '<Dataset: 4 samples, 60721 features>')
    print('Done!')

    print('Testing Dataset.copy')
    assert(ds.copy() == ds)
    print('Done!')

    print('Testing Dataset.copy with modifications')
    dsp = ds.copy()
    dsp._counts.iloc[0, 0] = -5
    assert(dsp != ds)
    print('Done!')

    print('Testing injection into counts table')
    dsp = ds.copy()
    dsp.counts.exclude_features(inplace=True)
    assert(dsp.counts.shape[0] == dsp.featuresheet.shape[0])
    print('Done!')
