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
    from singlet.dataset import Dataset
    ds = Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')

    print('Query sample by counts in one gene')
    ds_tmp = ds.query_samples_by_counts('KRIT1 > 100', inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('third_sample',))
    print('Done!')

    print('Query sample by total counts')
    ds_tmp = ds.query_samples_by_counts('total < 3000000', inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('second_sample',))
    print('Done!')

    print('Query sample by mapped counts')
    ds_tmp = ds.query_samples_by_counts('mapped < 100000', inplace=False)
    assert(tuple(ds_tmp.samplenames) == ('second_sample',))
    print('Done!')

    print('Query features')
    ds_tmp = ds.query_features('first_sample > 1000000', inplace=False)
    assert(tuple(ds_tmp.featurenames) == ('__alignment_not_unique',))
    print('Done!')
