#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTable class.
'''
# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.counts_table import CountsTableXR
    ct = CountsTableXR.from_tablename('example_table_tsv')

    print('Test statistics of CountsTable')
    assert(ct.get_statistics(metrics=('min', 'cv')).iloc[0, 0] == 29.0)
    print('Done!')

    print('Test normalization of CountsTable')
    ctn = ct.normalize('counts_per_million')
    assert(int(ctn.data[0, 0]) == 147)
    print('Done!')

    print('Test inplace normalization of CountsTable')
    ctn = ct.copy()
    ctn.normalize('counts_per_million', inplace=True)
    assert(int(ctn.data[0, 0]) == 147)
    print('Done!')
