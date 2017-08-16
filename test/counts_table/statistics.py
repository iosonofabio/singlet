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
    print('Test statistics of CountsTable')
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_tablename('example_table_tsv')

    assert(ct.get_statistics(metrics=('min', 'cv')).iloc[0, 0] == 0)
    print('Done!')
