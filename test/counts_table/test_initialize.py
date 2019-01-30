#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTable class.
'''
def test_initialize():
    print('Instantiating CountsTable')
    from singlet.counts_table import CountsTable
    ct = CountsTable.from_tablename('example_table_tsv')
    print('Done!')


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    test_initialize()
