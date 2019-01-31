#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test pickle parser for counts table.
'''
def test_parse_pickle():
    print('Parsing example TSV count table')
    from singlet.io import parse_counts_table
    table = parse_counts_table({'countsname': 'example_table_pickle'})
    print('Done!')
