#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test CSV/TSV parser for sample sheets.
'''
def test_parse_tsv():
    print('Parsing example TSV count table')
    from singlet.io import parse_counts_table
    table = parse_counts_table({'countsname': 'example_table_tsv'})
    print('Done!')


def test_parse_csv_split():
    print('Parsing example TSV count table (split)')
    from singlet.io import parse_counts_table
    table = parse_counts_table({'countsname': 'example_table_tsv_split'})
    print('Done!')
