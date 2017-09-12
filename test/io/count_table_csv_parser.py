#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test CSV/TSV parser for sample sheets.
'''
# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    print('Parsing example TSV count table')
    from singlet.io import parse_counts_table
    table = parse_counts_table('example_table_tsv')
    print('Done!')

    print('Parsing example TSV count table (split)')
    from singlet.io import parse_counts_table
    table = parse_counts_table('example_table_tsv_split')
    print('Done!')
