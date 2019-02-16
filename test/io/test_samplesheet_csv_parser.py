#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test CSV/TSV parser for sample sheets.
'''
def test_parse_samplesheet_tsv():
    from singlet.io import parse_samplesheet
    print('Parsing example TSV sample sheet')
    table = parse_samplesheet({'sheetname': 'example_sheet_tsv'})
    print('Done!')


def test_parse_samplesheet_csv():
    from singlet.io import parse_samplesheet
    print('Parsing example CSV sample sheet')
    table = parse_samplesheet({'sheetname': 'example_sheet_csv'})
    print('Done!')
