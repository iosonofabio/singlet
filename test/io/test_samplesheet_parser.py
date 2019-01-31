#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test parser for sample sheets.
'''
def test_parse_samplesheet():
    from singlet.io import parse_samplesheet
    print('Parsing example sample sheet')
    table = parse_samplesheet({'sheetname': 'example_sheet_tsv'})
    print('Done!')
