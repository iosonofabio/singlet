#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test parser for sample sheets.
'''
# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.io import parse_samplesheet
    print('Parsing example TSV sample sheet')
    table = parse_samplesheet('example_sheet_tsv')
    print('Done!')

    print('Instantiating SampleSheet')
    from singlet.samplesheet import SampleSheet
    table = SampleSheet.from_sheetname('example_sheet_tsv')
    print('Done!')
