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
    from singlet.io.csv import parse_samplesheet

    print('Parsing example TSV sample sheet')
    table = parse_samplesheet('example_sheet_tsv')

    #TODO: check it's correct
    print('Done!')
