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
    from singlet.io.googleapi import parse_samplesheet

    print('Parsing example Google Drive API sample sheet')
    table = parse_samplesheet('example_sheet_googleapi')

    #TODO: check it's correct
    print('Done!')
