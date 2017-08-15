#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test SampleSheet class.
'''
# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    print('Instantiating SampleSheet')
    from singlet.samplesheet import SampleSheet
    ss = SampleSheet.from_sheetname('example_sheet_tsv')
    print('Done!')
