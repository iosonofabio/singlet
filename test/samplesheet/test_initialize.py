#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test SampleSheet class.
'''
def test_initialize():
    print('Instantiating SampleSheet')
    from singlet.samplesheet import SampleSheet
    ss = SampleSheet.from_sheetname('example_sheet_tsv')
    print('Done!')
