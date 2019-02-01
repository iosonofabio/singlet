#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test SampleSheet class.
'''
def test_initialize():
    from singlet.samplesheet import SampleSheet
    ss = SampleSheet.from_sheetname('example_sheet_tsv')


def test_initialize_fromdataset():
    from singlet.samplesheet import SampleSheet
    ct = SampleSheet.from_datasetname('example_dataset')

