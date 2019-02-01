# vim: fdm=indent
'''
author:     Fabio Zanini
date:       31/01/19
content:    Test FeatureSheet initialization
'''
def test_initialize():
    from singlet.featuresheet import FeatureSheet
    ss = FeatureSheet.from_sheetname('example_sheet_tsv')


def test_initialize_fromdataset():
    from singlet.featuresheet import FeatureSheet
    ct = FeatureSheet.from_datasetname('example_dataset')


