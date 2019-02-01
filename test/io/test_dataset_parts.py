# vim: fdm=indent
'''
author:     Fabio Zanini
date:       31/01/19
content:    Test dataset built up of separate files.
'''
def test_parse_counts_table():
    print('Parsing example TSV count table from dataset')
    from singlet.io import parse_counts_table
    table = parse_counts_table({'datasetname': 'example_dataset'})
    print('Done!')


def test_parse_samplesheet():
    print('Parsing example TSV count table from dataset')
    from singlet.io import parse_samplesheet
    table = parse_samplesheet({'datasetname': 'example_dataset'})
    print('Done!')


def test_parse_featuresheet():
    print('Parsing example TSV count table from dataset')
    from singlet.io import parse_featuresheet
    table = parse_featuresheet({'datasetname': 'example_dataset'})
    print('Done!')
