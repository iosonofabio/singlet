# vim: fdm=indent
'''
author:     Fabio Zanini
date:       30/01/19
content:    Test parsing LOOM files.
'''
import sys
import pytest


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_loom_dataset():
    print('Parse integrated dataset as loom file')
    from singlet.io.loom import parse_dataset
    ds = parse_dataset(
        'example_data/dataset_PBMC.loom',
        axis_samples='columns',
        index_samples='_index',
        index_features='EnsemblID',
        )
    print('Done')


@pytest.mark.skipif(sys.version_info < (3, 6),
                    reason="requires python3.6 or higher")
def test_loom_dataset_config():
    print('Parse loom dataset from config file')
    from singlet.io import parse_dataset
    ds = parse_dataset({'datasetname': 'PBMC_loom'})
    print('Done')
