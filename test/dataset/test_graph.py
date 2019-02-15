# vim: fdm=indent
'''
author:     Fabio Zanini
date:       31/01/19
content:    Test graphs.
'''
import numpy as np
import pytest


try:
    import lshknn
    miss_lshknn = False
except ImportError:
    miss_lshknn = True



@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')


def test_knn(ds):
    print('KNN graph')
    (knn, similarity, neighbors) = ds.graph.knn(n_neighbors=2, return_sparse=False)
    assert(neighbors == [2, 2, 2, 2])
    print('Done')


def test_knn_sparse(ds):
    print('KNN graph')
    knn = ds.graph.knn(n_neighbors=2, return_sparse=True)
    assert(knn.row[0] == 0)
    print('Done')


@pytest.mark.skipif(miss_lshknn, reason='No lshknn available')
def test_lshknn(ds):
    print('KNN graph')
    (knn, similarity, neighbors) = ds.graph.lshknn(n_neighbors=2, return_sparse=False)
    assert(list(neighbors) == [2, 2, 2, 2])
    print('Done')
