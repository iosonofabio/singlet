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
    (knn, similarity, neighbors) = ds.graph.knn(n_neighbors=2, return_sparse=False)
    assert(neighbors == [2, 2, 2, 2])


def test_knn_sparse(ds):
    knn = ds.graph.knn(n_neighbors=2, return_sparse=True)
    assert(knn.row[0] == 0)


def test_knn_features(ds):
    ds2 = ds.query_features_by_name(ds.featurenames[:50])
    (knn, similarity, neighbors) = ds2.graph.knn(
            n_neighbors=2,
            return_sparse=False,
            axis='features')
    assert(neighbors[:4] == [0, 0, 0, 0])


@pytest.mark.skipif(miss_lshknn, reason='No lshknn available')
def test_lshknn(ds):
    (knn, similarity, neighbors) = ds.graph.lshknn(
        n_neighbors=2,
        return_sparse=False)
    assert(list(neighbors) == [2, 2, 2, 2])


@pytest.mark.skipif(miss_lshknn, reason='No lshknn available')
def test_lshknn_sparse(ds):
    knn = ds.graph.lshknn(
        n_neighbors=2,
        return_sparse=True)
    assert(knn.row[0] == 0)
