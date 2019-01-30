#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import numpy as np


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')

    print('KNN graph via all pair comparisons')
    res = ds.graph.lshknn(
            axis='samples',
            n_neighbors=1,
            threshold=0.2,
            n_planes=128,
            slice_length=None,
            )
    assert(np.allclose(
        res.data,
        [0.9996988186962041, 1.0, 1.0, 1.0, 0.9996988186962041, 1.0, 1.0, 1.0],
        rtol=1e-02, atol=1e-02))
    print('Done!')

    print('KNN graph via all LSH')
    res = ds.graph.lshknn(
            axis='samples',
            n_neighbors=1,
            threshold=0.2,
            n_planes=128,
            slice_length=4,
            )
    assert(np.allclose(
        res.data,
        [0.9996988186962041, 1.0, 1.0, 1.0, 0.9996988186962041, 1.0, 1.0, 1.0],
        rtol=5e-02, atol=5e-02))
    print('Done!')
