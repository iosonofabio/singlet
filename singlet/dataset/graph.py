# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to do graph analysis
# Modules
import numpy as np
import pandas as pd
import xarray as xr


# Classes / functions
class Graph():
    '''Graph analysis of gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Graph analysis of gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    def lshknn(
            self,
            axis='samples',
            n_neighbors=20,
            threshold=0.2,
            n_planes=100,
            slice_length=None,
            return_sparse=True,
            ):
        '''K nearest neighbors via Local Sensitive Hashing (LSH).

        Args:
            axis (str): 'samples' or 'features'
            n_neighbors (int): number of neighbors to include
            threshold (float): similarity threshold to cut neighbors at
            n_planes (int): number of random hyperplanes to use for signature
            slice_length (int or None): number of bits to use for the LSH. If \
                    None, perform all n^2 comparisons of signatures

        Returns:
            tuple with (knn, similarity, n_neighbors)
        '''
        from scipy.sparse import coo_matrix
        import lshknn

        # TODO: decide on what to do with DataFrames
        data = self.dataset.counts.values
        if axis == 'samples':
            pass
        elif axis == 'features':
            data = data.T
        else:
            raise ValueError('axis not understood')

        c = lshknn.Lshknn(
                data=data,
                k=n_neighbors,
                threshold=threshold,
                m=n_planes,
                slice_length=slice_length,
                )
        (knn, similarity, n_neighbors) = c()

        data = []
        i = []
        j = []
        for irow, (n, sim, nn) in enumerate(zip(knn, similarity, n_neighbors)):
            for icoli, icol in enumerate(n[:nn[0]]):
                data.append(sim[icoli])
                i.append(irow)
                j.append(icol)
                # Notice: the matrix is symmetric
                i.append(icol)
                j.append(irow)
                data.append(sim[icoli])

        matrix = coo_matrix(
                (data, (i, j)),
                shape=(knn.shape[0], knn.shape[0]),
                )

        return matrix
