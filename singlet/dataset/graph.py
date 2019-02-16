# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to do graph analysis
# Modules
import numpy as np
import pandas as pd
import xarray as xr
from .plugins import Plugin


# Classes / functions
class Graph(Plugin):
    '''Graph analysis of gene expression and phenotype in single cells'''

    def knn(self,
            axis='samples',
            n_neighbors=20,
            threshold=0.2,
            return_sparse=True,
            metric='pearson',
            metric_kwargs=None,
            ):
        '''K nearest neighbors.

        Args:
            axis (str): 'samples' or 'features'
            n_neighbors (int): number of neighbors to include
            threshold (float): similarity threshold to cut neighbors at
            n_planes (int): number of random hyperplanes to use for signature
            slice_length (int or None): number of bits to use for the LSH. If
                None, perform all n^2 comparisons of signatures
            return_sparse (bool): return sparse matrix instead of raw lshknn
                output.
            metric (str): metric to use to calculate the distance matrix. If
                this is a distance metric, similarity is -distance.
            metric_kwargs (dict or None): dictionary of keyword arguments for
                scipy.spatial.distance.pdist.

        Returns:
            tuple with (knn, similarity, n_neighbors) or COO sparse matrix
            similarities. The sparse matrix is NOT symmetric: each row has
            the k neighbors of the sample/feature corresponding to that row.
        '''
        from scipy.sparse import coo_matrix
        from scipy.spatial.distance import pdist, squareform

        if metric_kwargs is None:
            metric_kwargs = {}

        if (axis == 'samples') and (n_neighbors >= self.dataset.n_samples):
            raise ValueError('n_neighbors must be less than the number of samples')
        if (axis == 'features') and (n_neighbors >= self.dataset.n_features):
            raise ValueError('n_neighbors must be less than the number of features')

        # Get full similarity matrix
        if metric in ('pearson', 'spearman'):
            if axis == 'samples':
                similarity_matrix = self.dataset.correlation.correlate_samples(
                        samples='all',
                        method=metric,
                        )
            elif axis == 'features':
                similarity_matrix = self.dataset.correlation.correlate_features_features(
                        features='all',
                        method=metric,
                        )
            else:
                raise ValueError('axis not understood')
        else:
            data = self.dataset.counts.values
            if axis == 'samples':
                pass
            elif axis == 'features':
                data = data.T
            else:
                raise ValueError('axis not understood')

            similarity_matrix = -squareform(pdist(data, metric=metric, **metric_kwargs))

        # Get top k neighbors
        knn = []
        similarity = []
        nn_neighbors = []
        for irow, row in enumerate(similarity_matrix.values):
            knn.append([])
            similarity.append([])
            row[irow] = -np.inf
            ind = np.argpartition(row, -n_neighbors)[-n_neighbors:]
            indi = ind[row[ind] >= threshold]
            for i in indi:
                knn[-1].append((irow, i))
                similarity[-1].append(row[i])
            nn_neighbors.append(len(indi))

        if not return_sparse:
            return (knn, similarity, nn_neighbors)

        data = []
        i = []
        j = []
        for irow, (n, sim, nn) in enumerate(zip(knn, similarity, nn_neighbors)):
            for icoli, icol in enumerate(n[:nn]):
                data.append(sim[icoli])
                # NOTE: irow == icol[0], it's the first of the two nodes
                i.append(irow)
                j.append(icol[1])

        matrix = coo_matrix(
                (data, (i, j)),
                shape=(len(knn), len(knn)),
                )
        return matrix

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
            slice_length (int or None): number of bits to use for the LSH. If
                None, perform all n^2 comparisons of signatures
            return_sparse (bool): return sparse matrix instead of raw lshknn
                output.

        Returns:
            tuple with (knn, similarity, n_neighbors) or COO sparse matrix
            similarities. The sparse matrix is NOT symmetric: each row has
            the k neighbors of the cell corresponding to that row.
        '''
        from scipy.sparse import coo_matrix
        import lshknn

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

        if not return_sparse:
            return (knn, similarity, n_neighbors)

        data = []
        i = []
        j = []
        for irow, (n, sim, nn) in enumerate(zip(knn, similarity, n_neighbors)):
            for icoli, icol in enumerate(n[:nn[0]]):
                data.append(sim[icoli])
                i.append(irow)
                j.append(icol)

        matrix = coo_matrix(
                (data, (i, j)),
                shape=(knn.shape[0], knn.shape[0]),
                )

        return matrix
