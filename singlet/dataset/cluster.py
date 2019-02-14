# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to cluster samples, features, and phenotypes.
# Modules
import numpy as np
import pandas as pd

from .plugins import Plugin


# Classes / functions
class Cluster(Plugin):
    '''Cluster samples, features, and phenotypes'''

    def kmeans(
            self,
            n_clusters,
            axis,
            phenotypes=(),
            random_state=0):
        '''K-Means clustering.

        Args:
            n_clusters (int): The number of clusters you want.
            axis (string): It must be 'samples' or 'features'.
                The Dataset.counts matrix is used and
                either samples or features are clustered.
            phenotypes (iterable of strings): Phenotypes to add to the
                features for joint clustering.
            log_features (bool): Whether to add pseudocounts and take a log
                of the feature counts before calculating distances.
            random_state (int): Set to the same int for deterministic results.

        Returns:
            pd.Series with the labels of the clusters.
        '''
        from sklearn.cluster import KMeans

        data = self.dataset.counts

        if phenotypes is not None:
            data = data.copy()
            for pheno in phenotypes:
                data.loc[pheno] = self.dataset.samplesheet.loc[:, pheno]

        if axis == 'samples':
            data = data.T
        elif axis == 'features':
            pass
        else:
            raise ValueError('axis must be "samples" or "features"')

        model = (KMeans(n_clusters=n_clusters, random_state=random_state)
                 .fit(data.values))
        labels = pd.Series(model.labels_, index=data.index, dtype='category')
        return labels

    def dbscan(
            self,
            axis,
            phenotypes=(),
            **kwargs):
        '''Density-Based Spatial Clustering of Applications with Noise.

        Args:
            axis (string): It must be 'samples' or 'features'.
                The Dataset.counts matrix is used and
                either samples or features are clustered.
            phenotypes (iterable of strings): Phenotypes to add to the
                features for joint clustering.
            log_features (bool): Whether to add pseudocounts and take a log
                of the feature counts before calculating distances.
            **kwargs: arguments passed to sklearn.cluster.DBSCAN.

        Returns:
            pd.Series with the labels of the clusters.
        '''
        from sklearn.cluster import DBSCAN

        data = self.dataset.counts

        if phenotypes is not None:
            data = data.copy()
            for pheno in phenotypes:
                data.loc[pheno] = self.dataset.samplesheet.loc[:, pheno]

        if axis == 'samples':
            data = data.T
        elif axis == 'features':
            pass
        else:
            raise ValueError('axis must be "samples" or "features"')


        model = DBSCAN(**kwargs).fit(data.values)
        labels = pd.Series(model.labels_, index=data.index, dtype='category')
        return labels

    # NOTE: caching this one is tricky because it has non-kwargs AND it would
    # need a double cache, one for cells and one for features/phenotypes
    def hierarchical(
            self,
            axis,
            phenotypes=(),
            metric='correlation',
            method='average',
            log_features=False,
            optimal_ordering=False):
        '''Hierarchical clustering.

        Args:
            axis (string): It must be 'samples' or 'features'. The
                Dataset.counts matrix is used and either samples or features
                are clustered.
            phenotypes (iterable of strings): Phenotypes to add to the
                features for joint clustering.
            metric (string or matrix): Metric to calculate the distance matrix.
                If it is a matrix already, use it as distance (squared). Else
                it should be a string accepted by scipy.spatial.distance.pdist.
            method (string): Clustering method. Must be a string accepted by
                scipy.cluster.hierarchy.linkage.
            log_features (bool): Whether to add pseudocounts and take a log
                of the feature counts before calculating distances.
            optimal_ordering (bool): Whether to resort the linkage so that
                nearest neighbours have shortest distance. This may take
                longer than the clustering itself.
        Returns:
            dict with the linkage, distance matrix, and ordering.
        '''
        from scipy.spatial.distance import pdist, squareform
        from scipy.cluster.hierarchy import linkage, leaves_list, optimal_leaf_ordering

        data = self.dataset.counts

        if log_features:
            data = np.log10(self.dataset.counts.pseudocount + data)

        if phenotypes is not None:
            data = data.copy()
            for pheno in phenotypes:
                data.loc[pheno] = self.dataset.samplesheet.loc[:, pheno]

        if axis == 'samples':
            data = data.T
        elif axis == 'features':
            pass
        else:
            raise ValueError('axis must be "samples" or "features"')

        if isinstance(metric, str):
            Y = pdist(data.values, metric=metric)
        else:
            Y = np.asarray(metric)
            assert len(Y.shape) == 2
            assert Y.shape[0] == Y.shape[1]
            assert Y.shape[0] == data.shape[0]
            Y = squareform(Y)

        # Some metrics (e.g. correlation) give nan whenever the matrix has no
        # variation, default this to zero distance (e.g. two features that are
        # both total dropouts.
        Y = np.nan_to_num(Y)

        Z = linkage(Y, method=method)

        if optimal_ordering:
            Z = optimal_leaf_ordering(Z, Y)

        ids = data.index[leaves_list(Z)]

        return {
                'distance': Y,
                'linkage': Z,
                'leaves': ids,
                }

    def affinity_propagation(
            self,
            axis,
            phenotypes=(),
            metric='correlation',
            log_features=False,
            ):
        '''Affinity/label/message propagation.

        Args:
            axis (string): It must be 'samples' or 'features'. The
                Dataset.counts matrix is used and either samples or features
                are clustered.
            phenotypes (iterable of strings): Phenotypes to add to the
                features for joint clustering.
            metric (string or matrix): Metric to calculate the distance matrix.
                If it is a matrix already, use it as distance (squared). Else
                it should be a string accepted by scipy.spatial.distance.pdist.
            log_features (bool): Whether to add pseudocounts and take a log
                of the feature counts before calculating distances.
        Returns:
            dict with the linkage, distance matrix, and ordering.
        '''
        from scipy.spatial.distance import pdist, squareform
        from sklearn.cluster import AffinityPropagation

        data = self.dataset.counts

        if log_features:
            data = np.log10(self.dataset.counts.pseudocount + data)

        if phenotypes is not None:
            data = data.copy()
            for pheno in phenotypes:
                data.loc[pheno] = self.dataset.samplesheet.loc[:, pheno]

        if axis == 'samples':
            data = data.T
        elif axis == 'features':
            pass
        else:
            raise ValueError('axis must be "samples" or "features"')

        if isinstance(metric, str):
            Y = squareform(pdist(data.values, metric=metric))
        else:
            Y = np.asarray(metric)
            assert len(Y.shape) == 2
            assert Y.shape[0] == Y.shape[1]
            assert Y.shape[0] == data.shape[0]

        # Affinity is the opposite of distance
        Y = -Y
        model = AffinityPropagation(
                affinity='precomputed',
                )
        model.fit(Y)
        labels = pd.Series(model.labels_, index=data.index, dtype='category')
        return labels
