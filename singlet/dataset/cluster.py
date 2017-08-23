# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to cluster samples, features, and phenotypes.
# Modules
import numpy as np
import pandas as pd

from ..utils.cache import method_caches


# Classes / functions
class Cluster():
    '''Cluster samples, features, and phenotypes'''
    def __init__(self, dataset):
        '''Cluster samples, features, and phenotypes

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    # NOTE: caching this one is tricky because it has non-kwargs AND it would
    # need a double cache, one for cells and one for features/phenotypes
    def hierarchical(
            self,
            axis,
            phenotypes=(),
            metric='correlation',
            method='average',
            log_features=True,
            optimal_ordering=False,
            **kwargs):
        '''Hierarchical clustering.

        Args:
            axis (string): It must be 'samples' or 'features'. \
                    The Dataset.counts matrix is used and \
                    either samples or features are clustered.
            phenotypes (iterable of strings): Phenotypes to add to the \
                    features for joint clustering.
            metric (string): Metric to calculate the distance matrix. Should \
                    be a string accepted by scipy.spatial.distance.pdist.
            method (string): Clustering method. Must be a string accepted by \
                    scipy.cluster.hierarchy.linkage.
            log_features (bool): Whether to add pseudocounts and take a log \
                    of the feature counts before calculating distances.
            optimal_ordering (bool): Whether to resort the linkage so that \
                    nearest neighbours have shortest distance. This may take \
                    longer than the clustering itself.
        Return:
            dict with the linkage, distance matrix, and ordering.
        '''
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list

        if optimal_ordering:
            try:
                from polo import optimal_leaf_ordering
            except ImportError:
                raise ImportError(
                    'The package "polo" is needed for optimal leaf ordering')

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

        Y = pdist(data.values, metric=metric)

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
