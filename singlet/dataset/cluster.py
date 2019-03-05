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

    def _prepare_data(self, phenotypes, log_features=False):
        n_features = self.dataset.n_features
        n_phenotypes = len(phenotypes)
        n = n_features + n_phenotypes
        n_samples = self.dataset.n_samples
        data = np.zeros((n, n_samples), dtype=np.float32)

        # Features first
        data[:n_features] = self.dataset.counts.values
        data[np.isnan(data)] = np.nanmin(data[:n_features])
        if log_features:
            data[:n_features] += np.log10(self.dataset.counts.pseudocount + data[:n_features])

        # Phenotypes next
        for i, pheno in enumerate(phenotypes):
            data[n_features + i] = self.dataset.samplesheet.loc[:, pheno].values

        # Index and columns
        index = self.dataset.counts.index.tolist() + list(phenotypes)
        columns = self.dataset.counts.columns

        # Return as dataframe for convenience, it's dense so no data copying
        return pd.DataFrame(
                data=data,
                index=index,
                columns=columns,
                )

    def kmeans(
            self,
            n_clusters,
            axis,
            phenotypes=(),
            log_features=False,
            random_state=0,
            mini_batch=False):
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
            mini_batch (bool): Whether to use MiniBatchKMeans, a faster
                algorithm for large samples.

        Returns:
            pd.Series with the labels of the clusters.
        '''
        from sklearn.cluster import KMeans, MiniBatchKMeans

        if mini_batch:
            method = MiniBatchKMeans
        else:
            method = KMeans

        data = self._prepare_data(
            phenotypes=phenotypes,
            log_features=log_features,
            )

        if axis == 'samples':
            data = data.T
        elif axis == 'features':
            pass
        else:
            raise ValueError('axis must be "samples" or "features"')

        model = (method(n_clusters=n_clusters, random_state=random_state)
                 .fit(data.values))
        labels = pd.Series(model.labels_, index=data.index, dtype='category')
        return labels

    def dbscan(
            self,
            axis,
            phenotypes=(),
            log_features=False,
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

        data = self._prepare_data(
            phenotypes=phenotypes,
            log_features=log_features,
            )

        if axis == 'samples':
            data = data.T
        elif axis == 'features':
            pass
        else:
            raise ValueError('axis must be "samples" or "features"')

        model = DBSCAN(**kwargs).fit(data.values)
        labels = pd.Series(model.labels_, index=data.index, dtype='category')
        return labels

    def affinity_propagation(
            self,
            axis,
            phenotypes=(),
            metric='correlation',
            log_features=False,
            **kwargs):
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
            **kwargs: arguments passed to sklearn.cluster.AffinityPropagation.
        Returns:
            dict with the linkage, distance matrix, and ordering.
        '''
        from scipy.spatial.distance import pdist, squareform
        from sklearn.cluster import AffinityPropagation

        data = self._prepare_data(
            phenotypes=phenotypes,
            log_features=log_features,
            )

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
        # NaN have no affinity
        np.nan_to_num(Y, copy=False)
        model = AffinityPropagation(
                affinity='precomputed',
                **kwargs)
        model.fit(Y)
        labels = pd.Series(model.labels_, index=data.index, dtype='category')
        return labels

    def label_propagation(
            self,
            axis,
            label_column,
            unlabeled_key,
            phenotypes=(),
            method='label_spreading',
            kernel='knn',
            n_neighbors=7,
            log_features=False,
            **kwargs):
        '''Affinity/label/message propagation with some existing labels.

        Args:
            axis (string): It must be 'samples' or 'features'. The
                Dataset.counts matrix is used and either samples or features
                are clustered.
            label_column (string): name of the column in the samplesheet or
                featuresheet with the partial labels.
            unlabeled_key: samples/features with this value in the label_column
                are considered unlabeled.
            phenotypes (iterable of strings): Phenotypes to add to the
                features for joint clustering.
            method (string): 'label_spreading' or 'label_propagation'.
            kernel (string): 'rbf' or 'knn'
            n_neighborts (int): Number of neighbors for the knn kernel
            log_features (bool): Whether to add pseudocounts and take a log
                of the feature counts before calculating distances.
            **kwargs: arguments passed to sklearn.cluster.AffinityPropagation.
        Returns:
            dict with the linkage, distance matrix, and ordering.

        Unlike 'affinity_propagation', this method starts with a column of the
        samplesheet or featuresheet that has labels for part of the dataset and
        subsequently extends the labeling to the rest of the data.
        '''
        from sklearn.semi_supervised import LabelPropagation, LabelSpreading

        data = self._prepare_data(
            phenotypes=phenotypes,
            log_features=log_features,
            )

        if axis == 'samples':
            data = data.T
            labels = self.dataset.samplesheet[label_column]
        elif axis == 'features':
            labels = self.dataset.featuresheet[label_column]
        else:
            raise ValueError('axis must be "samples" or "features"')

        if method == 'label_spreading':
            clustering = LabelSpreading
        elif method == 'label_propagation':
            clustering = LabelPropagation
        else:
            raise ValueError('method must be "label_spreading" or "label_propagation"')

        if axis == 'samples':
            labels = self.dataset.samplesheet[label_column].values
        else:
            labels = self.dataset.featuresheet[label_column].values
        labels_unique = set(labels) - set([unlabeled_key])
        labels_unique = [unlabeled_key] + list(labels_unique)
        labels_unique = pd.Series(
            np.arange(len(labels_unique)) - 1,
            index=labels_unique)
        labels = labels_unique.loc[labels].values

        model = clustering(
                kernel=kernel,
                n_neighbors=n_neighbors,
                **kwargs)
        model.fit(data.values, y=labels)
        labels = pd.Series(
                model.transduction_,
                index=data.index,
                dtype='category')
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
