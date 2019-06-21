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
            pd.Series with the labels of the clusters.
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
            **kwargs: arguments passed to sklearn.cluster.LabelSpreading or
                sklearn.cluster.LabelPropagation.

        Returns:
            pd.Series with the labels of the clusters.

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
                labels_unique.index[model.transduction_ + 1],
                index=data.index,
                dtype='category')
        return labels

    def random_forest(
            self,
            axis,
            label_column,
            unlabeled_key,
            phenotypes=(),
            n_estimators=100,
            log_features=False,
            return_model=False,
            **kwargs):
        '''Random Forest Classifier learned from some existing labels.

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
            n_estimators (int): Number of trees in the forest.
            log_features (bool): Whether to add pseudocounts and take a log
                of the feature counts before calculating distances.
            return_model (bool): Whether to also return the trained model
            **kwargs: arguments passed to
                sklearn.ensemble.RandomForestClassifier.

        Returns:
            pd.Series with the labels of the clusters. If return_model is True,
            return a pair (labels, model).
        '''
        from sklearn.ensemble import RandomForestClassifier

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
        index = data.index
        data = data.values

        # Split into train/test data
        ind_train = labels != -1
        data_train = data[ind_train]
        labels_train = labels[ind_train]
        data_test = data[~ind_train]

        # Train model
        model = RandomForestClassifier(
                n_estimators=n_estimators,
                **kwargs)
        model.fit(data_train, y=labels_train)

        # Predict missing labels
        labels_test = model.predict(data_test)
        labels[~ind_train] = labels_test

        labels = pd.Series(
                labels_unique.index[labels + 1],
                index=index,
                dtype='category')

        if return_model:
            return (labels, model)
        else:
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

    def leiden(
            self,
            axis,
            edges,
            edge_weights=None,
            metric='cpm',
            resolution_parameter=0.001,
            ):
        '''Graph-based Leiden clustering

        Args:
            axis (string): It must be 'samples' or 'features'.
                The Dataset.counts matrix is used and
                either samples or features are clustered.
            edges (list of pairs): list of edges to make a graph used to
            cluster. Each member of a pair is an int referring to the index
            of the sample or feature in the sample/featuresheet.
            edge_weights (list of float or None): edge weights to use for
            clustering. If None, all edge weights are 1.
            metric (str): What metric to optimize. Can be 'modularity' or
            'cpm'.
            resolution_parameter (float): A number between 0 and 1 that sets
            how easy it is to call new clusters.

        Returns:
            pd.Series with the labels of the clusters.
        '''
        import igraph as ig
        import leidenalg

        if axis == 'samples':
            n_nodes = self.dataset.n_samples
            index = self.dataset.samplenames
        elif axis == 'features':
            n_nodes = self.dataset.n_features
            index = self.dataset.featurenames

        g = ig.Graph(n=n_nodes, edges=edges, directed=False)
        if edge_weights is not None:
            g.es['weight'] = edge_weights

        if metric == 'cpm':
            partition = leidenalg.CPMVertexPartition(
                    g,
                    resolution_parameter=resolution_parameter)
        elif metric == 'modularity':
            partition = leidenalg.ModularityVertexPartition(
                    g,
                    resolution_parameter=resolution_parameter)
        else:
            raise ValueError(
                'clustering_metric not understood: {:}'.format(metric))

        opt = leidenalg.Optimiser()
        opt.optimise_partition(partition)
        communities = partition.membership

        labels = pd.Series(communities, index=index)

        return labels
