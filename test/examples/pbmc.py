#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test examples on PBMCs.
'''
import sys
import matplotlib.pyplot as plt
import numpy as np


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(counts_table='example_PBMC')

    # Normalize
    ds.counts.normalize(method='counts_per_million', inplace=True)
    ds.counts.log(inplace=True)

    # Select features
    ds.feature_selection.expressed(n_samples=3, exp_min=1, inplace=True)
    ds.feature_selection.overdispersed_strata(
            n_features_per_stratum=20,
            inplace=True)

    # Reduce dimensionality
    vs = ds.dimensionality.tsne(
            n_dims=2,
            theta=0.5,
            perplexity=0.8)

    dsr = ds.copy()
    dsr.counts = vs.T

    # Cluster
    dsr.samplesheet['dbscan'] = dsr.cluster.dbscan(eps=5, axis='samples')
    dsr.samplesheet['kmeans'] = dsr.cluster.kmeans(n_clusters=7, axis='samples')

    # Plot t-SNE
    fig, axs = plt.subplots(
            nrows=1, ncols=2, sharex=True, sharey=True,
            figsize=(8, 4))
    dsr.plot.scatter_reduced_samples(vs, color_by='dbscan', ax=axs[0], zorder=10)
    dsr.plot.scatter_reduced_samples(vs, color_by='kmeans', ax=axs[1], zorder=10)

    axs[0].set_title('DBSCAN')
    axs[1].set_title('K-means, 7 clusters')

    plt.tight_layout()

    plt.show()
