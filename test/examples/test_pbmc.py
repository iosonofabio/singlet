#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test examples on PBMCs.
'''
import os
import sys
import numpy as np
import pytest


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(counts_table='example_PBMC')


def test_example(ds):
    if os.getenv('CONTINUOUS_INTEGRATION') == 'true':
        import matplotlib
        matplotlib.use('agg')
    import matplotlib.pyplot as plt

    print('Normalize')
    ds.counts.normalize(method='counts_per_million', inplace=True)
    ds.counts.log(inplace=True)

    print('Feature selection')
    ds.feature_selection.expressed(n_samples=3, exp_min=1, inplace=True)
    ds.feature_selection.overdispersed_strata(
            n_features_per_stratum=20,
            inplace=True)

    print('Dimensionality reduction')
    vs = ds.dimensionality.tsne(
            n_dims=2,
            theta=0.5,
            perplexity=0.8)

    dsr = ds.copy()
    dsr.counts = vs.T

    print('Cluster')
    dsr.samplesheet['dbscan'] = dsr.cluster.dbscan(eps=5, axis='samples')
    dsr.samplesheet['kmeans'] = dsr.cluster.kmeans(n_clusters=7, axis='samples')
    dsr.samplesheet['affinity'] = dsr.cluster.affinity_propagation(axis='samples', metric='correlation')

    print('Plot tSNE')
    fig, axs = plt.subplots(
            nrows=1, ncols=3, sharex=True, sharey=True,
            figsize=(8, 3))
    dsr.plot.scatter_reduced_samples(vs, color_by='dbscan', ax=axs[0], zorder=10)
    dsr.plot.scatter_reduced_samples(vs, color_by='kmeans', ax=axs[1], zorder=10)
    dsr.plot.scatter_reduced_samples(vs, color_by='affinity', ax=axs[2], zorder=10)

    axs[0].set_title('DBSCAN')
    axs[1].set_title('K-means, 7 clusters')
    axs[2].set_title('Affinity propagation')

    plt.tight_layout()

    ds.samplesheet['cluster'] = dsr.samplesheet['kmeans']
    ds_dict = ds.split(phenotypes=['cluster'])

    genes_by_cluster = {}
    for key, dsi in ds_dict.items():
        dso = ds.query_samples_by_metadata('cluster!=@key', local_dict=locals())
        genes_by_cluster[key] = dsi.compare(dso)['P-value'].nsmallest(10).index
    # FIXME: assertion is wrong??
    #assert(
    #    genes_by_cluster[1][:3].tolist() == ['ENSG00000138085', 'ENSG00000184076', 'ENSG00000116459'])

    if os.getenv('CONTINUOUS_INTEGRATION') == 'true':
        # FIXME: save to file and compare with preexisting PNG
        #fig.savefig('example_data/test_example_pbmc_tsnes.png', dpi=300)
        plt.close(fig)
    else:
        plt.show()


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    ds = ds()
    test_example(ds)
