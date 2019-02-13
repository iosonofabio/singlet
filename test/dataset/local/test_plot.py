#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import matplotlib.pyplot as plt
import numpy as np


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')

    # FIXME: move down
    #print('Plot clustermap')
    #ds.counts.normalize('counts_per_million', inplace=True)
    #ds.counts = ds.counts.iloc[:200]
    #vs = ds.plot.clustermap(
    #        subtract_mean=True,
    #        divide_std=True)
    #plt.show()
    #print('Done!')

    print('Plot coverage')
    ax = ds.plot.plot_coverage(color='blue', lw=3)
    ax = ds.plot.plot_coverage(
            features='other', color='red', linewidth=1,
            ax=ax)
    plt.show()
    print('Done!')

    print('Plot spike-in distributions')
    ax = ds.plot.plot_distributions(
            kind='swarm',
            features='spikeins',
            orientation='horizontal',
            sort='descending')
    plt.show()
    print('Done!')

    print('Plot normalized distributions of housekeeping genes')
    ds.counts.normalize('counts_per_million', inplace=True)
    ax = ds.plot.plot_distributions(
            kind='swarm',
            features=['ACTB', 'TUBB1', 'GAPDH'],
            orientation='vertical',
            bottom='pseudocount',
            grid=True,
            sort='descending')
    plt.show()
    print('Done!')

    print('Plot PCA')
    vs = ds.dimensionality.pca(
            n_dims=2,
            transform='log10',
            robust=False)['vs']
    ax = ds.plot.scatter_reduced_samples(
            vs,
            color_by='ACTB')
    plt.show()
    print('Done!')

    print('Plot t-SNE')
    ds.counts = ds.counts.iloc[:200]
    vs = ds.dimensionality.tsne(
            n_dims=2,
            transform='log10',
            theta=0.5,
            perplexity=0.8)
    ax = ds.plot.scatter_reduced_samples(
            vs,
            color_by='quantitative_phenotype_1_[A.U.]')
    plt.show()
    print('Done!')

    print('Gate features')
    selected = ds.feature_selection.gate_features_from_statistics(color='blue', lw=3)
    print(selected)
    print('Done!')
