# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to feature selection
# Modules
import numpy as np
import pandas as pd
import xarray as xr


# Classes / functions
class FeatureSelection():
    '''Plot gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Select features

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    def expressed(
            self,
            n_samples,
            exp_min,
            inplace=False):
        '''Select features that are expressed in at least some samples.

        Args:
            n_samples (int): Minimum number of samples the features should be
                expressed in.
            exp_min (float): Minimum level of expression of the features.
            inplace (bool): Whether to change the feature list in place.

        Returns:
            pd.Index of selected features if not inplace, else None.
        '''
        ind = (self.dataset.counts >= exp_min).sum(axis=1) >= n_samples
        if inplace:
            self.dataset.counts = self.dataset.counts.loc[ind]
        else:
            return self.dataset.featurenames[ind]

    def overdispersed_strata(
            self, bins=10,
            n_features_per_stratum=50,
            inplace=False):
        '''Select overdispersed features in strata of increasing expression.

        Args:
            bins (int or list): Bin edges determining the strata. If this is
                a number, use that number of quantiles.
            n_features_per_stratum (int): Number of features per stratum to
                select.

        Returns:
            pd.Index of selected features if not inplace, else None.

        Notice that the number of selected features may be smaller than
        expected if some strata have no dispersion (e.g. only dropouts).
        Because of this, it is recommended you restrict the counts to
        expressed features before using this function.
        '''

        stats = self.dataset.counts.get_statistics(metrics=('mean', 'cv'))
        mean = stats.loc[:, 'mean']

        if np.isscalar(bins):
            bins = mean.quantile(np.linspace(0, 1, bins+1)).values

        features = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                cvi = stats.loc[mean >= bins[i], 'cv']
            else:
                cvi = stats.loc[(mean >= bins[i]) & (mean < bins[i+1]), 'cv']
            features.append(cvi.nlargest(n_features_per_stratum).index)
        features = pd.Index(np.concatenate(features), name=cvi.index.name)

        if inplace:
            self.dataset.counts = self.dataset.counts.loc[features]
        else:
            return features
