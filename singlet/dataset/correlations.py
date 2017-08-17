# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to correlate gene expression and phenotypes
# Modules
import numpy as np
import pandas as pd
from scipy.stats import rankdata


# Classes / functions
class Correlation():
    '''Correlate gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Correlate gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    @staticmethod
    def _correlate(x, y, method):
        if method == 'pearson':
            xw = x
            yw = y
        elif method == 'spearman':
            xw = np.zeros_like(x, float)
            for ii, xi in enumerate(x):
                xw[ii] = rankdata(xi, method='average')
            yw = np.zeros_like(y, float)
            for ii, yi in enumerate(y):
                yw[ii] = rankdata(yi, method='average')
        else:
            raise ValueError('correlation method not understood')

        xw = ((xw.T - xw.mean(axis=1)) / xw.std(axis=1)).T
        yw = ((yw.T - yw.mean(axis=1)) / yw.std(axis=1)).T
        n = xw.shape[1]
        r = np.dot(xw, yw.T) / n
        return r

    def correlate_features_phenotypes(
            self,
            phenotypes,
            features='all',
            method='spearman',
            fillna=None,
            ):
        '''Correlate feature expression with one or more phenotypes.

        Args:
            phenotypes (list of string): list of phenotypes, i.e. columns of \
                    the samplesheet. Use a string for a single phenotype.
            features (list or string): list of features to correlate. Use a \
                    string for a single feature. The special string 'all' \
                    (default) uses all features.
            method (string): type of correlation. Must be one of 'pearson' or \
                    'spearman'.
            fillna (dict, int, or None): a dictionary with phenotypes as keys \
                    and numbers to fill for NaNs as values. None will do \
                    nothing, potentially yielding NaN as correlation \
                    coefficients.

        Returns:
            pandas.DataFrame with the correlation coefficients. If either \
                    phenotypes or features is a single string, the function \
                    returns a pandas.Series. If both are a string, it returns \
                    a single correlation coefficient.
        '''

        exp = self.dataset.counts
        if features != 'all':
            if isinstance(features, str):
                exp = exp.loc[[features]]
            else:
                exp = exp.loc[features]

        phe = self.dataset.samplesheet
        if isinstance(phenotypes, str):
            phe = phe.loc[:, [phenotypes]]
        else:
            phe = phe.loc[:, phenotypes]

        if fillna is not None:
            phe = phe.copy()
            if np.isscalar(fillna):
                phe.fillna(fillna, inplace=True)
            else:
                for key, fna in fillna.items():
                    phe.loc[:, key].fillna(fna, inplace=True)

        x = exp.values
        y = phe.values.T

        r = self._correlate(x, y, method=method)

        if (not isinstance(features, str)) and (not isinstance(phenotypes, str)):
            return pd.DataFrame(
                    data=r,
                    index=exp.index,
                    columns=phe.columns,
                    dtype=float)
        elif isinstance(features, str) and (not isinstance(phenotypes, str)):
            return pd.Series(
                    data=r[0],
                    index=phe.columns,
                    dtype=float)
        elif (not isinstance(features, str)) and isinstance(phenotypes, str):
            return pd.Series(
                    data=r[:, 0],
                    index=exp.index,
                    dtype=float)
        else:
            return r[0, 0]

    def correlate_features_features(
            self,
            features='all',
            features2=None,
            method='spearman'):
        '''Correlate feature expression with one or more phenotypes.

        Args:
            features (list or string): list of features to correlate. Use a \
                    string for a single feature. The special string 'all' \
                    (default) uses all features.
            features (list or string): list of features to correlate with. \
                    Use a string for a single feature. The special string \
                    'all' uses all features. None (default) takes the same \
                    list as features, returning a square matrix.
            method (string): type of correlation. Must be one of 'pearson' or \
                    'spearman'.

        Returns:
            pandas.DataFrame with the correlation coefficients. If either \
                    features or features2 is a single string, the function \
                    returns a pandas.Series. If both are a string, it returns \
                    a single correlation coefficient.
        '''
        exp_all = self.dataset.counts
        if features == 'all':
            exp = exp_all
        elif isinstance(features, str):
            exp = exp_all.loc[[features]]
        else:
            exp = exp_all.loc[features]

        if features2 is None:
            exp2 = exp
        elif features2 == 'all':
            exp2 = exp_all
        elif isinstance(features2, str):
            exp2 = exp_all.loc[[features2]]
        else:
            exp2 = exp_all.loc[features2]

        x = exp.values
        y = exp2.values

        r = self._correlate(x, y, method=method)

        if (not isinstance(features, str)) and (not isinstance(features2, str)):
            return pd.DataFrame(
                    data=r,
                    index=exp.index,
                    columns=exp2.index,
                    dtype=float)
        elif isinstance(features, str) and (not isinstance(features2, str)):
            return pd.Series(
                    data=r[0],
                    index=exp2.index,
                    dtype=float)
        elif (not isinstance(features, str)) and isinstance(features2, str):
            return pd.Series(
                    data=r[:, 0],
                    index=exp.index,
                    dtype=float)
        else:
            return r[0, 0]

    def correlate_phenotypes_phenotypes(
            self,
            phenotypes,
            phenotypes2=None,
            method='spearman',
            fillna=None,
            fillna2=None,
            ):
        '''Correlate feature expression with one or more phenotypes.

        Args:
            phenotypes (list of string): list of phenotypes, i.e. columns of \
                    the samplesheet. Use a string for a single phenotype.
            phenotypes2 (list of string): list of phenotypes, i.e. columns of \
                    the samplesheet. Use a string for a single phenotype. \
                    None (default) uses the same as phenotypes.
            method (string): type of correlation. Must be one of 'pearson' or \
                    'spearman'.
            fillna (dict, int, or None): a dictionary with phenotypes as keys \
                    and numbers to fill for NaNs as values. None will do \
                    nothing, potentially yielding NaN as correlation \
                    coefficients.
            fillna2 (dict, int, or None): as fillna, but for phenotypes2.

        Returns:
            pandas.DataFrame with the correlation coefficients. If either \
                    phenotypes or features is a single string, the function \
                    returns a pandas.Series. If both are a string, it returns \
                    a single correlation coefficient.
        '''

        phe_all = self.dataset.samplesheet
        if isinstance(phenotypes, str):
            phe = phe_all.loc[:, [phenotypes]]
        else:
            phe = phe_all.loc[:, phenotypes]

        if phenotypes2 is None:
            phe2 = phe.copy()
        elif isinstance(phenotypes2, str):
            phe2 = phe_all.loc[:, [phenotypes2]]
        else:
            phe2 = phe_all.loc[:, phenotypes2]

        if fillna is not None:
            phe = phe.copy()
            if np.isscalar(fillna):
                phe.fillna(fillna, inplace=True)
            else:
                for key, fna in fillna.items():
                    phe.loc[:, key].fillna(fna, inplace=True)

        if fillna2 is not None:
            phe2 = phe2.copy()
            if np.isscalar(fillna2):
                phe2.fillna(fillna2, inplace=True)
            else:
                for key, fna in fillna2.items():
                    phe2.loc[:, key].fillna(fna, inplace=True)

        x = phe.values.T
        y = phe2.values.T

        r = self._correlate(x, y, method=method)

        if (not isinstance(phenotypes, str)) and (not isinstance(phenotypes2, str)):
            return pd.DataFrame(
                    data=r,
                    index=phe.columns,
                    columns=phe2.columns,
                    dtype=float)
        elif isinstance(phenotypes, str) and (not isinstance(phenotypes2, str)):
            return pd.Series(
                    data=r[0],
                    index=phe2.columns,
                    dtype=float)
        elif (not isinstance(phenotypes, str)) and isinstance(phenotypes2, str):
            return pd.Series(
                    data=r[:, 0],
                    index=phe.columns,
                    dtype=float)
        else:
            return r[0, 0]
