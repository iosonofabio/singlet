# vim: fdm=indent
# author:     Fabio Zanini
# date:       09/08/17
# content:    Table of gene counts
# Modules
import numpy as np
import pandas as pd


# Classes / functions
class CountsTable(pd.DataFrame):
    '''Table of gene expression counts

    - Rows are features, e.g. genes.
    - Columns are samples.
    '''

    _metadata = [
            'name',
            '_spikeins', '_otherfeatures',
            '_normalized', 'pseudocount',
            ]

    pseudocount = 0.1

    @property
    def _constructor(self):
        return CountsTable

    @classmethod
    def from_tablename(cls, tablename):
        '''Instantiate a CountsTable from its name in the config file.

        Args:
            tablename (string): name of the counts table in the config file.

        Returns:
            CountsTable: the counts table.
        '''
        from .config import config
        from .io.csv import parse_counts_table

        self = cls(parse_counts_table(tablename))
        self.name = tablename
        self._spikeins = config['io']['count_tables'][tablename]['spikeins']
        self._otherfeatures = config['io']['count_tables'][tablename]['other']
        self._normalized = config['io']['count_tables'][tablename]['normalized']

        return self

    def exclude_features(self, spikeins=True, other=True, inplace=False,
                         errors='raise'):
        '''Get a slice that excludes secondary features.

        Args:
            spikeins (bool): Whether to exclude spike-ins
            other (bool): Whether to exclude other features, e.g. unmapped reads
            inplace (bool): Whether to drop those features in place.
            errors (string): Whether to raise an exception if the features \
                    to be excluded are already not present.

        Returns:
            CountsTable: a slice of self without those features.
        '''
        drop = []
        if spikeins:
            drop.extend(self._spikeins)
        if other:
            drop.extend(self._otherfeatures)
        return self.drop(drop, axis=0, inplace=inplace, errors=errors)

    def get_spikeins(self):
        '''Get spike-in features

        Returns:
            CountsTable: a slice of self with only spike-ins.
        '''
        return self.loc[self._spikeins]

    def get_other_features(self):
        '''Get other features

        Returns:
            CountsTable: a slice of self with only other features (e.g. unmapped).
        '''
        return self.loc[self._otherfeatures]

    def normalize(self, method='counts_per_million', include_spikeins=False, inplace=False, **kwargs):
        '''Normalize counts and return new CountsTable.

        Args:
            method (string or function): The method to use for normalization. \
                    One of 'counts_per_million', \
                    'counts_per_thousand_spikeins', \
                    'counts_per_thousand_features'. If this argument is a \
                    function, it must take the CountsTable as input and \
                    return the normalized one as output.
            include_spikeins (bool): Whether to include spike-ins in the \
                    normalization and result.
            inplace (bool): Whether to modify the CountsTable in place or \
                    return a new one.

        Returns:
            If `inplace` is False, a new, normalized CountsTable.
        '''
        if self._normalized:
            raise ValueError('CountsTable is already normalized')

        if method == 'counts_per_million':
            counts = self.exclude_features(spikeins=(not include_spikeins), other=True)
            counts_norm = 1e6 * counts / counts.sum(axis=0)
        elif method == 'counts_per_thousand_spikeins':
            counts = self.exclude_features(spikeins=(not include_spikeins), other=True)
            norm = self.get_spikeins().sum(axis=0)
            counts_norm = 1e3 * counts / norm
        elif method == 'counts_per_thousand_features':
            if 'features' not in kwargs:
                raise ValueError('Set features=<list of normalization features>')
            counts = self.exclude_features(spikeins=(not include_spikeins), other=True)
            norm = self.loc[kwargs['features']].sum(axis=0)
            counts_norm = 1e3 * counts / norm
        elif callable(method):
            counts_norm = method(self)
            method = 'custom'
        else:
            raise ValueError('Method not understood')

        if inplace:
            # The new CountsTable lacks other features and maybe spikeins
            drop = np.setdiff1d(self.index, counts_norm.index)
            self.drop(drop, inplace=True)
            self.loc[:, :] = counts_norm
            self._normalized = method
        else:
            counts_norm._normalized = method
            return counts_norm

    def get_statistics(self, metrics=('mean', 'cv')):
        '''Get statistics of the counts.

        Args:
            metrics (sequence of strings): any of 'mean', 'var', 'std', 'cv', \
                    'fano', 'min', 'max'.

        Returns:
            pandas.DataFrame with features as rows and metrics as columns.
        '''
        st = {}
        if 'mean' in metrics or 'cv' in metrics or 'fano' in metrics:
            st['mean'] = self.mean(axis=1)
        if ('std' in metrics or 'cv' in metrics or 'fano' in metrics or
           'var' in metrics):
            st['std'] = self.std(axis=1)
        if 'var' in metrics:
            st['var'] = st['std'] ** 2
        if 'cv' in metrics:
            st['cv'] = st['std'] / np.maximum(st['mean'], 1e-10)
        if 'fano' in metrics:
            st['fano'] = st['std'] ** 2 / np.maximum(st['mean'], 1e-10)
        if 'min' in metrics:
            st['min'] = self.min(axis=1)
        if 'max' in metrics:
            st['max'] = self.max(axis=1)

        df = pd.concat([st[m] for m in metrics], axis=1)
        df.columns = pd.Index(list(metrics), name='metrics')
        return df
