# vim: fdm=indent
# author:     Fabio Zanini
# date:       09/08/17
# content:    Table of gene counts
# Modules
import pandas as pd


# Classes / functions
class CountsTable(pd.DataFrame):
    '''Table of gene expression counts

    - Rows are features, e.g. genes.
    - Columns are samples.
    '''

    _metadata = ['name', '_spikeins', '_otherfeatures', '_normalized']

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

    def exclude_features(self, spikeins=True, other=True, inplace=False):
        '''Get a slice that excludes secondary features.

        Args:
            spikeins (bool): whether to exclude spike-ins
            other (bool): whether to exclude other features, e.g. unmapped reads

        Returns:
            CountsTable: a slice of self without those features.
        '''
        drop = []
        if spikeins:
            drop.extend(self._spikeins)
        if other:
            drop.extend(self._otherfeatures)
        return self.drop(drop, axis=0, inplace=inplace)

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
            include_spikeins (bool): Whether to include spike-ins in the
            normalization and result.
            inplace (bool): Whether to modify the CountsTable in place or return
            a new one.

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
            self.loc[:, :] = counts_norm.values
            self._normalized = method
        else:
            counts_norm._normalized = method
            return counts_norm
