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

    _metadata = ['sheet', 'name']

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
        self.sheet = config['io']['count_tables'][tablename]

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
            drop.extend(self.sheet['spikeins'])
        if other:
            drop.extend(self.sheet['other'])
        return self.drop(drop, axis=0, inplace=inplace)

    def get_spikeins(self):
        '''Get spike-in features

        Returns:
            CountsTable: a slice of self with only spike-ins.
        '''
        return self.loc[self.sheet['spikeins']]

    def get_other_features(self):
        '''Get other features

        Returns:
            CountsTable: a slice of self with only other features (e.g. unmapped).
        '''
        return self.loc[self.sheet['other']]
