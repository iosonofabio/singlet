# vim: fdm=indent
# author:     Fabio Zanini
# date:       09/08/17
# content:    Sparse table of gene counts
# Modules
import numpy as np
import pandas as pd


# Classes / functions
class CountsTableSparse(pd.SparseDataFrame):
    '''Sparse table of gene expression counts

    - Rows are features, e.g. genes.
    - Columns are samples.
    '''

    _metadata = [
            'name',
            '_spikeins',
            '_otherfeatures',
            '_normalized',
            'pseudocount',
            'dataset',
            ]

    _spikeins = ()
    _otherfeatures = ()
    _normalized = False
    pseudocount = 0.1
    dataset = None

    @property
    def _constructor(self):
        return CountsTableSparse


    @classmethod
    def from_tablename(cls, tablename):
        '''Instantiate a CountsTable from its name in the config file.

        Args:
            tablename (string): name of the counts table in the config file.

        Returns:
            CountsTable: the counts table.
        '''
        from .config import config
        from .io import parse_counts_table_sparse

        self = cls(parse_counts_table_sparse(tablename))
        self.name = tablename
        self._spikeins = config['io']['count_tables'][tablename].get('spikeins', [])
        self._otherfeatures = config['io']['count_tables'][tablename].get('other', [])
        self._normalized = config['io']['count_tables'][tablename]['normalized']

        return self
