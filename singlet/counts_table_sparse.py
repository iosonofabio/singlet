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
        config_table = config['io']['count_tables'][tablename]
        self._spikeins = config_table.get('spikeins', [])
        self._otherfeatures = config_table.get('other', [])
        self._normalized = config_table['normalized']
        return self

    def to_npz(self, filename):
        '''Save to numpy compressed file format'''
        from .io.npz import to_counts_table_sparse

        to_counts_table_sparse(self, filename)

    def exclude_features(self, spikeins=True, other=True, inplace=False,
                         errors='raise'):
        '''Get a slice that excludes secondary features.

        Args:
            spikeins (bool): Whether to exclude spike-ins
            other (bool): Whether to exclude other features, e.g. unmapped reads
            inplace (bool): Whether to drop those features in place.
            errors (string): Whether to raise an exception if the features
                to be excluded are already not present. Must be 'ignore'
                or 'raise'.

        Returns:
            CountsTable: a slice of self without those features.
        '''
        drop = []
        if spikeins:
            drop.extend(self._spikeins)
        if other:
            drop.extend(self._otherfeatures)
        out = self.drop(drop, axis=0, inplace=inplace, errors=errors)
        if inplace and (self.dataset is not None):
            self.dataset._featuresheet.drop(drop, inplace=True, errors=errors)
        return out

    def get_spikeins(self):
        '''Get spike-in features

        Returns:
            CountsTable: a slice of self with only spike-ins.
        '''
        return self.loc[self._spikeins]

    def get_other_features(self):
        '''Get other features

        Returns:
            CountsTable: a slice of self with only other features (e.g.
                unmapped).
        '''
        return self.loc[self._otherfeatures]

    def log(self, base=10):
        '''Take the pseudocounted log of the counts.

        Args:
            base (float): Base of the log transform

        Returns:
            A transformed CountsTableSparse with zeros at the zero-count items.
        '''
        from scipy.sparse import coo_matrix

        coo = self.to_coo()
        coobase = np.log(self.pseudocount) * coo_matrix((np.ones(coo.nnz), (coo.row, coo.col)), shape=coo.shape)
        coolog = ((coo / self.pseudocount).log1p() + coobase) / np.log(base)
        # NOTE: the entries that should be log(pseudocount) are zeros now

        clog = CountsTableSparse(
            coolog,
            index=self.index,
            columns=self.columns,
            dtype=float,
            default_fill_value=0)

        return clog

    def unlog(self, base=10):
        '''Reverse the pseudocounted log of the counts.

        Args:
            base (float): Base of the log transform

        Returns:
            A transformed CountsTableSparse.
        '''
        from scipy.sparse import coo_matrix

        coo = self.to_coo()

        coobase = np.log(self.pseudocount) * coo_matrix((np.ones(coo.nnz), (coo.row, coo.col)), shape=coo.shape)
        cooexp = (coo * np.log(base) - coobase).expm1() * self.pseudocount

        cexp = CountsTableSparse(
            cooexp,
            index=self.index,
            columns=self.columns,
            dtype=float,
            default_fill_value=0)

        return cexp
