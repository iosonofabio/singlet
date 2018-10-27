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
        from ..config import config
        from ..io import parse_counts_table_sparse

        self = cls(parse_counts_table_sparse({'countsname': tablename}))
        self.name = tablename
        config_table = config['io']['count_tables'][tablename]
        self._spikeins = config_table.get('spikeins', [])
        self._otherfeatures = config_table.get('other', [])
        self._normalized = config_table['normalized']
        return self

    @classmethod
    def from_datasetname(cls, datasetname):
        '''Instantiate a CountsTable from its name in the config file.

        Args:
            datasetename (string): name of the dataset in the config file.

        Returns:
            CountsTableSparse: the counts table.
        '''
        from ..config import config
        from ..io import parse_counts_table_sparse

        self = cls(parse_counts_table_sparse({'datasetname': datasetname}))
        self.name = datasetname
        config_table = config['io']['datasets'][datasetname]['counts_table']
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

    def normalize(
            self,
            method='counts_per_million',
            include_spikeins=False,
            **kwargs):
        '''Normalize counts and return new CountsTable.

        Args:
            method (string or function): The method to use for normalization.
                One of 'counts_per_million', 'counts_per_thousand_spikeins',
                'counts_per_thousand_features'. If this argument is a
                function, its signature depends on the inplace argument.
                It must take the CountsTable as input and return the normalized
                one as output. You can end your function by
                self[:] = <normalized counts>.
            include_spikeins (bool): Whether to include spike-ins in the
                normalization and result.
            inplace (bool): Whether to modify the CountsTable in place or
                return a new one.

        Returns:
            A new, normalized CountsTableSparse.
        '''
        import copy

        if method == 'counts_per_million':
            counts = self.exclude_features(spikeins=(not include_spikeins), other=True)
            norm = counts.sum(axis=0)
            counts_norm = 1e6 * counts / norm
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

        # Shallow copy of metadata
        for prop in self._metadata:
            # dataset if special, to avoid infinite loops
            if prop == 'dataset':
                counts_norm.dataset = None
            else:
                setattr(counts_norm, prop, copy.copy(getattr(self, prop)))
        counts_norm._normalized = method
        return counts_norm

    def get_statistics(self, metrics=('mean', 'cv')):
        '''Get statistics of the counts.

        Args:
            metrics (sequence of strings): any of 'mean', 'var', 'std', 'cv',
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
