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
        from .io import parse_counts_table

        self = cls(parse_counts_table(tablename))
        self.name = tablename
        config_table = config['io']['count_tables'][tablename]
        self._spikeins = config_table.get('spikeins', [])
        self._otherfeatures = config_table.get('other', [])
        self._normalized = config_table['normalized']
        return self

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
            CountsTable: a slice of self with only other features (e.g. unmapped).
        '''
        return self.loc[self._otherfeatures]

    def log(self, base=10, inplace=False):
        '''Take the pseudocounted log of the counts.

        Args:
            base (float): Base of the log transform
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTable

        Returns:
            If inplace is False, a transformed CountsTable.
        '''
        clog = np.log(self.pseudocount + self) / np.log(base)
        if inplace:
            self[:] = clog.values
        else:
            return clog

    def unlog(self, base=10, inplace=False):
        '''Reverse the pseudocounted log of the counts.

        Args:
            base (float): Base of the log transform
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTable

        Returns:
            If inplace is False, a transformed CountsTable.
        '''
        cunlog = base**self - self.pseudocount
        if inplace:
            self[:] = cunlog.values
        else:
            return cunlog

    def center(self, axis='samples', inplace=False):
        '''Center the counts table (subtract mean).

        Args:
            axis (string): The axis to average over, has to be 'samples' \
                    or 'features'.
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTable

        Returns:
            If inplace is False, a transformed CountsTable.
        '''
        if inplace:
            out = self
        else:
            out = self.copy()

        if axis == 'samples':
            out.loc[:] = (self.values.T - self.values.mean(axis=1)).T
        elif axis == 'features':
            out.loc[:] = self.values - self.values.mean(axis=0)
        else:
            raise ValueError('Axis not found')

        if not inplace:
            return out

    def z_score(self, axis='samples', inplace=False, add_to_den=0):
        '''Calculate the z scores of the counts table.

        In other words, subtract the mean and divide by the standard \
                deviation.

        Args:
            axis (string): The axis to average over, has to be 'samples' \
                    or 'features'.
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTable
            add_to_den (float): Whether to add a (small) value to the \
                    denominator to avoid NaNs. 1e-5 or so should be fine.

        Returns:
            If inplace is False, a transformed CountsTable.
        '''
        if inplace:
            out = self
        else:
            out = self.copy()

        if axis == 'samples':
            out.loc[:] = ((self.values.T - self.values.mean(axis=1)) / (add_to_den + self.values.std(axis=1))).T
        elif axis == 'features':
            out.loc[:] = (self.values - self.values.mean(axis=0)) / (add_to_den + self.values.std(axis=0))
        else:
            raise ValueError('Axis not found')

        if not inplace:
            return out

    def standard_scale(self, axis='samples', inplace=False, add_to_den=0):
        '''Subtract minimum and divide by (maximum - minimum).

        Args:
            axis (string): The axis to average over, has to be 'samples' \
                    or 'features'.
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTable
            add_to_den (float): Whether to add a (small) value to the \
                    denominator to avoid NaNs. 1e-5 or so should be fine.

        Returns:
            If inplace is False, a transformed CountsTable.
        '''
        if inplace:
            out = self
        else:
            out = self.copy()

        if axis == 'samples':
            mi = self.values.min(axis=1)
            ma = self.values.max(axis=1)
            out.loc[:] = ((self.values.T - mi) / (add_to_den + ma - mi)).T
        elif axis == 'features':
            mi = self.values.min(axis=0)
            ma = self.values.max(axis=0)
            out.loc[:] = (self.values - mi) / (add_to_den + ma - mi)
        else:
            raise ValueError('Axis not found')

        if not inplace:
            return out

    def normalize(
            self,
            method='counts_per_million',
            include_spikeins=False,
            inplace=False,
            **kwargs):
        '''Normalize counts and return new CountsTable.

        Args:
            method (string or function): The method to use for normalization. \
                    One of 'counts_per_million', \
                    'counts_per_thousand_spikeins', \
                    'counts_per_thousand_features'. If this argument is a \
                    function, its signature depends on the inplace argument. \
                    If inplace=False, it must take the CountsTable as input \
                    and return the normalized one as output. If inplace=True, \
                    it must take the CountsTable as input and modify it in \
                    place. Notice that if inplace=True and you do non-inplace \
                    operations you might lose the _metadata properties. You \
                    can end your function by self[:] = <normalized counts>.
            include_spikeins (bool): Whether to include spike-ins in the \
                    normalization and result.
            inplace (bool): Whether to modify the CountsTable in place or \
                    return a new one.

        Returns:
            If `inplace` is False, a new, normalized CountsTable.
        '''
        import copy

        if self._normalized:
            raise ValueError('CountsTable is already normalized')

        if inplace:
            if method == 'counts_per_million':
                self.exclude_features(
                        spikeins=(not include_spikeins),
                        other=True,
                        inplace=True)
                self[:] *= 1e6 / self.sum(axis=0)
            elif method == 'counts_per_thousand_spikeins':
                spikeins = self.get_spikeins().sum(axis=0)
                self.exclude_features(
                        spikeins=(not include_spikeins),
                        other=True,
                        inplace=True)
                self[:] *= 1e3 / spikeins
            elif method == 'counts_per_thousand_features':
                if 'features' not in kwargs:
                    raise ValueError('Set features=<list of normalization features>')
                features = self.loc[kwargs['features']].sum(axis=0)
                self.exclude_features(
                        spikeins=(not include_spikeins),
                        other=True,
                        inplace=True)
                self[:] *= 1e3 / features
            elif callable(method):
                method(self)
                method = 'custom'
            else:
                raise ValueError('Method not understood')

            self._normalized = method

        else:
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

    def bin(self, bins=5, result='index', inplace=False):
        '''Bin feature counts.

        Args:
            bins (int, array, or list of arrays): If an int, number \
                    equal-width bins between pseudocounts and the max of \
                    the counts matrix. If an array of indices of the same \
                    length as the number of feature, use a different number \
                    of equal-width bins for each feature. If an array of any \
                    other length, use these bin edges (including rightmost \
                    edge) for all features. If a list of arrays, it has to be \
                    as long as the number of features, and every array in the \
                    list determines the bin edges (including rightmost edge) \
                    for that feature, in order.
            result (string): Has to be one of 'index' (default), 'left', \
                    'center', 'right'. 'index' assign to the feature the \
                    index (starting at 0) of that bin, 'left' assign the left \
                    bin edge, 'center' the bin center, 'right' the right \
                    edge. If result is 'index', out-of-bounds values will be \
                    assigned the value -1, which means Not A Number in ths \
                    context.
            inplace (bool): Whether to perform the operation in place.

        Returns:
            If inplace is False, a CountsTable with the binned counts.
        '''
        if result == 'index':
            out_dtype = int
        else:
            out_dtype = float

        out = np.zeros_like(self.values, dtype=out_dtype)

        nf = self.shape[0]
        if np.isscalar(bins):
            bins = np.linspace(self.pseudocount, self.values.max(), bins + 1)
            bins = np.repeat(bins, nf).reshape((len(bins), nf)).T
        elif len(bins) == nf:
            bins_new = []
            for (key, c), nbin in zip(self.iterrows(), bins):
                bins_new.append(np.linspace(self.pseudocount, c.max(), nbin + 1))
            bins = bins_new

        for i, (bini, (fea, count)) in enumerate(zip(bins, self.iterrows())):
            if result == 'index':
                labels = False
            elif result == 'left':
                labels = bini[:-1]
            elif result == 'right':
                labels = bini[1:]
            elif result == 'center':
                labels = 0.5 * (bini[1:] + bini[:-1])
            else:
                raise ValueError('result parameter not understood')

            cbin = pd.cut(
                    np.maximum(self.pseudocount, count), bini,
                    labels=labels,
                    right=True,
                    include_lowest=True)

            if result == 'index':
                cbin[np.isnan(cbin)] = -1

            out[i] = cbin.values

        if inplace:
            self.loc[:] = out
        else:
            counts = self.copy()
            counts.loc[:] = out
            return out
