# vim: fdm=indent
'''
author:     Fabio Zanini
date:       26/10/18
content:    Table of feature counts, using xarrays and possibly dask arrays.

NOTE: We use composition over inheritance, but it's a discrete pain.
'''
import numpy as np
import pandas as pd
import xarray as xr


def unwrap_data1(fun):
    def _unwrapped(self, other, *args, **kwargs):
        if isinstance(other, CountsTableXR):
            return fun(self, other._data, *args, **kwargs)
        else:
            return fun(self, other, *args, **kwargs)
    return _unwrapped


class CountsTableXR(object):
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

    def __init__(self, data):
        self._data = xr.DataArray(data)

    def _set_metadata_from_other(self, other, exclude=('dataset',)):
        for key in self._metadata:
            if key not in exclude:
                setattr(self, key, getattr(other, key))

    def copy(self, deep=True):
        c = self.__class__(self._data.copy(deep=deep))
        c._set_metadata_from_other(self)
        return c

    def __str__(self):
        return self._data.__str__().replace(
            'xarray.DataArray',
            'singlet.CountsTableXR')

    def __repr__(self):
        return self._data.__repr__().replace(
            'xarray.DataArray',
            'singlet.CountsTableXR')

    def __abs__(self):
        c = self.__class__(self._data.__abs__())
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __add__(self, b):
        c = self.__class__(self._data.__add__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __and__(self, b):
        c = self.__class__(self._data.__and__(b))
        c._set_metadata_from_other(self)
        return c

    def __copy__(self):
        c = self.__class__(self._data.__copy__())
        c._set_metadata_from_other(self)
        return c

    def __deepcopy__(self, memo=None):
        c = self.__class__(self._data.__deepcopy__(memo=memo))
        c._set_metadata_from_other(self)
        return c

    def __delitem__(self, key):
        return self._data.__delitem__(key)

    @unwrap_data1
    def __eq__(self, other):
        c = self.__class__(self._data.__eq__(other))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __ge__(self, b):
        c = self.__class__(self._data.__ge__(b))
        c._set_metadata_from_other(self)
        return c

    def __getitem__(self, key):
        c = self.__class__(self._data.__getitem__(key))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __gt__(self, b):
        c = self.__class__(self._data.__gt__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __iadd__(self, b):
        self._data.__iadd__(b)
        return self

    @unwrap_data1
    def __iand__(self, b):
        self._data.__iand__(b)
        return self

    @unwrap_data1
    def __ifloordiv__(self, b):
        self._data.__ifloordiv__(b)
        return self

    @unwrap_data1
    def __imod__(self, b):
        self._data.__imod__(b)
        return self

    @unwrap_data1
    def __imul__(self, b):
        self._data.__imul__(b)
        return self

    def __invert__(self):
        c = self.__class__(self._data.__invert__())
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __ior__(self, b):
        self._data.__ior__(b)
        return self

    @unwrap_data1
    def __ipow__(self, b):
        self._data.__ipow__(b)
        return self

    @unwrap_data1
    def __isub__(self, b):
        self._data.__isub__(b)
        return self

    @unwrap_data1
    def __itruediv__(self, b):
        self._data.__itruediv__(b)
        return self

    @unwrap_data1
    def __ixor__(self, b):
        self._data.__ixor__(b)
        return self

    @unwrap_data1
    def __le__(self, b):
        c = self.__class__(self._data.__le__(b))
        c._set_metadata_from_other(self)
        return c

    def __len__(self):
        return self._data.__len__()

    @unwrap_data1
    def __lt__(self, b):
        c = self.__class__(self._data.__lt__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __mod__(self, b):
        c = self.__class__(self._data.__mod__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __mul__(self, b):
        c = self.__class__(self._data.__mul__(b))
        c._set_metadata_from_other(self)
        return c

    def __neg__(self):
        c = self.__class__(self._data.__neg__())
        c._set_metadata_from_other(self)
        return c

    def __nonzero__(self):
        return self._data.__nonzero__()

    @unwrap_data1
    def __or__(self, b):
        c = self.__class__(self._data.__or__(b))
        c._set_metadata_from_other(self)
        return c

    def __pos__(self):
        c = self.__class__(self._data.__pos__())
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __radd__(self, b):
        c = self.__class__(self._data.__radd__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rand__(self, b):
        c = self.__class__(self._data.__rand__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rfloordiv__(self, b):
        c = self.__class__(self._data.__rfloordiv__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rmod__(self, b):
        c = self.__class__(self._data.__rmod__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rmul__(self, b):
        c = self.__class__(self._data.__rmul__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __ror__(self, b):
        c = self.__class__(self._data.__ror__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rpow__(self, b):
        c = self.__class__(self._data.__rpow__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rsub__(self, b):
        c = self.__class__(self._data.__rsub__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rtruediv__(self, b):
        c = self.__class__(self._data.__rtruediv__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __rxor__(self, b):
        c = self.__class__(self._data.__rxor__(b))
        c._set_metadata_from_other(self)
        return c

    def __setitem__(self):
        #FIXME
        pass

    @unwrap_data1
    def __sub__(self, b):
        c = self.__class__(self._data.__sub__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __truediv__(self, b):
        c = self.__class__(self._data.__truediv__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __floordiv__(self, b):
        c = self.__class__(self._data.__floordiv__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __xor__(self, b):
        c = self.__class__(self._data.__xor__(b))
        c._set_metadata_from_other(self)
        return c

    @property
    def T(self):
        self._data = self._data.T
        return self

    def all(self, *args, **kwargs):
        return self._data.all(*args, **kwargs)

    def any(self, *args, **kwargs):
        return self._data.any(*args, **kwargs)

    def argmax(self, *args, **kwargs):
        return self._data.argmax(*args, **kwargs)

    def argmin(self, *args, **kwargs):
        return self._data.argmin(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return self._data.argsort(*args, **kwargs)

    def astype(self, *args, **kwargs):
        c = self.__class__(self._data.astype(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def chunk(self, *args, **kwargs):
        c = self.__class__(self._data.chunk(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @property
    def chunks(self):
        return self._data.chunks

    def clip(self, *args, **kwargs):
        c = self.__class__(self._data.clip(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def close(self):
        self._data.close()

    @unwrap_data1
    def combine_first(self, other):
        c = self.__class__(self._data.combine_first(other))
        c._set_metadata_from_other(self)
        return c

    def compute(self, **kwargs):
        c = self.__class__(self._data.compute(**kwargs))
        c._set_metadata_from_other(self)
        return c

    @property
    def coords(self):
        return self._data.coords

    def count(self, *args, **kwargs):
        return self._data.count(*args, **kwargs)

    def cumprod(self, *args, **kwargs):
        c = self.__class__(self._data.cumprod(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def cumsum(self, *args, **kwargs):
        c = self.__class__(self._data.cumsum(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @property
    def data(self):
        return self._data.data

    def diff(self, *args, **kwargs):
        c = self.__class__(self._data.diff(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @property
    def dims(self):
        return self._data.dims

    @unwrap_data1
    def dot(self, other, dims=None):
        return self._data.dot(other, dims=dims)

    def drop(self, *args, **kwargs):
        c = self.__class__(self._data.drop(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def dropna(self, *args, **kwargs):
        c = self.__class__(self._data.dropna(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def encoding(self):
        return self._data.encoding

    def equals(self, other):
        '''True if two DataArrays have the same dimensions, coordinates,
           values, and _metadata; otherwise False.
        '''
        if not isinstance(other, self.__class__):
            return False
        if not self._data.equals(other._data):
            return False
        for key in self._metadata:
            if getattr(self, key) != getattr(other, key):
                return False
        return True

    def expand_dims(self, *args, **kwargs):
        c = self.__class__(self._data.expand_dims(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def fillna(self, value):
        c = self.__class__(self._data.fillna(value))
        c._set_metadata_from_other(self)
        return c

    def get_axis_num(self, dim):
        return self._data.get_axis_num(dim)

    def get_index(self, key):
        return self._data.get_index(key)

    def groupby(self, group, squeeze=True):
        return self._data.groupby(group, squeeze=squeeze)

    def groupby_bins(self, *args, **kwargs):
        return self._data.groupby_bins(*args, **kwargs)

    def identical(self, other):
        '''True if two DataArrays have the same dimensions, coordinates,
           values, array name, attributes, attributes on all coordinates,
           and _metadata; otherwise False.
        '''
        if not isinstance(other, self.__class__):
            return False
        if not self._data.identical(other._data):
            return False
        for key in self._metadata:
            if getattr(self, key) != getattr(other, key):
                return False
        return True

    @property
    def indexes(self):
        self._data.indexes

    def interp(self, *args, **kwargs):
        c = self.__class__(self._data.interp(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def interp_like(other, self, *args, **kwargs):
        c = self.__class__(self._data.interp_like(other, *args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def interpolate_na(self, *args, **kwargs):
        c = self.__class__(self._data.interpolate_na(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def isel(self, *args, **kwargs):
        c = self.__class__(self._data.isel(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def isel_points(self, *args, **kwargs):
        c = self.__class__(self._data.isel_points(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def isin(self, test_elements):
        c = self.__class__(self._data.isin(test_elements))
        c._set_metadata_from_other(self)
        return c

    def load(self, **kwargs):
        c = self.__class__(self._data.load(**kwargs))
        c._set_metadata_from_other(self)
        return c

    @property
    def loc(self):
        return self._data.loc

    def max(self, *args, **kwargs):
        return self._data.max(*args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._data.mean(*args, **kwargs)

    def median(self, *args, **kwargs):
        return self._data.median(*args, **kwargs)

    def min(self, *args, **kwargs):
        return self._data.min(*args, **kwargs)

    @property
    def name(self):
        self._data.name

    @name.setter
    def name(self, name):
        self._data.name = name

    def persist(self, **kwargs):
        self._data.persist(**kwargs)

    @property
    def plot(self):
        return self._data.plot

    def prod(self, *args, **kwargs):
        return self._data.prod(*args, **kwargs)

    def quantile(self, *args, **kwargs):
        return self._data.quantile(*args, **kwargs)

    def rank(self, *args, **kwargs):
        c = self.__class__(self._data.rank(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def reduce(self, *args, **kwargs):
        return self._data.reduce(*args, **kwargs)

    def reindex(self, *args, **kwargs):
        c = self.__class__(self._data.reindex(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def reindex(self, other, *args, **kwargs):
        c = self.__class__(self._data.reindex(other, *args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def rename(self, *args, **kwargs):
        c = self.__class__(self._data.rename(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def reorder_levels(self, inplace=False, **dim_order):
        c = self.__class__(self._data.reorder_level(inplace=inplace, **dim_order))
        if not inplace:
            c._set_metadata_from_other(self)
            return c

    def roll(self, **shifts):
        c = self.__class__(self._data.roll(**shifts))
        c._set_metadata_from_other(self)
        return c

    def round(self, *args, **kwargs):
        c = self.__class__(self._data.round(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def sel(self, *args, **kwargs):
        c = self.__class__(self._data.sel(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def sel_points(self, *args, **kwargs):
        c = self.__class__(self._data.sel_points(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def set_index(self, append=False, inplace=False, **indexes):
        c = self.__class__(self._data.reorder_level(
            append=append, inplace=inplace, **indexes))
        if not inplace:
            c._set_metadata_from_other(self)
            return c

    @property
    def shape(self):
        return self._data.shape

    def shift(self, **shifts):
        c = self.__class__(self._data.shift(**shifts))
        c._set_metadata_from_other(self)
        return c

    @property
    def size(self):
        return self._data.size

    @property
    def sizes(self):
        return self._data.sizes

    def sortby(self, *args, **kwargs):
        c = self.__class__(self._data.sortby(*args, **kwargs))
        c._set_metadata_from_other(self)
        return c

    def squeeze(self, *args, **kwargs):
        return self._data.squeeze(*args, **kwargs)

    def stack(self, **dimensions):
        return self._data.stack(**dimensions)

    def std(self, *args, **kwargs):
        return self._data.std(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return self._data.sum(*args, **kwargs)

    def swap_dims(self, dims_dict):
        c = self.__class__(self._data.swap_dims(dims_dict))
        c._set_metadata_from_other(self)
        return c

    def transpose(self, *dims):
        c = self.__class__(self._data.transpose(*dims))
        c._set_metadata_from_other(self)
        return c

    def unstack(self, dim):
        return self._data.unstack(dim)

    @property
    def values(self):
        return self._data.values

    def var(self, *args, **kwargs):
        return self._data.var(*args, **kwargs)

    def where(self, *args, **kwargs):
        return self._data.where(*args, **kwargs)

    @classmethod
    def from_tablename(cls, tablename):
        '''Instantiate a CountsTable from its name in the config file.

        Args:
            tablename (string): name of the counts table in the config file.

        Returns:
            CountsTableXR: the counts table.
        '''
        from ..config import config
        from ..io import parse_counts_table

        # Initially parsed as a dataframe (xarray does not parse csv directly)
        self = cls(xr.DataArray(parse_counts_table({'countsname': tablename})))
        if self._data.dims[1] == 'dim_1':
            self._data = self._data.rename({'dim_1': 'sample name'})

        self.name = tablename
        config_table = config['io']['count_tables'][tablename]
        self._spikeins = config_table.get('spikeins', [])
        self._otherfeatures = config_table.get('other', [])
        self._normalized = config_table['normalized']
        return self

    # FIXME
    @classmethod
    def from_datasetname(cls, datasetname):
        '''Instantiate a CountsTable from its name in the config file.

        Args:
            datasetname (string): name of the dataset in the config file.

        Returns:
            CountsTable: the counts table.
        '''
        from ..config import config
        from ..io import parse_counts_table

        # TODO: support lazy evaluation
        self = cls(parse_counts_table({'datasetname': datasetname}))
        if self._data.dims[1] == 'dim_1':
            self._data = self._data.rename({'dim_1': 'sample name'})

        self.name = datasetname
        config_table = config['io']['datasets'][datasetname]['counts_table']
        self._spikeins = config_table.get('spikeins', [])
        self._otherfeatures = config_table.get('other', [])
        self._normalized = config_table['normalized']
        return self

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
        clog = np.log(self.pseudocount + self._data) / np.log(base)
        if inplace:
            self._data = clog
        else:
            clog = self.__class__(clog)
            clog._set_metadata_from_other(self)
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
        cunlog = base**self._data - self.pseudocount
        if inplace:
            self = cunlog
        else:
            cunlog = self.__class__(cunlog)
            cunlog._set_metadata_from_other(self)
            return cunlog

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
            st['mean'] = self._data.mean(axis=1)
        if ('std' in metrics or 'cv' in metrics or 'fano' in metrics or
           'var' in metrics):
            st['std'] = self._data.std(axis=1)
        if 'var' in metrics:
            st['var'] = st['std'] ** 2
        if 'cv' in metrics:
            st['cv'] = st['std'] / np.maximum(st['mean'], 1e-10)
        if 'fano' in metrics:
            st['fano'] = st['std'] ** 2 / np.maximum(st['mean'], 1e-10)
        if 'min' in metrics:
            st['min'] = self._data.min(axis=1)
        if 'max' in metrics:
            st['max'] = self._data.max(axis=1)

        df = pd.concat([st[m].to_series() for m in metrics], axis=1)
        df.columns = pd.Index(list(metrics), name='metrics')
        return df

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

        drop_bool = [True if x not in drop else False for x in self._data.get_index(self._data.dims[0])]

        if not inplace:
            c = self.__class__(self._data[drop_bool])
            c._set_metadata_from_other(self)
            return c

        self._data = self._data[drop_bool]
        if self.dataset is not None:
            self.dataset._featuresheet.drop(drop, inplace=True, errors=errors)

    def normalize(
            self,
            method='counts_per_million',
            include_spikeins=False,
            inplace=False,
            **kwargs):
        '''Normalize counts and return new CountsTableXR.

        Args:
            method (string or function): The method to use for normalization.
            One of 'counts_per_million', 'counts_per_thousand_spikeins',
            'counts_per_thousand_features'. If this argument is a function, its
            signature depends on the inplace argument. If inplace=False, it
            must take the CountsTable as input and return the normalized one as
            output. If inplace=True, it must take the CountsTableXR as input
            and modify it in place. Notice that if inplace=True and you do
            non-inplace operations you might lose the _metadata properties. You
            can end your function by self[:] = <normalized counts>.
            include_spikeins (bool): Whether to include spike-ins in the
            normalization and result.
            inplace (bool): Whether to modify the CountsTableXR in place or
            return a new one.

        Returns:
            If `inplace` is False, a new, normalized CountsTableXR.
        '''
        import copy

        if self._normalized:
            raise ValueError('CountsTableXR is already normalized')

        if inplace:
            if method == 'counts_per_million':
                self.exclude_features(
                        spikeins=(not include_spikeins),
                        other=True,
                        inplace=True)
                self._data *= 1e6 / self._data.sum(axis=0)
            elif method == 'counts_per_thousand_spikeins':
                spikeins = self.get_spikeins().sum(axis=0)
                self.exclude_features(
                        spikeins=(not include_spikeins),
                        other=True,
                        inplace=True)
                self._data *= 1e3 / spikeins
            elif method == 'counts_per_thousand_features':
                if 'features' not in kwargs:
                    raise ValueError('Set features=<list of normalization features>')
                features = self.loc[kwargs['features']].sum(axis=0)
                self.exclude_features(
                        spikeins=(not include_spikeins),
                        other=True,
                        inplace=True)
                self._data *= 1e3 / features
            elif callable(method):
                method(self)
                method = 'custom'
            else:
                raise ValueError('Method not understood')

            self._normalized = method

        else:
            if method == 'counts_per_million':
                counts = self.exclude_features(spikeins=(not include_spikeins), other=True)._data
                norm = counts.sum(axis=0)
                counts_norm = 1e6 * counts / norm
            elif method == 'counts_per_thousand_spikeins':
                counts = self.exclude_features(spikeins=(not include_spikeins), other=True)._data
                norm = self.get_spikeins()._data.sum(axis=0)
                counts_norm = 1e3 * counts / norm
            elif method == 'counts_per_thousand_features':
                if 'features' not in kwargs:
                    raise ValueError('Set features=<list of normalization features>')
                counts = self.exclude_features(spikeins=(not include_spikeins), other=True)._data
                norm = self.loc[kwargs['features']]._data.sum(axis=0)
                counts_norm = 1e3 * counts / norm
            elif callable(method):
                counts_norm = method(self)
                method = 'custom'
            else:
                raise ValueError('Method not understood')

            c = self.__class__(counts_norm)
            c._set_metadata_from_other(self)
            c._normalized = method
            return c

    def center(self, axis='samples', inplace=False):
        '''Center the counts table (subtract mean).

        Args:
            axis (string): The axis to average over, has to be 'samples' \
                    or 'features'.
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTableXR

        Returns:
            If inplace is False, a transformed CountsTableXR.
        '''
        if axis == 'samples':
            mean = self.mean(axis=1)
        elif axis == 'features':
            mean = self.mean(axis=0)
        else:
            raise ValueError('Axis not found')

        if inplace:
            self -= mean
        else:
            return self - mean

    def z_score(self, axis='samples', inplace=False, add_to_den=0):
        '''Calculate the z scores of the counts table.

        In other words, subtract the mean and divide by the standard \
                deviation.

        Args:
            axis (string): The axis to average over, has to be 'samples' \
                    or 'features'.
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTableXR
            add_to_den (float): Whether to add a (small) value to the \
                    denominator to avoid NaNs. 1e-5 or so should be fine.

        Returns:
            If inplace is False, a transformed CountsTableXR.
        '''
        if axis == 'samples':
            mean = self.mean(axis=1)
            den = add_to_den + self.std(axis=1)
        elif axis == 'features':
            mean = self.mean(axis=0)
            den = add_to_den + self.std(axis=0)
        else:
            raise ValueError('Axis not found')

        if inplace:
            self -= mean
            self /= den
        else:
            return (self - mean) / den

    def standard_scale(self, axis='samples', inplace=False, add_to_den=0):
        '''Subtract minimum and divide by (maximum - minimum).

        Args:
            axis (string): The axis to average over, has to be 'samples' \
                    or 'features'.
            inplace (bool): Whether to do the operation in place or return \
                    a new CountsTableXR
            add_to_den (float): Whether to add a (small) value to the \
                    denominator to avoid NaNs. 1e-5 or so should be fine.

        Returns:
            If inplace is False, a transformed CountsTableXR.
        '''
        if axis == 'samples':
            mi = self.min(axis=1)
            ma = self.max(axis=1)
        elif axis == 'features':
            mi = self.min(axis=0)
            ma = self.max(axis=0)
        else:
            raise ValueError('Axis not found')

        den = (add_to_den + ma - mi)

        if inplace:
            self -= mi
            self /= den
        else:
            return (self - mi) / den

    def bin(self, bins=5, result='index', inplace=False):
        '''Bin feature counts.

        Args:
            bins (int, array, or list of arrays): If an int, number
                equal-width bins between pseudocounts and the max of
                the counts matrix. If an array of indices of the same
                length as the number of features, use a different number
                of equal-width bins for each feature. If an array of any
                other length, use these bin edges (including rightmost
                edge) for all features. If a list of arrays, it has to be
                as long as the number of features, and every array in the
                list determines the bin edges (including rightmost edge)
                for that feature, in order.
            result (string): Has to be one of 'index' (default), 'left',
                'center', 'right'. 'index' assign to the feature the
                index (starting at 0) of that bin, 'left' assign the left
                bin edge, 'center' the bin center, 'right' the right
                edge. If result is 'index', out-of-bounds values will be
                assigned the value -1, which means Not A Number in ths
                context.
            inplace (bool): Whether to perform the operation in place.

        Returns:
            If inplace is False, a CountsTableXR with the binned counts.
        '''
        # Prepare bin edges: a list of lists
        nf = self.shape[0]
        if np.isscalar(bins):
            bins = np.linspace(self.pseudocount, self._data.data.max(), bins + 1)
            bins = np.repeat(bins, nf).reshape((len(bins), nf)).T
        elif len(bins) == nf:
            bins_new = []
            for (key, c), nbin in zip(self.iterrows(), bins):
                bins_new.append(np.linspace(self.pseudocount, c.max(), nbin + 1))
            bins = bins_new
        elif np.isscalar(bins[0]):
            bins = [bins for i in range(nf)]

        # Prepare output data structure
        if result == 'index':
            out_dtype = int
        else:
            out_dtype = float
        out = xr.zeros_like(self._data, dtype=out_dtype)

        # Bin data
        for i, bini in enumerate(bins):
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
                    np.maximum(self.pseudocount, self._data.data[i]),
                    bini,
                    labels=labels,
                    right=True,
                    include_lowest=True)

            if result == 'index':
                cbin = cbin.astype(int)
                cbin[np.isnan(cbin)] = -1
            else:
                cbin = cbin.astype(out.dtype)

            out.data[i] = cbin

        if inplace:
            self._data = out
        else:
            c = self.__class__(out)
            c._set_metadata_from_other(self)
            return c
