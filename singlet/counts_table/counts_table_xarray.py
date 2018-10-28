# vim: fdm=indent
'''
author:     Fabio Zanini
date:       26/10/18
content:    Table of feature counts, using xarrays and possibly dask arrays.

NOTE: We use composition over inheritance, but it's a discrete pain.
'''
import numpy as np
import xarray as xr


def unwrap_data1(fun):
    def _unwrapped(b, *args, **kwargs):
        if isinstance(b, CountsTableXR):
            return fun(b._data, *args, **kwargs)
        else:
            return fun(b, *args, **kwargs)
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

    def _set_metadata_from_other(self, other):
        for key in self._metadata:
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
        self._data.__delitem__(key)

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

    def __getitem__(key):
        #FIXME
        pass

    @unwrap_data1
    def __gt__(self, b):
        c = self.__class__(self._data.__gt__(b))
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __iadd__(self, b):
        self._data.__iadd__(b)

    @unwrap_data1
    def __iand__(self, b):
        self._data.__iand__(b)

    @unwrap_data1
    def __ifloordiv__(self, b):
        self._data.__ifloordiv__(b)

    @unwrap_data1
    def __imod__(self, b):
        self._data.__imod__(b)

    @unwrap_data1
    def __imul__(self, b):
        self._data.__imul__(self, b)

    def __invert__(self):
        c = self.__class__(self._data.__invert__())
        c._set_metadata_from_other(self)
        return c

    @unwrap_data1
    def __ior__(self, b):
        self._data.__ior__(b)

    @unwrap_data1
    def __ipow__(self, b):
        self._data.__ipow__(b)

    @unwrap_data1
    def __isub__(self, b):
        self._data.__isub__(b)

    @unwrap_data1
    def __itruediv__(self, b):
        self._data.__itruediv__(b)

    @unwrap_data1
    def __ixor__(self, b):
        self._data.__ixor__(b)

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
