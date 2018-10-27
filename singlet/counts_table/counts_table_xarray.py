# vim: fdm=indent
'''
author:     Fabio Zanini
date:       26/10/18
content:    Table of feature counts, using xarrays and possibly dask arrays.

NOTE: We use composition over inheritance, but it's a discrete pain.
'''
import numpy as np
import xarray as xr


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

    def copy(self):
        c = self.__class__(self._data.copy())
        for key in self._metadata:
            setattr(c, key, getattr(self, key))
        return c

    def __str__(self):
        return self._data.__str__().replace(
            'xarray.DataArray',
            'singlet.CountsTableXR')

    def __repr__(self):
        return self._data.__repr__().replace(
            'xarray.DataArray',
            'singlet.CountsTableXR')

    def __iadd__(self, b):
        self._data.__iadd__(b)

    def __isub__(self, b):
        self._data.__isub__(b)

    def __imul__(self, b):
        self._data.__imul__(self, b)

    def __ifloordiv__(self, b):
        self._data.__ifloordiv__(b)

    def __imod__(self, b):
        self._data.__imod__(b)

    def __iand__(self, b):
        self._data.__iand__(b)

    def __ior__(self, b):
        self._data.__ior__(b)

    def __ixor__(self, b):
        self._data.__ixor__(b)

    def __invert__(self):
        c = self.__class__(self._data.__invert__())
        for key in self._metadata:
            setattr(c, key, getattr(self, key))
        return c

