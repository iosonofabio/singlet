# vim: fdm=indent
# author:     Fabio Zanini
# date:       21/08/17
# content:    Caching utils.
# Modules
import inspect
from functools import wraps


# Classes / functions
def method_caches(f):
    @wraps(f)
    def _wrapped(self, *args, **kwargs):
        fargs = inspect.getargvalues(inspect.currentframe()).locals['kwargs']
        cachename = '_'+f.__name__+'_cache'

        # Check cache
        if hasattr(self, cachename):
            cache = getattr(self, cachename)
            if cache['func_args'] == fargs:
                return cache['cache']

        res = f(self, *args, **kwargs)

        # Cache results
        cache = {'cache': res,
                 'func_args': dict(fargs)}
        setattr(self, cachename, cache)

        return res
    return _wrapped
