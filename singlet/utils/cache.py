# vim: fdm=indent
# author:     Fabio Zanini
# date:       21/08/17
# content:    Caching utils.
# Modules
import inspect
from functools import wraps


# Classes / functions
def method_caches(f):
    '''Decorator for instance methods that cache results.

    Args:
        f (function): The function to decorate. Note that f must have only \
                one positional argument, self, referring to the instance \
                calling the method. All other arguments must be keyword-only \
                arguments. This is a limitation of the current implementation.

    Returns:
        The wrapped function, including functools.wraps - so that the name \
                and docstring of the original functions should be mimicked.
    '''
    @wraps(f)
    def _wrapped(self, *args, **kwargs):
        fargs = inspect.getargvalues(inspect.currentframe()).locals['kwargs']
        cachename = '_'+f.__name__+'_cache'

        # Check cache
        if hasattr(self, cachename):
            cache = getattr(self, cachename)
            if cache['func_kwargs'] == fargs:
                return cache['cache']

        res = f(self, *args, **kwargs)

        # Cache results
        cache = {'cache': res,
                 'func_kwargs': dict(fargs)}
        setattr(self, cachename, cache)

        return res
    return _wrapped
