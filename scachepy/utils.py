import functools
import types


_caching_fn_doc = '''
    
    Caching function arguments.

    Arguments
    ---------
    fname: Str, optional (default: `None`)
        filename under the cache directory where to load/save the results
        if `None`, use `default_fname`
    force: Bool, optional (default: `False`)
        whether to force the computation even if the cache exists
        if `True`, will override when `call=False`
    call: Bool, optional (default: `True`)
        whether to call the callback prior to actual caching
    skip: Bool, optional (default: `False`)
        whether to skip keys which are not found during caching
    verbose: Bool, optional (default: `True`)
        whether to print additional information
'''


# to simulate scanpy's .pp, .tl, .pl
class Module:

    def __init__(self, typp, **kwargs):
        self._fun_names = tuple(kwargs.keys())
        self._typp = typp
        for k, fn in kwargs.items():
            setattr(self, k, fn)

    def __iter__(self):  # to list available functions
        return iter(self._fun_names)

    def __repr__(self):
        return f'<{self.__module__}.{self._typp}>'


class FunctionWrapper():

    def __init__(self, wrapped, def_fn=None, assigned=functools.WRAPPER_ASSIGNMENTS):
        assert callable(wrapped), f'Function must be callable, but is of type `{type(wrapped)}`.'
        if callable(def_fn):
            for attr in assigned:
                if attr == '__module__':  # to have auto-complete
                    continue
                if attr == '__doc__':
                    old_doc = getattr(def_fn, attr)
                    setattr(self, attr, ('' if old_doc is None else old_doc) + _caching_fn_doc)
                    continue
                setattr(self, attr, getattr(def_fn, attr))
            name = f'{def_fn.__module__}.{def_fn.__name__}'
        elif def_fn is None:
            name = 'None'
        else:
            raise ValueError(f'Expected default_function to be either callable or NoneType, got: `type(def_fn)`.')

        self._fn = wrapped
        self._name = f'<caching function of "{name}">'

        super().__init__()

    def __get__(self, obj, objtype):
        return types.MethodType(self.__call__, obj)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __repr__(self):
        return self._name
