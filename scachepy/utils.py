import functools
import types

class Wrapper():

    def __init__(self, wrapped, def_fn=None, assigned=functools.WRAPPER_ASSIGNMENTS):
        if callable(def_fn):
            for attr in assigned:
                setattr(self, attr, getattr(def_fn, attr))
            name = f'{def_fn.__module__}.{def_fn.__name__}'
        elif def_fn is None:
            name = 'None'
        else:
            raise ValueError(f'Expected default_function to be either callable or NoneType, got: `type(def_fn)`.')

        self._name = f'<caching function of "{name}">'
        super().__init__()

    def __get__(self, obj, objtype):
        return types.MethodType(self.__call__, obj)

    def __repr__(self):
        return self._name
