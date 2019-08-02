class ReprWrapper():

    def __init__(self, fn, def_fn):
        self._fn = fn

        if def_fn is None:
            name = 'None'
        elif callable(def_fn):
            name = f'{def_fn.__module__}.{def_fn.__name__}'
        else:
            name = 'Unknown'
        self._name = f'<caching function of "{name}">'

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __repr__(self):
        return self._name
