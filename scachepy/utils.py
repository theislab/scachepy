from PIL import Image
from inspect import signature
from abc import ABC

import scanpy as sc
import numpy as np
import anndata
import functools
import warnings
import types
import os


UNS_PLOT_KEY = 'scachepy_plot'
_caching_fn_doc = '''

    Caching function arguments.

    Arguments
    ---------
    fname: Str, optional (default: `None`)
        filename under the cache directory where to load/save the results
        if `None`, use `default_fname` for the given function
    force: Bool, optional (default: `False`)
        whether to force the computation even if the cache exists
        if `True`, will override `call=False`
    call: Bool, optional (default: `True`)
        whether to call the callback prior to caching
    skip: Bool, optional (default: `False`)
        whether to skip mandatory keys which are not found
    verbose: Bool, optional (default: `True`)
        whether to print additional information
'''


def wrap_as_adata(fn, *, ret_attr):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            adata = args[0] if isinstance(args[0], np.ndarray) else kwargs.get('adata')
        else:
            adata = kwargs.get('adata')

        assert isinstance(adata, np.ndarray), f'Expected `{adata}` to be of type `np.ndarray`.'
        # wrap the np.ndarray in AnnData
        adata = anndata.AnnData(adata)

        if isinstance(args[0], np.ndarray):
            # can't assing to tuple
            args = list(args)
            args[0] = adata
            args = tuple(args)
        else:
            kwargs['adata'] = adata

        res = fn(*args, **kwargs)

        # if copy was specified, operate on that object
        # other use the original (inplace modification)
        res = res if res is not None else adata

        out = []
        # currently, only 1 key is supported
        for attr, k in ret_attr.items():
            out.append(getattr(res, attr)[k])

        if len(ret_attr) == 1:
            return out[0]

        return tuple(out)

    return wrapper


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


def plotting_wrapper(fn):

    @functools.wraps(fn)
    def wrapper(adata, *args, **kwargs):
        if UNS_PLOT_KEY in adata.uns:
            return

        verbosity = sc.settings.verbosity
        sc.settings.verbosity = 0  # don't want any warnings

        sig = signature(fn).parameters.keys()
        # return_fig = kwargs.pop('return_fig', None)

        # TODO:
        # this does not properly load legends
        # leave the code here for future reference

        # if 'return_fig' in sig and False:
            # these are not nicely aligned (as it is the case with 'save'),
            # but better than to writing to disk
            # fig = fn(adata, *args, **kwargs, return_fig=True)
            # adata.uns[UNS_PLOT_KEY] = fig2data(fig)

            # sc.settings.verbosity = verbosity

        # correct, but less efficient than above
        if 'save' in sig:
            if kwargs.pop('save', None) is not None:
                warnings.warn(f'Ignoring option `save=\'{save}\'`.')

            key = str(np.random.randint(1_000_000))
            fn(adata, *args, **kwargs, save=f'{key}.png')

            assert os.path.isdir(sc.settings.figdir), f'No directory found under `sc.settings.figdir=\'{sc.settings.figdir}\'`.'
            possible_fnames = [f for f in os.listdir(sc.settings.figdir) if key in f]

            # restoring verbosity level
            sc.settings.verbosity = verbosity

            if not len(possible_fnames):
                raise RuntimeError(f'Unable to find the saved figure. This shouldn\'t have happened.')
            elif len(possible_fnames) > 1:
                raise RuntimeError('Found ambiguous matches for the figure. This shouldn\'t have happened.')

            fpath = os.path.join(sc.settings.figdir, possible_fnames[0])
            adata.uns[UNS_PLOT_KEY] = np.array(Image.open(fpath))
            os.remove(fpath)  # cleanup

        else:
            raise RuntimeError(f'Plotting function `{fn.__name__}` has no argument `save`.')

    # TODO: this was called for return_fig
    # see above TODO
    def fig2data(fig):
        from matplotlib.backends.backend_tkagg import FigureCanvasAgg

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        data, (width, height) = canvas.print_to_buffer()

        return np.fromstring(data, np.uint8).reshape((height, width, 4))

    return wrapper
