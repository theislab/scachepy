from .backends import Backend, PickleBackend
from .utils import *
from collections import Iterable 
from abc import ABC
from inspect import signature
from PIL import Image

import scvelo as scv
import scanpy as sc
import anndata

import numpy as np
import matplotlib as mpl
import re
import traceback
import warnings


class Module(ABC):

    _backends = dict(pickle=PickleBackend)
    _extensions = dict(pickle='.pickle')

    def __init__(self, backend, **kwargs):
        if isinstance(backend, Backend):
            self._backend = backend
        else:
            assert isinstance(backend, str)
            backend_type = self._backends.get(backend, None)
            if backend_type is None:
                raise ValueError(f'Unknown backend type: `{backend}`.'
                                  ' Supported backends are: `{", ".join(self._backends.keys())}`.')
            if 'ext' not in kwargs:
                kwargs['ext'] = self._extensions.get(backend, '.scdata')
            self._backend = backend_type(**kwargs)

        for k, fn in self._functions.items():
            setattr(self, k, fn)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, _):
        raise RuntimeError('Setting backend is disallowed. To change the directory, try `dir` attribute of `backend.`')

    def _clear(self, verbose=1, *, separator=None):
        self.backend._clear(verbose, self._type, separator=separator)

    def clear(self, verbose=1):
        self._clear(verbose)

    def __iter__(self):  # to list available functions
        return iter(self._functions.keys())

    def __repr__(self):
        return f'<{self.__module__}.{self._type}>'

    def _create_cache_fn(self, *args, default_fname=None):

        def wrapper(adata, fname=None, recache=False, verbose=True, skip=False, *args, **kwargs):
            try:
                if fname is None:
                    fname = default_fname
                if not fname.endswith(self.backend.ext):
                    fname += self.backend.ext

                if recache:
                    possible_vals = set(args) | set(kwargs.values())
                    return self.backend.save(adata, fname, attrs, keys,
                                             skip=skip, is_optional=is_optional,
                                             possible_vals=possible_vals, verbose=verbose)

                if (self.backend.dir / fname).is_file():
                    if verbose:
                        print(f'Loading data from: `{fname}`.')

                    return self.backend.load(adata, fname, verbose=verbose, skip=skip)

                return False

            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    if recache:
                        print(traceback.format_exc())
                else:
                    print(f'No cache found in `{self.backend.dir / fname}`.')

                return False

        # if you're here because of the doc, you're here correctly
        if len(args) == 1:
            collection = args[0]
            if isinstance(collection, dict):
                attrs = tuple(collection.keys())
                keys = tuple(collection.values())
            elif isinstance(collection, Iterable) and len(next(iter(collection))) == 2:
                attrs, keys = tuple(zip(*collection))
            else:
                raise RuntimeError('Unable to decode the args of length 1.')
        elif len(args) == 2:
            attrs, keys = args
            if isinstance(attrs, str):
                attrs = (attrs, )
            if isinstance(keys, str):
                keys = (keys, )
        else:
            raise RuntimeError('Expected the args to be of length 1 or 2.')

        pat = re.compile(r'.*_opt')
        is_optional = tuple(pat.match(a) is not None for a in attrs)

        # strip the postfix
        pat = re.compile(r'(:?_opt)|(?:_cache\d+)')
        attrs = tuple(pat.sub('', a) for a in attrs)

        return wrapper

    def cache(self, *args, **kwargs):
        '''
        Create a caching function.

        Params
        --------
        args: Dict[Str, Union[Str, Iterable[Union[Str, re._pattern_type]]]]
            attributes are supplied as dictionary keys and
            values as dictionary values (need not be an `Iterable`)
            for caching multiple attributes of the same name,
            append to them postfixes of the following kind: `_cache1, _cache2, ...`
            there are also other ways of specifying this, please
            refer the source code of `_create_cache_fn`
        default_fname: Str
            default filename where to save the data
        default_fn: Callable, optional (default: `None`)
            function to call before caching the values

        Returns
        --------
        caching_function: Callable
            caching function accepting as the first argument either
            `anndata.AnnData` object or a `callable` and `anndata.AnnData`
            object as the second argument
            the `callable` either needs to return an `anndata.AnnData` object
            (if `copy=True`) or just modify it inplace
        '''

        def wrapper(*args, **kwargs):
            fname = kwargs.pop('fname', None)
            force = kwargs.pop('force', False)
            verbose = kwargs.pop('verbose', True)
            call = kwargs.pop('call', True)  # if we do not wish to call the callback
            skip = kwargs.pop('skip', False)
            # leave it in kwargs
            copy = kwargs.get('copy', False) and not is_plot

            assert fname is not None or def_fname is not None, f'No filename or default specified.'

            callback = None
            if len(args) > 1 and callable(args[0]):
                callback, *args = args

            is_raw = False
            if len(args) > 0 and isinstance(args[0], (anndata.AnnData, anndata.Raw)):
                if isinstance(args[0], anndata.Raw):
                    # get rid of the raw type
                    args = (args[0]._adata, *args[1:])
                    is_raw = True
                adata = args[0]
            elif 'adata' in kwargs:
                if isinstance(kwargs['adata'], anndata.Raw):
                    # get rid of the raw type
                    kwargs['adata'] = kwargs['adata']._adata
                    is_raw = True
                adata = kwargs['adata']
            else:
                raise ValueError(f'Unable to locate adata object in `*args` or `**kwargs`.')

            # at this point, it's impossible for adata to be of type anndata.Raw
            # but the message should tell it's possible for it to be an input
            assert isinstance(adata, (anndata.AnnData, )), f'Expected `{adata}` to be of type `anndata.AnnData` or `anndata.Raw`, found `{type(adata)}`.'

            # forcing always forces the callback
            if (call or force) and callback is None:
                if default_fn is None:
                    raise RuntimeError('No callback specified and default is None; specify it as a 1st argument. ')
                callback = default_fn
                assert callable(callback), f'`{callback}` is not callable.'

            if force:
                if verbose:
                    print('Computing values (forced).')
                if not call:
                    warnings.warn('Specifying `call=False` and `force=True` still forces the computation.')
                res = callback(*args, **kwargs)
                ret = cache_fn(res if copy else adata, fname, True, verbose, skip, *args, **kwargs)
                assert ret, 'Caching failed, horribly.'

                if is_plot:
                    # callback will show the plot
                    if adata.uns.pop(UNS_PLOT_KEY, None) is None:
                        # bad callback
                        warnings.warn(f'Plotting callbacks require the `adata` object to have `.uns[\'{UNS_PLOT_KEY}\']`' \
                                      ' containing the np.ndarray to plot (not found). You are likely seeing this because `skip=True`.')
                    return

                return anndata.Raw(res) if is_raw and res is not None else res

            # when loading to cache and copy is true, modify the copy
            if copy:
                adata = adata.copy()

            # we need to pass the *args and **kwargs in order to
            # get the right field when using regexes
            if not cache_fn(adata, fname, False, verbose, skip, *args, **kwargs):
                if verbose:
                    f = fname if fname is not None else def_fname
                    print(f'No cache found in `{str(f) + self.backend.ext}`, ' + ('computing values.' if call else 'searching for values.'))
                res = callback(*args, **kwargs) if call else adata if copy else None
                ret = cache_fn(res if copy else adata, fname, True, False, skip, *args, **kwargs)
                assert ret, 'Caching failed, horribly.'

                if is_plot:
                    # callback will show the plot
                    if adata.uns.pop(UNS_PLOT_KEY, None) is None:
                        # bad callback
                        warnings.warn(f'Plotting callbacks require the `adata` object to have `.uns[\'{UNS_PLOT_KEY}\']`' \
                                      ' containing the np.ndarray to plot (not found). You are likely seeing this because `skip=True`.')
                    return

                return anndata.Raw(res) if is_raw and res is not None else res

            if is_plot:
                data = adata.uns.pop(UNS_PLOT_KEY, None)
                if data is None:
                    # bad data loading
                    raise RuntimeError('Unable to load plot. No data found in `adata.uns[\'{UNS_PLOT_KEY}\']`.')

                return Image.fromarray(data)

            # if cache was found and not modifying inplace
            if not copy:
                return

            if is_raw:
                return anndata.Raw(adata)

            return adata

        def_fname = kwargs.get('default_fname', None)  # keep in in kwargs
        default_fn = kwargs.pop('default_fn', lambda *_x, **_y: None)
        is_plot = kwargs.pop('is_plot', False)  # plotting fuctions are treated as special

        cache_fn = self._create_cache_fn(*args, **kwargs)

        return FunctionWrapper(wrapper, default_fn)


class PpModule(Module):

    def __init__(self, backend, **kwargs):
        self._type = 'pp'
        self._functions = {
            'pcarr': wrap_as_adata(self.cache(dict(obsm='X_pca'),
                                              default_fname='pca_arr',
                                              default_fn=sc.pp.pca),
                                   ret_attr=dict(obsm='X_pca')),
            'expression': self.cache(dict(X=None), default_fname='expression'),
            'moments': self.cache(dict(uns_opt='pca',
                                       uns_opt_cache1='neighbors',
                                       obsm_opt='X_pca',
                                       varm_opt='PCs',
                                       layers='Ms',
                                       layers_cache1='Mu'),
                                  default_fn=scv.pp.moments,
                                  default_fname='moments'),
             'pca': self.cache(dict(obsm='X_pca',
                                    varm='PCs',
                                    uns=['pca', 'variance_ratio'],
                                    uns_cache1=['pca', 'variance']),
                               default_fname='pca',
                               default_fn=sc.pp.pca),
             'neighbors': self.cache(dict(uns='neighbors'),
                                     default_fname='neighs',
                                     default_fn=sc.pp.neighbors)
        }
        super().__init__(backend, **kwargs)


class TlModule(Module):

    def __init__(self, backend, **kwargs):
        self._type = 'tl'
        self._functions = {
            'louvain': self.cache(dict(obs='louvain'),
                                  default_fname='louvain',
                                  default_fn=sc.tl.louvain),
            'tsne': self.cache(dict(obsm='X_tsne'),
                               default_fname='tsne',
                               default_fn=sc.tl.tsne),
            'umap': self.cache(dict(obsm='X_umap'),
                               default_fname='umap',
                               default_fn=sc.tl.umap),
            'diffmap': self.cache(dict(obsm='X_diffmap',
                                       uns='diffmap_evals',
                                       uns_cache1='iroot'),
                                  default_fname='diffmap',
                                  default_fn=sc.tl.diffmap),
            'paga': self.cache(dict(uns='paga'),
                               default_fn=sc.tl.paga,
                               default_fname='paga'),
            'velocity': self.cache(dict(var='velocity_gamma',
                                        var_cache1='velocity_r2',
                                        var_cache2='velocity_genes',
                                        layers='velocity'),
                                   default_fn=scv.tl.velocity,
                                   default_fname='velo'),
            'velocity_graph': self.cache(dict(uns=re.compile(r'(.+)_graph$'),
                                              uns_cache1=re.compile('(.+)_graph_neg$')),
                                         default_fn=scv.tl.velocity_graph,
                                         default_fname='velo_graph'),
            'velocity_embedding': self.cache(dict(obsm=re.compile(r'^velocity_(.+)$')),
                                             default_fn=scv.tl.velocity_embedding,
                                             default_fname='velo_emb'),
            'draw_graph': self.cache(dict(obsm=re.compile(r'^X_draw_graph_(.+)$'),
                                          uns='draw_graph'),
                                     default_fn=sc.tl.draw_graph,
                                     default_fname='draw_graph'),
            'recover_dynamics': self.cache(
                dict(uns='recover_dynamics',
                     layers='fit_t',
                     layers_cache1='fit_tau',
                     layers_cache2='fit_tau_',
                     varm_opt='loss',
                     # the keys are taken from the source file
                     # and var is optional, since it's still under development
                     **{f'var_cache{i}_opt':re.compile(rf'(.+)_{name}$')
                        for i, name in enumerate(['alpha', 'beta', 'gamma', 't_', 'scaling',
                                                  'std_u', 'std_s', 'likelihood', 'u0', 's0',
                                                  'pval_steady', 'steady_u', 'steady_s'])}),
                 default_fn=scv.tl.recover_dynamics,
                 default_fname='recover_dynamics'
            )
        }
        super().__init__(backend, **kwargs)


class PlModule(Module):


    def __init__(self, backend, **kwargs):
        self._type = 'pl'
        self._functions = {
            fn.__name__:self.cache(dict(uns=UNS_PLOT_KEY),
                                   default_fname=f'{fn.__name__}_plot',
                                   default_fn=plotting_wrapper(fn),
                                   is_plot=True)

            for fn in filter(lambda fn: np.in1d(['return_fig', 'save'],  # only this works (wanted to  have with 'show')
                                                list(signature(fn).parameters.keys())).any(),
                             filter(callable, map(lambda name: getattr(sc.pl, name), dir(sc.pl))))
        }
        super().__init__(backend, **kwargs)
