from .backends import Backend, PickleBackend
from .utils import *
from collections import Iterable 
from abc import ABC
from inspect import signature
from PIL import Image
from random import randint

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

        self.verbose = True
        self.force = False

        for k, fn in self._functions.items():
            setattr(self, k, fn)

    @property
    def backend(self):
        '''
        The backend for this module.
        '''

        return self._backend

    @backend.setter
    def backend(self, _):
        raise RuntimeError('Setting backend is disallowed. To change the directory, try `dir` attribute of `backend`.')

    @property
    def verbose(self):
        '''
        Verbosity level for this module.
        Overriden by `verbose=...` when explictly specified during the function call.
        '''

        return self._verbose

    @verbose.setter
    def verbose(self, val):
        assert isinstance(val, bool), f'Value must be of type `bool`, found `{type(val).__name__}`.'
        self._verbose = val

    @property
    def force(self):
        '''
        Whether to force computation for this module.
        Overriden by `force=...` when explictly specified during the function call.
        '''

        return self._force

    @force.setter
    def force(self, val):
        assert isinstance(val, bool), f'Value must be of type `bool`, found `{type(val).__name__}`.'
        self._force = val

    def _clear(self, verbose=1, *, separator=None):
        self.backend._clear(verbose, self._type, separator=separator)

    def clear(self, verbose=1):
        self._clear(verbose)

    def __iter__(self):  # to list available functions
        return iter(self._functions.keys())

    def __repr__(self):
        return f'<{self.__module__}.{self._type}>'

    def _create_cache_fn(self, *args, default_fname=None):

        def wrapper(adata, fname=None, recache=False, verbose=True,
                    skip=False, keyhint=None, watchers={}, *args, **kwargs):
            try:
                if fname is None:
                    fname = default_fname
                if not fname.endswith(self.backend.ext):
                    fname += self.backend.ext

                if recache:
                    return self.backend.save(adata, fname, attrs, keys,
                                             skip=skip, is_optional=is_optional,
                                             keyhint=keyhint,
                                             watcher_keys=watcher_keys,
                                             watchers=watchers,
                                             verbose=verbose)

                if (self.backend.dir / fname).is_file():
                    if verbose:
                        print(f'Loading data from: `{fname}`.')

                    # TODO: watchers for backend? is it reasonable?
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
        pat = re.compile(r'(?:_opt)|(?:_cache\d+)')
        watcher_keys = attrs  # needed for watchers
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
        default_keyhint: Str
            when ambiguous matches occurr, save all values which have
            `default_keyhint` inside
            overridable by `keyhint` when calling the function

        Returns
        --------
        caching_function: Callable
            caching function accepting as the first argument either
            `anndata.AnnData` object or a `callable` and `anndata.AnnData`
            object as the second argument
            the `callable` either needs to return an `anndata.AnnData` object
            (if `copy=True`) or just modify it inplace
        '''

        def get_watchers(callback, *args, **kwargs):
            try:
                sig = signature(callback)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                res = {}
                for k, vs in watchers_.items():
                    tmp = {}
                    for v in vs:
                        to_ignore = None
                        if '!' in v:
                            v, to_ignore = v.split('!')

                        # TODO: this is hacky and ugly, refactor
                        # ignore if default/not specified
                        if to_ignore is not None and v not in kwargs:
                            tmp[to_ignore] = f'IGNORE_{randint(0, 128)}'
                        elif '<' in v:
                            v, default = v.split('<')
                            # in case args change
                            # if v in bound.arguments:
                            tmp[v] = default if bound.arguments[v] is None else bound.arguments[v]
                        else:
                            v, *default= v.split('>')
                            # if v in bound.arguments:
                            tmp[v] = default[0] if default and bound.arguments[v] is not None else bound.arguments[v] 
                    res[k] = tmp

                return res

            except TypeError:
                return {}

        def wrapper(*args, **kwargs):
            fname = kwargs.pop('fname', None)
            force = kwargs.pop('force') if 'force' in kwargs else self.force 
            verbose = kwargs.pop('verbose') if 'verbose' in kwargs else self.verbose
            call = kwargs.pop('call', True)  # if we do not wish to call the callback
            skip = kwargs.pop('skip', False)
            # resolver of ambigous matches
            keyhint = kwargs.pop('keyhint', None) or def_keyhint
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
                assert callable(callback), f'Function `{callback}` is not callable.'

            watchers = get_watchers(callback, *args, **kwargs)

            if force:
                if verbose:
                    print('Computing values (forced).')
                if not call:
                    warnings.warn('Specifying `call=False` and `force=True` still forces the computation.')
                res = callback(*args, **kwargs)
                ret = cache_fn(res if copy else adata, fname, True, verbose,
                               skip, keyhint, watchers, *args, **kwargs)
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
            if not cache_fn(adata, fname, False, verbose, skip, keyhint, watchers, *args, **kwargs):
                if verbose:
                    f = fname if fname is not None else def_fname
                    print(f'No cache found in `{str(f) + self.backend.ext}`, ' + ('computing values.' if call else 'searching for values.'))

                res = callback(*args, **kwargs) if call else adata if copy else None
                ret = cache_fn(res if copy else adata, fname, True, False,
                               skip, keyhint, watchers, *args, **kwargs)
                assert ret, 'Caching failed, horribly.'

                if is_plot:
                    # callback will show the plot
                    if adata.uns.pop(UNS_PLOT_KEY, None) is None:
                        # bad callback
                        warnings.warn(f'Plotting callbacks require the `adata` object to have `.uns[\'{UNS_PLOT_KEY}\']`' \
                                      ' containing `np.ndarray` to plot (not found). You are likely seeing this because `skip=True`.')
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
        def_keyhint = kwargs.pop('default_keyhint', None)
        default_fn = kwargs.pop('default_fn', lambda *_x, **_y: None)

        # watchers can't be done in _create_cache_fn,
        # because the callback is dynamic as well
        watchers_ = kwargs.pop('watchers', {})
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
                                     default_fn=sc.pp.neighbors),
             'combat': self.cache(dict(X=None),
                                  default_fn=sc.pp.combat,
                                  default_fname='combat'),
             'regress_out': self.cache(dict(X=None),
                                          default_fname='regress_out',
                                          default_fn=sc.pp.regress_out),
             'scale': self.cache(dict(X=None),
                                    default_fname='scale',
                                    default_fn=sc.pp.scale)
        }
        super().__init__(backend, **kwargs)


class TlModule(Module):

    def __init__(self, backend, **kwargs):
        self._type = 'tl'
        self._functions = {
            'rank_genes_groups': self.cache(dict(uns=re.compile(rf'(?P<key_added>.*)')),
                                            watchers=dict(uns=['key_added<rank_genes_groups']),
                                            default_fn=sc.tl.rank_genes_groups,
                                            default_fname='rank_genes_groups'),
            'louvain': self.cache(dict(obs=re.compile(r'(?P<key_added>.*?)(?P<restrict_to>_R)?$')),
                                  watchers=dict(obs=['key_added', 'restrict_to>_R']),
                                  default_fname='louvain',
                                  default_fn=sc.tl.louvain),
            'leiden': self.cache(dict(obs=re.compile(r'(?P<key_added>.*?)(?P<restrict_to>_R)?$')),
                                 watchers=dict(obs=['key_added', 'restrict_to>_R']),
                                 default_fname='leiden',
                                 default_fn=sc.tl.leiden),
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
            # this is a bit overkill, but I want to save the extra info only for
            # `n_branchings` > 0
            # TODO: refactor
            'dpt': self.cache(dict(obs='dpt_pseudotime',
                                   obs_opt_cache1=re.compile(rf'^dpt_groups$'),
                                   obs_opt_cache2=re.compile(rf'^dpt_order$'),
                                   obs_opt_cache3=re.compile(rf'^dpt_order_indices$'),
                                   uns_opt_cache1=re.compile(rf'^dpt_changepoints$'),
                                   uns_opt_cache2=re.compile(rf'^dpt_grouptips$')),
                              watchers=dict(obs_opt_cache1=['n_branchings!dpt_groups'],
                                            obs_opt_cache2=['n_branchings!dpt_order'],
                                            obs_opt_cache3=['n_branchings!dpt_order_indices'],
                                            uns_opt_cache1=['n_branchings!dpt_changepoints'],
                                            uns_opt_cache2=['n_branchings!dpt_grouptips']),
                              default_fn=sc.tl.dpt,
                              default_fname='dpt'),
            'embedding_density': self.cache(dict(obs=re.compile(r'(?P<basis>.*)_density_?(?P<groupby>.*)'),
                                                 # don't do greede groupby and since users can be evil
                                                 # don't use [^_]*
                                                 uns=re.compile(r'(?P<basis>.*)_density_(?P<groupby>.*?)_?params')),
                                            watchers=dict(obs=['basis', 'groupby'],
                                                          uns=['basis', 'groupby']),
                                            default_fn=sc.tl.embedding_density,
                                            default_fname='embedding_density'),
            'velocity': self.cache(dict(var_opt=re.compile('(?P<vkey>.*)_genes'), # dyn
                                        layers=re.compile('(?P<vkey>.*)'),  # all
                                        layers_opt_cache1=re.compile('(?P<vkey>.*)_u'), # dyn
                                        layers_opt_cache2=re.compile('variance_(?P<vkey>.*)'),  # stoch, ...
                                        **{f'var_opt_cache{i}': re.compile(rf'(?P<vkey>.*)_{name}_?.*')
                                           for i, name in enumerate(['offset', 'offset2', 'beta',
                                                                     'gamma', 'r2', 'genes'])}),
                                   watchers=dict(var_opt=['vkey'], layers=['vkey'],
                                                 layers_opt_cache1=['vkey'], layers_opt_cache2=['vkey'],
                                                 **{f'var_opt_cache_{i}': ['vkey']
                                                    for i, name in enumerate(['offset', 'offset2', 'beta',
                                                                              'gamma', 'r2', 'genes'])}),
                                   default_fn=scv.tl.velocity,
                                   default_fname='velo'),
            'velocity_graph': self.cache(dict(uns=re.compile(r'(?P<vkey>.*)_graph'),
                                              uns_cache1=re.compile(r'(?P<vkey>.*)_graph_neg'),
                                              obs=re.compile(r'(?P<vkey>.*)_self_transition')),
                                         watchers=dict(uns=['vkey'],
                                                       uns_cache1=['vkey'],
                                                       obs=['vkey']),
                                         default_fn=scv.tl.velocity_graph,
                                         default_fname='velo_graph'),
            'velocity_embedding': self.cache(dict(obsm=re.compile(r'(?P<vkey>.*)_(?P<basis>.*)')),
                                             watchers=dict(obsm=['vkey', 'basis']),
                                             default_fn=scv.tl.velocity_embedding,
                                             default_fname='velo_emb'),
            # this is the closest correct version
            'draw_graph': self.cache(dict(obsm=re.compile(r'X_draw_graph_(?P<layout>.*)'),
                                          uns='draw_graph'),
                                     watchers=dict(obsm=['layout']),
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
                     **{f'var_opt_cache{i}':re.compile(rf'(.+)_{name}$')
                        for i, name in enumerate(['alpha', 'beta', 'gamma', 't_', 'scaling',
                                                  'std_u', 'std_s', 'likelihood', 'u0', 's0',
                                                  'pval_steady', 'steady_u', 'steady_s'])}),
                 default_fn=scv.tl.recover_dynamics,
                 default_fname='recover_dynamics',
                 default_keyhint='fit'
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
            # only this works (wanted to have with 'show')
            for fn in filter(lambda fn: np.in1d(['return_fig', 'save'],
                                                list(signature(fn).parameters.keys())).any(),
                             filter(callable, map(lambda name: getattr(sc.pl, name), dir(sc.pl))))
        }
        super().__init__(backend, **kwargs)

