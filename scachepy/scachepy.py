#!/usr/bin/env python3

from .backends import PickleBackend
from .utils import Dummy, Wrapper

from collections import Iterable, namedtuple

import scvelo as scv
import scanpy as sc
import anndata

import os
import re
import numpy as np
import pickle
import traceback


class Cache:

    _backends = dict(pickle=PickleBackend)
    _extensions = dict(pickle='.pickle')

    def __init__(self, cache_dir, backend='pickle',
                 ext=None, make_dir=True):
        '''
        Params
        --------
        cache_dir: str
            path to directory where to save the files
        backend: str, optional (default: `'pickle'`)
            which backend to use
        ext: str, optional (default: `None`)
            file extensions, defaults to '.pickle' for
            'pickle' backend; defaults to '.scdata' if non applicable
        make_dir: bool, optional (default: `True`)
            make the `cache_dir` if it does not exist
        '''
        self._backend = self._backends.get(backend, None)
        if self._backend is None:
            raise ValueError(f'Unknown backend type: `{backend}`. Supported backends are: `{", ".join(self._backends.keys())}`.')

        cache_dir = os.path.expanduser(cache_dir)
        cache_dir = os.path.abspath(cache_dir)

        self._backend = self._backend(cache_dir, make_dir=make_dir)
        self._ext = ext if ext is not None else self._extensions.get(backend, '.scdata')

        self._init_pp()
        self._init_tl()
        self._init_pl()

    def _init_pp(self):
        functions = {
            # TODO: not ideal - the Wrapper requires the function to be specified
            # we also must wrap the last function as opposed to the function returned by self.cache
            'pcarr': Wrapper(self._wrap_as_adata(self.cache(dict(obsm='X_pca'),
                                                                default_fname='pca_arr',
                                                                default_fn=sc.pp.pca, wrap=False),
                                                     ret_attr=dict(obsm='X_pca')),
                                 sc.pp.pca),
            'expression': self.cache(dict(X=None), default_fname='expression'),
            'moments': self.cache(dict(uns='pca',
                                       uns_cache1='neighbors',
                                       obsm='X_pca',
                                       varm='PCs',
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
        self.pp = Dummy('pp', **functions)

    def _init_tl(self):
        functions = {
            'louvain': self.cache(dict(obs='louvain'),
                                  default_fname='louvain',
                                  default_fn=sc.tl.louvain),
            'umap': self.cache(dict(obsm='X_umap'),
                               default_fname='umap',
                               default_fn=sc.tl.umap),
            'diffmap': self.cache(dict(obsm='X_diffmap', uns='diffmap_evals', uns_cache1='iroot'),
                                  default_fname='diffmap',
                                  default_fn=sc.tl.diffmap),
            'paga': self.cache(dict(uns=['paga', 'connectivities'],
                                    uns_cache1=['paga','connectivities_tree'],
                                    uns_cache2=['paga', 'groups'],
                                    uns_cache3=['paga', 'pos']),
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
                                     default_fname='draw_graph')
        }
        self.tl = Dummy('tl', **functions)

    def _init_pl(self):
        functions = dict()
        self.pl = Dummy('pl', **functions)

    def __repr__(self):
        return f"{self.__class__.__name__}(backend={self.backend}, ext='{self._ext}')"

    def _wrap_as_adata(self, fn, *, ret_attr):

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

        return Wrapper(wrapper, fn)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, _):
        raise RuntimeError('Setting is disallowed.') 

    def _create_cache_fn(self, *args, default_fname=None):

        def wrapper(adata, fname=None, recache=False, verbose=True, *args, **kwargs):
            try:
                if fname is None:
                    fname = default_fname
                if not fname.endswith(self._ext):
                    fname += self._ext

                if recache:
                    possible_vals = set(args) | set(kwargs.values())
                    return self.backend.save(adata, fname, attrs, keys, possible_vals=possible_vals, verbose=verbose)

                if (self.backend.dir / fname).is_file():
                    if verbose:
                        print(f'Loading data from: `{fname}`.')

                    return self.backend.load(adata, fname, verbose=verbose)

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

        # strip the postfix
        pat = re.compile(r'_cache\d+$')
        attrs = tuple(pat.sub('', a) for a in attrs)

        return wrapper


    def cache(self, *args, wrap=True, **kwargs):
        """
        Create a caching function.
        Params
        --------

        args: dict(str, Union[str, Iterable[Union[str, re._pattern_type]]])
            attributes are supplied as dictionary keys and
            values as dictionary values (need not be an Iterable)
            for caching multiple attributes of the same name,
            append to them postfixes of the following kind: `_cache1, _cache2, ...`
            there are also other ways of specifying this, please
            refer the source code of `_create_cache_fn`
        wrap: bool, optional (default: `True`)
            whether to wrap in a pretty printing wrapper
        default_fname: str
            default filename where to save the pickled data
        default_fn: callable, optional (default: `None`)
            function to call before caching the values

        Returns
        --------
        a caching function accepting as the first argument either
        anndata.AnnData object or a callable and anndata.AnnData
        object as the second argument
        """


        def wrapper(*args, **kwargs):
            fname = kwargs.pop('fname', None)
            force = kwargs.pop('force', False)
            verbose = kwargs.pop('verbose', True)
            copy = kwargs.get('copy', False)

            callback = None
            if len(args) > 1:
                callback, *args = args

            is_raw = False
            if len(args) > 0 and isinstance(args[0], (anndata.AnnData, anndata.Raw)):
                if isinstance(args[0], anndata.Raw):
                    args = (args[0]._adata, *args[1:])
                    is_raw = True
                adata = args[0]
            elif 'adata' in kwargs:
                if isinstance(kwargs['adata'], anndata.Raw):
                    kwargs['adata'] = kwargs['adata']._adata
                    is_raw = True
                adata = kwargs['adata']
            else:
                raise ValueError(f'Unable to locat adata object in args or kwargs.')

            # at this point, it's impossible for adata to be of type anndata.Raw
            # but the message should tell it's possible for it to be an input
            assert isinstance(adata, (anndata.AnnData, )), f'Expected `{adata}` to be of type `anndata.AnnData` or `anndata.Raw`.'

            if callback is None:
                callback = default_fn
            assert callable(callback), f'`{callblack}` is not callable.'

            if force:
                if verbose:
                    print('Forced computing values.')
                res = callback(*args, **kwargs)
                ret = cache_fn(res if copy else adata, fname, True, verbose, *args, **kwargs)
                assert ret, 'Caching failed.'

                return anndata.Raw(res) if is_raw and res is not None else res

            # when loading to cache and copy is true, modify the copy
            if copy:
                adata = adata.copy()

            # we need to pass the *args and **kwargs in order to
            # get the right field when using regexes
            if not cache_fn(adata, fname, False, verbose, *args, **kwargs):
                if verbose:
                    print('No cache found, computing values.')
                res = callback(*args, **kwargs)
                ret = cache_fn(res if copy else adata, fname, True, False, *args, **kwargs)
                assert ret, 'Caching failed.'

                return anndata.Raw(res) if is_raw and res is not None else res

            # if cache was found and not modifying inplace
            if not copy:
                return None

            if is_raw:
                return anndata.Raw(adata)

            return adata

        default_fn = kwargs.pop('default_fn', lambda *_x, **_y: None)
        cache_fn = self._create_cache_fn(*args, **kwargs)

        return Wrapper(wrapper, default_fn) if wrap else wrapper 
