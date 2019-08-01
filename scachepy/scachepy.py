#!/usr/bin/env python3


from pathlib import Path
from collections import Iterable, namedtuple

import scvelo as scv
import scanpy as sc
import anndata

import os
import re
import numpy as np
import pickle
import traceback
import warnings


# TODO: improvements
# 1. simplify logic, if possible (should be)
# 2. get docstrings working
# 3. use module-like approach instead of the namedtuple
# (namely because of autocompletion, code cleanliness)
# 4. .h5ad backend?


class PrintWrapper():

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


class Cache():

    def __init__(self, cache_dir, ext='.pickle', make_dir=True):
        cache_dir = os.path.expanduser(cache_dir)
        cache_dir = os.path.abspath(cache_dir)

        # TODO: maybe use cache_dir/{pp,pl,tl} subdirs
        if make_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_dir = cache_dir
        self._ext = ext

        self._init_pp()
        self._init_tl()
        self._init_pl()

    def _init_pp(self):
        functions = {
            # TODO: not ideal - the PrintWrapper required also the function
            # opens doors for errors
            # we must wrapper the last function as opposed to the function returned
            # by self.cache
            'pcarr': PrintWrapper(self._wrap_as_adata(self.cache(dict(obsm='X_pca'),
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
                                     default_fn=sc.pp.neighbors),
        }

        self._pp = namedtuple('pp', functions.keys())(**functions)

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
            'draw_graph': self.cache(dict(obsm=re.compile(r'^X_draw_graph_(.+)$'),
                                          uns='draw_graph'),
                                     default_fn=sc.tl.draw_graph,
                                     default_fname='draw_graph'),
        }

        self._tl = namedtuple('tl', functions.keys())(**functions)

    def _init_pl(self):
        functions = dict()
        self._pl = namedtuple('pl', functions.keys())(**functions)

    def __repr__(self):
        return f"{self.__class__.__name__}(dir='{self._cache_dir}', ext='{self._ext}')"


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
            # currently, only 1 key supported
            for attr, k in ret_attr.items():
                out.append(getattr(res, attr)[k])

            if len(ret_attr) == 1:
                return out[0]
    
            return tuple(out)

        return PrintWrapper(wrapper, fn)


    @property
    def pp(self):
        return self._pp

    @pp.setter
    def pp(self, _):
        raise RuntimeError('Setting not allowed.') 

    @property
    def tl(self):
        return self._tl

    @tl.setter
    def tl(self, _):
        raise RuntimeError('Setting not allowed.') 

    @property
    def pl(self):
        return self._pl

    @pl.setter
    def pl(self, _):
        raise RuntimeError('Setting not allowed.') 

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value):
        if not isinstance(value, Path):
            value = Path(value)

        if not value.exists():
            warnings.warn(f'Path `{value}` does not exist.')
        elif not value.is_dir():
            warnings.warn(f'`{value}` is not a directory.')

        self._cache_dir = value

    def _create_cache_fn(self, *args, default_fname=None):

        def helper(adata, fname=None, recache=False, verbose=True, *args, **kwargs):

            def _get_val(obj, keys):
                if keys is None:
                    return obj

                if isinstance(keys, str) or not isinstance(keys, Iterable):
                    keys = (keys, )

                for k in keys:
                    obj = obj[k]

                return obj

            def _convert_key(attr, key):
                if key is None or isinstance(key, str):
                    return key

                if isinstance(key, re._pattern_type):
                    km = {key.match(k).groups()[0]:k for k in getattr(adata, attr).keys() if key.match(k) is not None}
                    res = set(km.keys()) & possible_vals

                    if len(res) == 0:
                        # default value was not specified during the call
                        assert len(km) == 1, f'Found ambiguous matches for `{key}` in attribute `{attr}`: `{set(km.keys())}`.'
                        return tuple(km.values())[0]

                    assert len(res) == 1, f'Found ambiguous matches for `{key}` in attribute `{attr}`: `{res}`.'
                    return km[res.pop()]

                assert isinstance(key, Iterable)

                # converting to tuple because it's hashable
                return tuple(key)

            possible_vals = set(args) | set(kwargs.values())
            try:

                if fname is None:
                    fname = default_fname

                if not fname.endswith(self._ext):
                    fname += self._ext

                if recache:
                    if verbose:
                        print(f'Caching data to: `{fname}`.')
                    data = [((attr, (key, ) if key is None or isinstance(key, str) else key),
                              _get_val(getattr(adata, attr), key)) for attr, key in map(lambda a_k: (a_k[0], _convert_key(*a_k)), zip(attrs, keys))]
                    with open(self.cache_dir / fname, 'wb') as fout:
                        pickle.dump(data, fout)

                    return True

                if (self.cache_dir / fname).is_file():
                    if verbose:
                        print(f'Loading data from: `{fname}`.')

                    with open(self.cache_dir / fname, 'rb') as fin:
                        attrs_keys, vals = zip(*pickle.load(fin))

                    for (attr, key), val in zip(attrs_keys, vals):
                        if key is None or isinstance(key, str):
                            key = (key, )

                        if not hasattr(adata, attr):
                            if attr == 'obsm': shape = (adata.n_obs, )
                            elif attr == 'varm': shape = (adata.n_vars, )
                            else: raise AttributeError('Support only for `.varm` and `.obsm` attributes.')

                            assert len(keys) == 1, 'Multiple keys not allowed in this case.'
                            setattr(adata, attr, np.empty(shape))

                        if key[0] is not None:
                            at = getattr(adata, attr)
                            msg = [f'adata.{attr}']

                            for k in key[:-1]:
                                if k not in at.keys():
                                    at[k] = dict()
                                at = at[k]
                                msg.append(f'[{k}]')

                            if verbose and key[-1] in at.keys():
                                print(f'Warning: `{"".join(msg)}` already contains key: `{key[-1]}`.')
                            at[key[-1]] = val
                        else:
                            if verbose and hasattr(adata, attr):
                                print(f'Warning: `adata.{attr}` already exists.')
                            setattr(adata, attr, val)

                    return True

                return False

            except Exception as e:
                if not isinstance(e, FileNotFoundError):
                    if recache:
                        print(traceback.format_exc())
                else:
                    print(f'No cache found in `{self._cache_dir / fname}`.')

                return False

        if len(args) == 1:
            collection = args[0]
            if isinstance(collection, dict):
                attrs = tuple(collection.keys())
                keys = tuple(collection.values())

                pat = re.compile(r'_cache\d+$')
                attrs = tuple(pat.sub('', a) for a in attrs)

                return helper

            if isinstance(collection, Iterable) and len(next(iter(collection))) == 2:
                attrs, keys = tuple(zip(*collection))

                return helper

        assert len(args) == 2
        attrs, keys = args

        if isinstance(attrs, str):
            attrs = (attrs, )
        if isinstance(keys, str):
            keys = (keys, )

        return helper


    def cache(self, *args, wrap=True, **kwargs):
        """
        Create a caching function.

        :param: keys_attributes (dict, list(tuple))
        :param: wrap (bool)
            whether to use PrintWrapper for nicer output
        :param: default_fname (str)
            default filename where to save the pickled data
        """

        def _run(*args, **kwargs):
            """
            :param: *args
            :param: **kwargs (fname, force, verbose)
            """

            fname = kwargs.pop('fname', None)
            force = kwargs.pop('force', False)
            verbose = kwargs.pop('verbose', True)
            copy = kwargs.get('copy', False)


            callback = None
            if len(args) > 1:
                callback, *args = args

            if len(args) > 0:
                adata = args[0] if isinstance(args[0], (anndata.AnnData, anndata.base.Raw)) else kwargs.get('adata')
            else:
                adata = kwargs.get('adata')

            assert isinstance(adata, (anndata.AnnData, anndata.base.Raw)), f'Expected `{adata}` to be of type `anndata.AnnData`.'

            if callback is None:
                callback = (lambda *_x, **_y: None) if default_fn is None else default_fn

            assert callable(callback), f'`{callblack}` is not callable.'

            if force:
                if verbose:
                    print('Forced computing values.')
                res = callback(*args, **kwargs)
                cache_fn(res if copy else adata, fname, True, verbose, *args, **kwargs)
                return res

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

                return res

            # if cache was found and not modifying inplace
            return adata if copy else None

        default_fn = kwargs.pop('default_fn', None)
        cache_fn = self._create_cache_fn(*args, **kwargs)

        return PrintWrapper(_run, default_fn) if wrap else _run
