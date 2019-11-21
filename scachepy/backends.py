from abc import ABC, abstractmethod
from collections import Iterable
from pathlib import Path

import numpy as np
import os
import re
import pickle
import warnings


class Backend(ABC):

    def __init__(self, dirname, ext, *, cache):
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        self._dirname = Path(dirname)
        self._cache = cache
        self.ext = ext

    def _clear(self, verbose=1, typp='all', separator=None):
        files = [f for f in os.listdir(self.dir) if f.endswith(self.ext) and
                 os.path.isfile(os.path.join(self.dir, f))]
        if verbose > 0:
            print(f'Deleting {len(files)} files from `{typp}`.')

        for f in files:
            if verbose > 1:
                print(f'Deleting `{f}`.')
            os.remove(os.path.join(self.dir, f))

        if separator is not None:
            print(separator)
    
    @property
    def dir(self):
        return self._dirname

    @dir.setter
    def dir(self, value):
        if not isinstance(value, Path):
            value = Path(value)

        if self._cache._separate_dirs:
            self._dirname = self._cache.root_dir / value
        else:
            self._dirname = value
            self._cache._root_dir = value

        if not self._dirname.exists():
            os.makedirs(self._dirname)

    @abstractmethod
    def load(self, adata, fname, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, adata, fname, attrs, keys, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(dir='{self.dir}', ext='{self.ext}')"


class PickleBackend(Backend):

    def load(self, adata, fname, *args, **kwargs):
        skip_if_not_found = kwargs.get('skip', False)
        verbose = kwargs.get('verbose', False)

        with open(os.path.join(self.dir, fname), 'rb') as fin:
            attrs_keys, vals = zip(*pickle.load(fin))

            for (attr, key), val in zip(attrs_keys, vals):
                if val is None:
                    msg = f'Cache contains empty value for attribute `{attr}`, key `{key}`. This could have happened when caching these values failed.'
                    if not skip_if_not_found:
                        raise RuntimeError(msg + ' Use `skip=True` to skip the aforementioned keys.')
                    warnings.warn(msg + ' Skipping.')
                    continue

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
                        msg.append(f'[\'{k}\']')

                    if verbose and key[-1] in at.keys():
                        print(f'Warning: `{"".join(msg)}` already contains key: `\'{key[-1]}\'`.')

                    at[key[-1]] = val
                else:
                    if verbose and hasattr(adata, attr):
                        print(f'Warning: `adata.{attr}` already exists.')

                    setattr(adata, attr, val)

        return True

    def save(self, adata, fname, attrs, keys, *args, keyhint=None, **kwargs):
        
        # value not found from _get_val
        sentinel = object()

        def _get_val(obj, keys, optional):
            try:
                if keys is None:
                    return obj

                if isinstance(keys, str) or not isinstance(keys, Iterable):
                    keys = (keys, )

                for k in keys:
                    obj = obj[k]

            except KeyError as e:
                if optional:
                    return sentinel

                msg = f'Unable to find keys `{", ".join(map(str, keys))}`.'
                if not skip_not_found:
                    raise RuntimeError(msg + ' Use `skip=True` to skip the aforementioned keys.') from e
                warnings.warn(msg + ' Skipping.')

                return sentinel

            return obj

        def _convert_key(attr, key):
            if key is None or isinstance(key, str):
                return key

            if isinstance(key, re._pattern_type):
                km = {key.match(k).groups()[0]:k for k in getattr(adata, attr).keys() if key.match(k) is not None}
                res = set(km.keys()) & possible_vals

                if len(res) == 0:
                    # default value was not specified during the call
                    if len(km) != 1:
                        assert keyhint is not None, \
                                f'Found ambiguous matches for `{key}` in attribute `{attr}`: `{set(km.values())}`. ' \
                                'Try specifying `keyhint=\'...\'`.'

                        return tuple(v for v in km.values() if keyhint in v)

                    return tuple(km.values())[0]

                if len(res) != 1:
                    assert keyhint is not None, \
                                f'Found ambiguous matches for `{key}` in attribute `{attr}`: `{res}`. ' \
                                'Try specifying `keyhint=\'...\'`.'

                    return tuple(v for v in km.values() if keyhint in v)

                return km[res.pop()]

            assert isinstance(key, Iterable)

            # converting to tuple because it's hashable
            return tuple(key)


        verbose = kwargs.get('verbose', False)
        skip_not_found = kwargs.get('skip', False)
        possible_vals = kwargs.get('possible_vals', {})
        is_optional = kwargs.get('is_optional', [False] * len(attrs))

        data = []
        for attr, key, opt in zip(attrs, keys, is_optional):
            if not hasattr(adata, attr):
                if opt:
                    continue
                raise AttributeError(f'`adata` object has no attribute `{attr}` and'
                                      ' was not specified as optional.')

            key = _convert_key(attr, key)

            value = _get_val(getattr(adata, attr), key, opt)
            if value is sentinel:
                # value not found - either skipping or optional
                continue

            if key is None or isinstance(key, str):
                key = (key, )

            head = (attr, key)
            data.append((head, value))

        with open(os.path.join(self.dir, fname), 'wb') as fout:
            pickle.dump(data, fout)

        return True
