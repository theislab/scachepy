from abc import ABC, abstractmethod
from collections import Iterable
from pathlib import Path


import os
import re
import pickle
import warnings


class Backend(ABC):

    def __init__(self, dirname, make_dir):
        if make_dir and not os.path.exists(dirname):
            os.makedirs(dirname)

        self.dir = dirname

    @property
    def dir(self):
        return self._dirname

    @dir.setter
    def dir(self, value):
        if not isinstance(value, Path):
            value = Path(value)

        if not value.exists():
            warnings.warn(f'Path `{value}` does not exist.')
        elif not value.is_dir():
            warnings.warn(f'`{value}` is not a directory.')

        self._dirname = value

    @abstractmethod
    def load(self, adata, fname, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, adata, fname, attrs, keys, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(dir='{self.dir}')"


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
                        msg.append(f'[{k}]')

                    if verbose and key[-1] in at.keys():
                        print(f'Warning: `{"".join(msg)}` already contains key: `{key[-1]}`.')

                    at[key[-1]] = val
                else:
                    if verbose and hasattr(adata, attr):
                        print(f'Warning: `adata.{attr}` already exists.')

                    setattr(adata, attr, val)

        return True


    def save(self, adata, fname, attrs, keys, *args, **kwargs):

        def _get_val(obj, keys):
            try:
                if keys is None:
                    return obj

                if isinstance(keys, str) or not isinstance(keys, Iterable):
                    keys = (keys, )

                for k in keys:
                    obj = obj[k]

            except KeyError as e:
                msg = f'Unable to find keys `{", ".join(map(str, keys))}`.'
                if not skip_not_found:
                    raise RuntimeError(msg + ' Use `skip=True` to skip the aforementioned keys.') from e
                warnings.warn(msg + ' Skipping.')

                return None

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


        verbose = kwargs.get('verbose', False)
        skip_not_found = kwargs.get('skip', False)
        possible_vals = kwargs.get('possible_vals', {})

        data = []
        for attr, key in zip(attrs, keys):
            key = _convert_key(attr, key)

            value = _get_val(getattr(adata, attr), key)
            if value is None:
                continue

            if key is None or isinstance(key, str):
                key = (key, )

            head = (attr, key)
            data.append((head, value))

        with open(os.path.join(self.dir, fname), 'wb') as fout:
            pickle.dump(data, fout)

        return True
