from .modules import PpModule, TlModule, PlModule
from .backends import PickleBackend
from .utils import *
from pathlib import Path

import os


class Cache:

    _backends = dict(pickle=PickleBackend)
    _extensions = dict(pickle='.pickle')

    def __init__(self, root_dir, backend='pickle',
                 ext=None, separate_dirs=False):
        '''
        Params
        --------
        root_dir: Str
            path to directory where to save the files
        backend: Str, optional (default: `'pickle'`)
            which backend to use
        ext: Str, optional (default: `None`)
            file extensions, defaults to '.pickle' for
            'pickle' backend; defaults to '.scdata' if non applicable
        seperate_dirs: Bool, optional (default: `True`)
            whether to create 'pp', 'tl' and 'pl' directories
            under the `root_dir`
        '''

        self._separate_dirs = separate_dirs

        backend_type = self._backends.get(backend, None)
        if backend_type is None:
            raise ValueError(f'Unknown backend type: `{backend_type}`. Supported backends are: `{", ".join(self._backends.keys())}`.')

        self._root_dir = os.path.expanduser(root_dir)
        self._root_dir = Path(self._root_dir)#os.path.abspath(self._root_dir))
        self._ext = ext if ext is not None else self._extensions.get(backend, '.scdata')
        if not self._ext.startswith('.'):
            self._ext = '.' + self._ext

        if self._separate_dirs:
            for where, Mod in zip(['pp', 'tl', 'pl'], [PpModule, TlModule, PlModule]):
                setattr(self, where, Mod(backend, dirname=os.path.join(self._root_dir, where), ext=self._ext, cache=self))
        else:
            # shared backend
            self._backend = backend_type(root_dir, self._ext, cache=self)
            self.pp = PpModule(self._backend)
            self.tl = TlModule(self._backend)
            self.pl = PlModule(self._backend)

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, dir):
        raise RuntimeError('Setting root directory is disallowed.')

    def clear(self, verbose=1):
        if not self._separate_dirs:
            # same backend for everyone
            self._backend._clear(verbose=verbose)
        else:
            # |Deleting| = 8
            self.pp._clear(verbose, separator='-' * 8)
            self.tl._clear(verbose, separator='-' * 8)
            self.pl._clear(verbose)

    def __repr__(self):
        return f"{self.__class__.__name__}(root={self._root_dir}, ext='{self._ext}')"
