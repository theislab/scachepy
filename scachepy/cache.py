from .modules import PpModule, TlModule, PlModule
from .backends import PickleBackend
from .utils import *
from pathlib import Path

import compress_pickle as cpickle

import os


class Cache:

    _backends = dict(pickle=PickleBackend)
    _extensions = dict(pickle='.pickle')
    _compressions = cpickle.get_known_compressions()

    def __init__(self, root_dir, backend='pickle',
                 compression=None,
                 ext=None, separate_dirs=False):
        f"""
        Params
        --------
        root_dir: Str
            path to directory where to save the files
        backend: Str, optional (default: `'pickle'`)
            which backend to use
        compression: Str, optional (default: `'zip'`)
            compression scheme to use, valid options are:
            `{', '.join(map(str, self._compressions))}`
        ext: Str, optional (default: `None`)
            file extensions, defaults to '.pickle' for
            'pickle' backend; defaults to '.scdata' if non applicable
        seperate_dirs: Bool, optional (default: `True`)
            whether to create 'pp', 'tl' and 'pl' directories
            under the `root_dir`
        """
        assert compression in self._compressions, 'Invalid compression type: `{compression}`. ' \
            f'Valid options are: `{self._compressions}`.'

        backend_type = self._backends.get(backend, None)
        if backend_type is None:
            raise ValueError(f'Unknown backend type: `{backend_type}`. '
                             'Supported backends are: `{", ".join(self._backends.keys())}`.')

        self._separate_dirs = separate_dirs

        self._root_dir = os.path.expanduser(root_dir)
        self._root_dir = Path(self._root_dir)

        self._ext = ext if ext is not None else self._extensions.get(backend, '.scachepy')
        if not self._ext.startswith('.'):
            self._ext = f'.{self._ext}'

        self._compression = compression
        if self._compression is not None and self._compression != 'pickle':
            self._ext += f'.{self._compression}'

        if self._separate_dirs:
            for where, Mod in zip(['pp', 'tl', 'pl'], [PpModule, TlModule, PlModule]):
                setattr(self, where, Mod(backend, dirname=os.path.join(self._root_dir, where),
                        ext=self._ext, cache=self))
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

    def verbose(self, val):
        '''
        Set the verbosity level for all modules.
        For module-specific behavior, use `c.<module>.verbose = ...`.

        Params
        -------
        val: Bool
            whether to be verbose

        Returns
        -------
        None
        '''

        self.pp.verbose = val
        self.tl.verbose = val
        self.pl.verbose = val

    def force(self, val):
        '''
        Set whether to force computation for all modules.
        For module-specific behavior, use `c.<module>.force = ...`.
        
        Params
        -------
        val: Union[Bool, NoneType]
            whether to force the computation of values

        Returns
        -------
        None
        '''

        self.pp.force = val
        self.tl.force = val
        self.pl.force = val

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
        return f"{self.__class__.__name__}(root={self._root_dir}, ext='{self._ext}', compression='{self._compression}')"

