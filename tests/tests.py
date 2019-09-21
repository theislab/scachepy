#!/usr/bin/env python3

from scipy.sparse import issparse

import scanpy as sc
import scvelo as scv
import scachepy

import numpy as np

import unittest
import os
import shutil


_cache_dir = 'cache_test'
shutil.rmtree(_cache_dir, ignore_errors=True)

_cache = scachepy.Cache(_cache_dir, backend='pickle')
_adata = sc.datasets.paul15()


def mtd(data):  # maybe to dense
    return data.todense() if issparse(data) else data


class ScachepyTestCase(unittest.TestCase):
        
    adata = _adata.copy()


class PpTests(ScachepyTestCase):

    def test_pca(self):

        def pca_asserts(adata):
            self.assertIn('X_pca', adata.obsm)
            self.assertIn('PCs', adata.varm)
            self.assertIn('pca', adata.uns)
            self.assertIn('variance_ratio', adata.uns['pca'])
            self.assertIn('variance', adata.uns['pca'])

        fname = os.path.join(_cache.backend.dir, 'pca' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.pp.pca(self.adata)

        self.assertTrue(os.path.isfile(fname))
        pca_asserts(self.adata)

        # don't really want to order the tests
        del self.adata.obsm['X_pca']
        del self.adata.varm['PCs']
        del self.adata.uns['pca']

        _cache.pp.pca(self.adata)
        pca_asserts(self.adata)

    def test_pcarr(self):
        fname = os.path.join(_cache.backend.dir, 'pca_arr' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        arr1 = mtd(_cache.pp.pcarr(_adata.X))

        self.assertTrue(os.path.isfile(fname))
        self.assertIsInstance(arr1, np.ndarray)

        arr2 = mtd(_cache.pp.pcarr(_adata.X))

        self.assertTrue(np.array_equal(arr1, arr2))

    def test_expression(self):
        fname = os.path.join(_cache.backend.dir, 'expression' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.pp.expression(self.adata)

        self.assertTrue(os.path.isfile(fname))

        X = mtd(self.adata.X.copy())
        self.adata.X = np.zeros_like(X)

        _cache.pp.expression(self.adata)

        self.assertFalse(np.array_equal(mtd(self.adata.X), np.zeros_like(X)))
        self.assertTrue(np.array_equal(mtd(self.adata.X), X))

    def test_neighbors(self):
        if not 'X_pca' in self.adata.obsm:  # test has not yet been executed
            sc.pp.pca(self.adata)  # don't use cache version's here, since they have side effects

        fname = os.path.join(_cache.backend.dir, 'neighs' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.pp.neighbors(self.adata)

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('neighbors', self.adata.uns)

        neighbors = self.adata.uns['neighbors'].copy()
        del self.adata.uns['neighbors']

        _cache.pp.neighbors(self.adata)

        self.assertIn('neighbors', self.adata.uns)
        self.assertTrue(np.array_equal(mtd(self.adata.uns['neighbors']['connectivities']),
                                       mtd(neighbors['connectivities'])))
        self.assertTrue(np.array_equal(mtd(self.adata.uns['neighbors']['distances']),
                                       mtd(neighbors['distances'])))

    def test_moments(self):
        if 'spliced' not in self.adata.layers or \
           'unspliced' not in self.adata.layers:
               # dummy data
               self.adata.layers['spliced'] = self.adata.X
               self.adata.layers['unspliced'] = self.adata.X / 2

        if not 'X_pca' in self.adata.obsm:
            sc.pp.pca(self.adata)

        if not 'neighbors' in self.adata.uns:
            sc.pp.neighbors(self.adata)

        fname = os.path.join(_cache.backend.dir, 'moments' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.pp.moments(self.adata)

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('Mu', self.adata.layers)
        self.assertIn('Ms', self.adata.layers)

        Ms = self.adata.layers['Ms'].copy()
        Mu = self.adata.layers['Mu'].copy()
        del self.adata.layers['Ms']
        del self.adata.layers['Mu']

        _cache.pp.moments(self.adata)

        self.assertIn('Ms', self.adata.layers)
        self.assertIn('Mu', self.adata.layers)
        self.assertTrue(np.array_equal(mtd(self.adata.layers['Ms']),
                                       mtd(self.adata.layers['Ms'])))
        self.assertTrue(np.array_equal(mtd(self.adata.layers['Mu']),
                                       mtd(self.adata.layers['Mu'])))


class TlTests(ScachepyTestCase):
    pass


class PlTests(ScachepyTestCase):
    pass


if __name__ == '__main__':
    unittest.main()
