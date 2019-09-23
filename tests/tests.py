#!/usr/bin/env python3

from scipy.sparse import issparse

import scanpy as sc
import scvelo as scv
import scachepy

import numpy as np

import unittest
import os
import shutil

# TODO: automatic test creation

_cache_dir = 'cache_test'
shutil.rmtree(_cache_dir, ignore_errors=True)

_cache = scachepy.Cache(_cache_dir, backend='pickle')
_adata_pp = sc.datasets.paul15()
_adata_pp.layers['spliced'] = _adata_pp.X
_adata_pp.layers['unspliced'] = _adata_pp.X / 2

_adata_tl = _adata_pp.copy()
sc.pp.pca(_adata_tl)
sc.pp.neighbors(_adata_tl)


def mtd(data):  # maybe to dense
    return data.todense() if issparse(data) else data


class GeneralTest(unittest.TestCase):
    pass


class PpTests(unittest.TestCase):

    adata = _adata_pp.copy()

    def test_pca(self):

        def pca_asserts(adata):
            self.assertIn('X_pca', adata.obsm)
            self.assertIn('PCs', adata.varm)
            self.assertIn('pca', adata.uns)
            self.assertIn('variance_ratio', adata.uns['pca'])
            self.assertIn('variance', adata.uns['pca'])

        fname = os.path.join(_cache.pp.backend.dir, 'pca' + _cache._ext)
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
        fname = os.path.join(_cache.pp.backend.dir, 'pca_arr' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        arr1 = mtd(_cache.pp.pcarr(self.adata.X))

        self.assertTrue(os.path.isfile(fname))
        self.assertIsInstance(arr1, np.ndarray)

        arr2 = mtd(_cache.pp.pcarr(self.adata.X))

        self.assertTrue(np.array_equal(arr1, arr2))

    def test_expression(self):
        fname = os.path.join(_cache.pp.backend.dir, 'expression' + _cache._ext)
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

        fname = os.path.join(_cache.pp.backend.dir, 'neighs' + _cache._ext)
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
        if not 'X_pca' in self.adata.obsm:
            sc.pp.pca(self.adata)

        if not 'neighbors' in self.adata.uns:
            sc.pp.neighbors(self.adata)

        fname = os.path.join(_cache.pp.backend.dir, 'moments' + _cache._ext)
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


class TlTests(unittest.TestCase):

    adata = _adata_tl.copy()

    def test_louvain(self):
        fname = os.path.join(_cache.tl.backend.dir, 'louvain' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.tl.louvain(self.adata)

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('louvain', self.adata.obs)

        louvain = self.adata.obs['louvain'].copy()
        del self.adata.obs['louvain']

        _cache.tl.louvain(self.adata)

        self.assertIn('louvain', self.adata.obs)
        self.assertTrue(np.array_equal(self.adata.obs['louvain'], louvain))


    def test_embeddings(self):

        def test_embedding(basis):

            def diffmap_check():
                self.assertIn('iroot', self.adata.uns)
                self.assertIn('diffmap_evals', self.adata.uns)

            fname = os.path.join(_cache.tl.backend.dir, basis + _cache._ext)
            key = f'X_{basis}'
            self.assertFalse(os.path.isfile(fname))

            method = getattr(_cache.tl, basis)
            method(self.adata)

            self.assertTrue(os.path.isfile(fname))
            self.assertIn(key, self.adata.obsm)
            if basis == 'diffmap':
                diffmap_check()

            data = self.adata.obsm[key].copy()
            del self.adata.obsm[key]
            if basis == 'diffmap':
                del self.adata.uns['iroot']
                del self.adata.uns['diffmap_evals']

            method(self.adata)

            self.assertIn(key, self.adata.obsm)
            self.assertTrue(np.array_equal(self.adata.obsm[key], data))
            if basis == 'diffmap':
                diffmap_check()

        for basis in ['umap', 'diffmap', 'tsne']:
            test_embedding(basis)

    def test_paga(self):
        fname = os.path.join(_cache.tl.backend.dir, 'paga' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        if 'louvain' not in self.adata.obsm:
            sc.tl.louvain(self.adata)

        _cache.tl.paga(self.adata)

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('paga', self.adata.uns)

        paga = self.adata.uns['paga'].copy()
        del self.adata.uns['paga']

        _cache.tl.paga(self.adata)

        self.assertIn('paga', self.adata.uns)
        self.assertTrue(np.array_equal(mtd(self.adata.uns['paga']['connectivities']),
                                       mtd(paga['connectivities'])))
        self.assertTrue(np.array_equal(mtd(self.adata.uns['paga']['connectivities_tree']),
                                       mtd(paga['connectivities_tree'])))
        self.assertTrue(np.array_equal(self.adata.uns['paga']['groups'], paga['groups']))


    def test_velocity(self):
        fname = os.path.join(_cache.tl.backend.dir, 'velo' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.tl.velocity(self.adata)

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('velocity_gamma', self.adata.var)
        self.assertIn('velocity_r2', self.adata.var)
        self.assertIn('velocity_genes', self.adata.var)
        self.assertIn('velocity', self.adata.layers)

    def test_velocity_graph(self):
        fname = os.path.join(_cache.tl.backend.dir, 'velo_graph' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.tl.velocity_graph(self.adata, vkey='velocity')

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('velocity_graph', self.adata.uns)
        self.assertIn('velocity_graph_neg', self.adata.uns)

        G = self.adata.uns['velocity_graph'].copy()
        G_neg = self.adata.uns['velocity_graph_neg'].copy()
        del self.adata.uns['velocity_graph']
        del self.adata.uns['velocity_graph_neg']

        _cache.tl.velocity_graph(self.adata, vkey='velocity')

        self.assertIn('velocity_graph', self.adata.uns)
        self.assertIn('velocity_graph_neg', self.adata.uns)
        self.assertTrue(np.array_equal(mtd(self.adata.uns['velocity_graph']), mtd(G)))
        self.assertTrue(np.array_equal(mtd(self.adata.uns['velocity_graph_neg']), mtd(G_neg)))

    def test_velocity_embedding(self):
        if 'X_umap' not in self.adata.obsm:
            sc.tl.umap(self.adata)
        if 'velocity' not in self.adata.layers:
            scv.tl.velocity(self.adata)
        if 'velocity_graph' not in self.adata.uns:
            scv.tl.velocity_graph(self.adata)

        fname = os.path.join(_cache.tl.backend.dir, 'velo_emb' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.tl.velocity_embedding(self.adata, basis='umap')

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('velocity_umap', self.adata.obsm)

        emb = self.adata.obsm['velocity_umap'].copy()
        del self.adata.obsm['velocity_umap']
        
        _cache.tl.velocity_embedding(self.adata, basis='umap')

        self.assertIn('velocity_umap', self.adata.obsm)
        self.assertTrue(np.array_equal(mtd(self.adata.obsm['velocity_umap']), mtd(emb)))

    def test_draw_graph(self):
        fname = os.path.join(_cache.tl.backend.dir, 'draw_graph' + _cache._ext)
        self.assertFalse(os.path.isfile(fname))

        _cache.tl.draw_graph(self.adata, layout='fa')

        self.assertTrue(os.path.isfile(fname))
        self.assertIn('X_draw_graph_fa', self.adata.obsm)
        self.assertIn('draw_graph', self.adata.uns)

        emb = self.adata.obsm['X_draw_graph_fa'].copy()
        dg = self.adata.uns['draw_graph'].copy()
        del self.adata.obsm['X_draw_graph_fa']
        del self.adata.uns['draw_graph']

        _cache.tl.draw_graph(self.adata)

        self.assertIn('X_draw_graph_fa', self.adata.obsm)
        self.assertIn('draw_graph', self.adata.uns)
        self.assertTrue(np.array_equal(mtd(self.adata.obsm['X_draw_graph_fa']), mtd(emb)))
        self.assertTrue(np.array_equal(mtd(self.adata.uns['draw_graph']), mtd(dg)))


class PlTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
