#!/usr/bin/env python3

from scipy.sparse import issparse

import scanpy as sc
import scvelo as scv
import anndata
import scachepy

import numpy as np

import unittest
import shutil
import os

# TODO: automatic test creation

_cache_dir = 'cache_test'
shutil.rmtree(_cache_dir, ignore_errors=True)

_cache = scachepy.Cache(_cache_dir, backend='pickle', separate_dirs=False)
_adata_pp = sc.datasets.paul15()
_adata_pp.layers['spliced'] = _adata_pp.X
_adata_pp.layers['unspliced'] = _adata_pp.X / 2

_adata_tl = _adata_pp.copy()
sc.pp.pca(_adata_tl)
sc.pp.neighbors(_adata_tl)


def mtd(data):  # maybe to dense
    return data.todense() if issparse(data) else data

# this could be done more nicely
class GeneralTests(unittest.TestCase):

    adata_pp = _adata_pp.copy()
    adata_tl = _adata_tl.copy()

    def test_invalid_backend(self):
        with self.assertRaises(ValueError):
            _ = scachepy.Cache('foo', backend='bar')

    def test_extension(self):
        c = scachepy.Cache('foo', separate_dirs=False, ext='bar')
        c.pp.pca(self.adata_pp, fname='pca')
        self.assertTrue(os.path.exists('foo/pca.bar'))
        shutil.rmtree('foo')

    def test_extension_override(self):
        c = scachepy.Cache('foo', separate_dirs=False, ext='.bar')
        c.pp.pca(self.adata_pp, fname='pca.foo')
        self.assertTrue(os.path.exists('foo/pca.foo.bar'))
        shutil.rmtree('foo')

    def test_fname(self):
        c = scachepy.Cache('foo', separate_dirs=False)
        c.pp.pca(self.adata_pp, fname='bar.pickle')
        self.assertTrue(os.path.exists('foo/bar.pickle'))
        shutil.rmtree('foo')

    def test_fname_sep(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.pp.pca(self.adata_pp, fname='bar.pickle')
        self.assertTrue(os.path.exists('foo/pp/bar.pickle'))
        shutil.rmtree('foo')

    def test_copy_save(self):
        c = scachepy.Cache('foo', separate_dirs=False)
        adata = c.pp.neighbors(self.adata_pp, copy=True)
        self.assertTrue(isinstance(adata, anndata.AnnData))
        self.assertTrue('neighbors' in adata.uns)
        self.assertFalse('neighbors' in self.adata_pp.uns)
        self.assertTrue(os.path.exists('foo/neighs.pickle'))
        shutil.rmtree('foo')

    def test_copy_load(self):
        c = scachepy.Cache('foo', separate_dirs=False)
        c.pp.neighbors(self.adata_pp)
        del self.adata_pp.uns['neighbors']
        adata = c.pp.neighbors(self.adata_pp, copy=True)
        self.assertTrue(isinstance(adata, anndata.AnnData))
        self.assertTrue('neighbors' in adata.uns)
        self.assertFalse('neighbors' in self.adata_pp.uns)
        shutil.rmtree('foo')

    def test_callback(self):
        
        def callback(adata, *args, **kwargs):
            raise RuntimeError('this should have happened')

        c = scachepy.Cache('foo', separate_dirs=False)
        with self.assertRaisesRegex(RuntimeError, 'this should have happened'):
            c.pp.pca(callback, self.adata_pp)
        shutil.rmtree('foo')

    def test_force(self):

        def callback(adata, *args, **kwargs):
            nonlocal sentinel
            sc.pp.pca(adata)
            sentinel = True

        c = scachepy.Cache('foo', separate_dirs=False, ext='.bar')
        sentinel = False
        c.pp.pca(callback, self.adata_pp, force=True)
        self.assertTrue(sentinel)
        shutil.rmtree('foo')

    def test_not_call_key(self):

        def callback(adata, *args, **kwargs):
            nonlocal sentinel
            sc.pp.pca(adata)
            sentinel = True

        c = scachepy.Cache('foo', separate_dirs=False, ext='.bar')
        sentinel = False
        # adata_tl already has pca
        c.pp.pca(callback, self.adata_tl, call=False)
        self.assertFalse(sentinel)
        shutil.rmtree('foo')

    def test_force_not_call(self):

        def callback(adata, *args, **kwargs):
            nonlocal sentinel
            sc.pp.pca(adata)
            sentinel = True

        c = scachepy.Cache('foo', separate_dirs=False, ext='.bar')
        sentinel = False
        # adata_tl already has pca
        c.pp.pca(callback, self.adata_tl, call=False, force=True)
        self.assertTrue(sentinel)
        shutil.rmtree('foo')

    def test_not_call_no_key(self):

        def callback(adata, *args, **kwargs):
            nonlocal sentinel
            scv.pp.moments(adata)
            sentinel = True

        c = scachepy.Cache('foo', separate_dirs=False, ext='.bar')
        sentinel = False
        # adata_pp already has no moments
        with self.assertRaises(AssertionError):
            c.pp.moments(callback, self.adata_pp, call=False)
        shutil.rmtree('foo')

    def test_backend_dir_set(self):
        c = scachepy.Cache('foo', separate_dirs=False)
        self.assertTrue(str(c.root_dir) == 'foo')
        shutil.rmtree('foo')

    def test_root_dir_set(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        with self.assertRaises(RuntimeError):
            c.root_dir = 'bar'

    def test_backend_sep_set_dir(self):
        c = scachepy.Cache('foo', separate_dirs=False)
        c.pp.backend.dir = 'bar'
        self.assertTrue(str(c.pp.backend.dir) == 'bar')
        self.assertTrue(str(c.tl.backend.dir) == 'bar')
        self.assertTrue(str(c.pl.backend.dir) == 'bar')
        shutil.rmtree('foo')

    def test_backend_sep_set_dir(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.pp.backend.dir = 'bar'
        c.tl.backend.dir = 'baz'
        c.pl.backend.dir = 'quux'
        print(c.pp.backend.dir)
        self.assertTrue(str(c.pp.backend.dir) == 'foo/bar')
        self.assertTrue(str(c.tl.backend.dir) == 'foo/baz')
        self.assertTrue(str(c.pl.backend.dir) == 'foo/quux')
        shutil.rmtree('foo')

    def test_pp_clear(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.pp.pca(self.adata_pp)
        self.assertTrue(len(os.listdir(c.root_dir / 'pp')) == 1)
        c.pp.clear()
        self.assertTrue(len(os.listdir(c.root_dir / 'pp')) == 0)
        shutil.rmtree('foo')

    def test_tl_clear(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.tl.louvain(self.adata_tl)
        self.assertTrue(len(os.listdir(c.root_dir / 'tl')) == 1)
        c.tl.clear()
        self.assertTrue(len(os.listdir(c.root_dir / 'tl')) == 0)
        shutil.rmtree('foo')

    def test_pl_clear(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.pl.pca(self.adata_pp)
        self.assertTrue(len(os.listdir(c.root_dir / 'pl')) == 1)
        c.pl.clear()
        self.assertTrue(len(os.listdir(c.root_dir / 'pl')) == 0)
        shutil.rmtree('foo')

    def test_clear_all_sep(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.pp.pca(self.adata_tl)
        c.tl.louvain(self.adata_tl)
        c.pl.pca(self.adata_tl)
        for dir in ['pp', 'tl', 'pl']:
            self.assertTrue(len(os.listdir(c.root_dir / dir)) == 1)
        c.clear()
        for dir in ['pp', 'tl', 'pl']:
            self.assertTrue(len(os.listdir(c.root_dir / dir)) == 0)
        shutil.rmtree('foo')

    def test_clear_all_no_sep(self):
        c = scachepy.Cache('foo', separate_dirs=False)
        c.pp.pca(self.adata_tl)
        c.tl.louvain(self.adata_tl)
        c.pl.pca(self.adata_tl)
        print(os.listdir(c.root_dir))
        self.assertTrue(len(os.listdir(c.root_dir)) == 3)
        c.clear()
        self.assertTrue(len(os.listdir(c.root_dir)) == 0)
        shutil.rmtree('foo')

    def test_clear_no_remove(self):
        c = scachepy.Cache('foo', separate_dirs=True)
        c.pp.pca(self.adata_pp)
        self.assertTrue(len(os.listdir(c.root_dir / 'pp')) == 1)
        c.tl.clear()
        c.pl.clear()
        self.assertTrue(len(os.listdir(c.root_dir / 'pp')) == 1)
        shutil.rmtree('foo')


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


# TODO: this really needs to be automated
class PlTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
