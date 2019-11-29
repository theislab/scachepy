# scachepy
Caching extension for **Scanpy** and **Scvelo**. Useful when you want to make sure that your clustering, UMAP, etc. are always *exactly* the same, and enables you to share and load these attributes conveniently. Everything in an AnnData object can be cached using scachepy - if not implemented, we show in the [tutorial notebook](./notebooks/scachepy_tutorial.ipynb) how you can easily set up your own caching function. 

## Installation
```bash
pip install git+https://github.com/theislab/scachepy
```

## Usage
We recommend checking out the [tutorial notebook](./notebooks/scachepy_tutorial.ipynb). In essence, you can:
```python
import scachepy
c = scachepy.Cache(<directory>, separate_dirs=True) 

# set verbosity level
c.verbose(False)
# set whether to recache
c.force(True)

# view available functions
print(list(c.pl))

c.pp.pca(adata)
# also display and save some plots
c.pl.pca(adata)

# remove cached files
c.pp.clear()

# create a copy
adata_copy = c.pp.neighbors(adata, ..., copy=True)

# easily cache fields specified at runtime
# and override global settings
c.tl.louvain(adata, key_added='foo', force=False)
c.tl.louvain(adata, key_added='bar', verbose=True)

# and if you forgot to call the caching version of possibly
# long running function, we got you covered!
scv.tl.velocity_graph(adata, ...)
c.tl.velocity_graph(adata, ..., call=False)
...
```
