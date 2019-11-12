# scachepy
Caching extension for Scanpy. Useful when you want to make sure that your clustering, UMAP, etc. are always *exactly* the same, and enables you to share and load these attributes conveniently. Everything in an AnnData object can be cached using scachepy - if not implemented, we show in the [tutorial notebook](./notebooks/scachepy_tutorial.ipynb) how you can easily set up your own caching function. 

## Installation
```bash
pip install git+https://github.com/theislab/scachepy
```

## Usage
We recommend checking out the [tutorial notebook](./notebooks/scachepy_tutorial.ipynb).
```python
import scachepy
c = scachepy.Cache(<directory>)

c.pp.pca(...)
c.pp.neighbors(...)
...
```
