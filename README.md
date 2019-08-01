# scachepy
Caching extension for Scanpy.

## Installation
```bash
pip install git+https://github.com/michalk8/scachepy
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
