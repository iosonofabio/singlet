[![Build Status](https://travis-ci.org/iosonofabio/singlet.svg?branch=master)](https://travis-ci.org/iosonofabio/singlet)
[![Documentation Status](https://readthedocs.org/projects/singlet/badge/?version=master)](http://singlet.readthedocs.io)

![Logo](docs/_static/logo.png)
# Singlet
Single cell RNA-Seq analysis with quantitative phenotypes.

## Requirements
Python 3.4+ is required. Moreover, you will need:
- [pyyaml](https://pyyaml.org/)
- [numpy](http://www.numpy.org/)
- [scipy](https://www.scipy.org/)
- [pandas](http://pandas.pydata.org/)
- [xarray](http://xarray.pydata.org/en/stable/)
- [scikit-learn](http://scikit-learn.org)

Optional dependencies:
- **plotting**:
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
- **dimensionality reduction/knn graphs**:
  - [numba](https://numba.pydata.org/)
  - [umap](https://github.com/lmcinnes/umap)
  - [lshknn](https://github.com/iosonofabio/lshknn)

Get those from your Linux distribution, `pip`, `conda`, or whatever other source of poison.

Singlet is pure Python for the time being. So it should work on any platform supported by its dependencies, in particular various Linux distributions, recent-ish OSX, and Windows. It is *tested* on Linux and OSX, but if you are a Windows user and know how to use AppVeyor let's set it up!

## Install
To get the latest **stable** version, use pip:
```bash
pip install singlet
```

To get the latest **development** version, clone the git repo and then call:
```bash
python3 setup.py install
```

## Usage example
You can have a look inside the `test` folder for examples. To start using the example dataset:
- Set the environment variable `SINGLET_CONFIG_FILENAME` to the location of the example YAML file
- Open a Python/IPython shell and type:

```python
import matplotlib.pyplot as plt
from singlet.dataset import Dataset
ds = Dataset(
    samplesheet='example_PBMC2',
    counts_table='example_PBMC2',
    featuresheet='example_PBMC2',
    )
ds.counts.log(inplace=True)
ds.samplesheet['cluster'] = ds.cluster.kmeans(axis='samples', n_clusters=5)
vs = ds.dimensionality.tsne(perplexity=15)
ax = ds.plot.scatter_reduced_samples(
    vs,
    color_by='cellType',
    figsize=(5, 4),    
    )
plt.show()
```

This will calculate a t-SNE embedding of the log-transformed features and then show your samples in the reduced space, colored by cluster. It should look more or less like this:

![t-SNE example](docs/_static/example_tsne_2.png)

