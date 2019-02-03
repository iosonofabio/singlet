# vim: fdm=indent
'''
author:     Fabio Zanini
date:       03/02/19
content:    Test plot functions using some of matplotlib's tooling
'''
import numpy as np
import pandas as pd
import pytest
try:
    import matplotlib
    miss_mpl = False
except ImportError:
    miss_mpl = True
if not miss_mpl:
    from matplotlib.testing.compare import compare_images
    tmp_fdn = '/tmp/'
    matplotlib.use('agg')
    import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(counts_table='example_table_tsv')


@pytest.fixture(scope='module')
def vs():
    return pd.DataFrame(
            data=np.arange(8).reshape((4, 2)),
            columns=['dim1', 'dim2'])



@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_scatter_reduced(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.scatter_reduced_samples(
            vs,
            ax=ax,
            tight_layout=False,
            )
    fn = 'test_scatter_reduced.png'
    fig.savefig(tmp_fdn+fn)
    plt.close(fig)
    assert(compare_images('test/baseline_images/'+fn, tmp_fdn+fn, tol=5) is None)
