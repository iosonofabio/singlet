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
    fdn_tmp = '/tmp/'
    fdn_base = 'test/baseline_images/'
    matplotlib.use('agg')
    import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def ds():
    from singlet.dataset import Dataset
    return Dataset(
            samplesheet='example_sheet_tsv',
            counts_table='example_table_tsv')


@pytest.fixture(scope='module')
def vs(ds):
    return pd.DataFrame(
            data=np.arange(2 * ds.n_samples).reshape((ds.n_samples, 2)),
            columns=['dim1', 'dim2'],
            index=ds.samplenames)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_scatter_reduced(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.scatter_reduced_samples(
            vs,
            ax=ax,
            tight_layout=False,
            )
    fn = 'test_scatter_reduced.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=5) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_scatter_reduced_colorby(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.scatter_reduced_samples(
            vs,
            ax=ax,
            tight_layout=False,
            color_by='quantitative_phenotype_1_[A.U.]')
    fn = 'test_scatter_reduced_colorby_qp.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=5) is None)
