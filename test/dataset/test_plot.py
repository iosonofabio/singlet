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


tol = 3


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


def test_initialize():
    from singlet.dataset.plot import Plot
    return Plot(None)


@pytest.fixture(scope='module')
def plot_empty():
    from singlet.dataset.plot import Plot
    return Plot(None)


def test_sanitize_props(plot_empty):
    kwargs = {'lw': 1, 'kkk': 2}
    plot_empty._sanitize_plot_properties(kwargs)
    assert(kwargs['linewidth'] == 1)
    assert(kwargs['kkk'] == 2)


def test_update_props(plot_empty):
    kwargs = {'lw': 1, 'kkk': 2}
    defaults = {'ls': '-', 'linewidth': 3}
    plot_empty._update_properties(kwargs, defaults)
    assert(kwargs['linewidth'] == 1)
    assert(kwargs['linestyle'] == '-')
    assert(kwargs['kkk'] == 2)


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
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


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
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_plot_coverage_total(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_coverage(
            features='total',
            ax=ax,
            tight_layout=False,
            legend=False,
            )
    fn = 'test_plot_coverage_total.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_plot_coverage_mapped(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_coverage(
            features='mapped',
            ax=ax,
            tight_layout=False,
            legend=False,
            )
    fn = 'test_plot_coverage_mapped.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_plot_coverage_spikeins(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_coverage(
            features='spikeins',
            ax=ax,
            tight_layout=False,
            legend=False,
            )
    fn = 'test_plot_coverage_spikeins.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_plot_coverage_other(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_coverage(
            features='other',
            ax=ax,
            tight_layout=False,
            legend=False,
            )
    fn = 'test_plot_coverage_other.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_plot_scatter_statistics(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.scatter_statistics(
            features='mapped',
            ax=ax,
            tight_layout=False,
            legend=False,
            grid=True,
            )
    fn = 'test_scatter_statistics_mapped.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_distribution_spikeins_violin(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_distributions(
            features='spikeins',
            kind='violin',
            ax=ax,
            tight_layout=False,
            legend=False,
            grid=True,
            )
    fn = 'test_distribution_spikeins.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_distribution_spikeins_box(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_distributions(
            features='spikeins',
            kind='box',
            ax=ax,
            tight_layout=False,
            legend=False,
            grid=True,
            )
    fn = 'test_distribution_spikeins_box.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_distribution_spikeins_swarm(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_distributions(
            features='spikeins',
            kind='swarm',
            ax=ax,
            tight_layout=False,
            legend=False,
            grid=True,
            )
    fn = 'test_distribution_spikeins_swarm.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)


@pytest.mark.skipif(miss_mpl, reason='No maplotlib available')
def test_distribution_spikeins_box_horizontal(ds, vs):
    fig, ax = plt.subplots()
    ds.plot.plot_distributions(
            features='spikeins',
            kind='box',
            orientation='horizontal',
            ax=ax,
            tight_layout=False,
            legend=False,
            grid=False,
            )
    fn = 'test_distribution_spikeins_box_horizontal.png'
    fig.savefig(fdn_tmp+fn)
    plt.close(fig)
    assert(compare_images(fdn_base+fn, fdn_tmp+fn, tol=tol) is None)



