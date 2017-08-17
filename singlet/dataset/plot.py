# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to plot gene expression and phenotypes
# Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# Classes / functions
class Plot():
    '''Plot gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Plot gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    @staticmethod
    def _update_properties(kwargs, defaults):
        Plot._sanitize_plot_properties(kwargs)
        for key, val in defaults.items():
            if key not in kwargs:
                kwargs[key] = val

    @staticmethod
    def _sanitize_plot_properties(kwargs):
        aliases = {
                'linewidth': 'lw',
                'antialiased': 'aa',
                'color': 'c',
                'linestyle': 'ls',
                'markeredgecolor': 'mec',
                'markeredgewidth': 'mew',
                'markerfacecolor': 'mfc',
                'markerfacecoloralt': 'mfcalt',
                'markersize': 'ms',
                }
        for key, alias in aliases.items():
            if alias in kwargs:
                kwargs[key] = kwargs.pop(alias)

    def plot_coverage(
            self,
            features='total',
            kind='cumulative',
            ax=None,
            tight_layout=True,
            legend=False,
            **kwargs):
        '''Plot number of reads for each sample

        Args:
            features (list or string): Features to sum over. The string \
                    'total' means all features including spikeins and other, \
                    'mapped' means all features excluding spikeins and other.
            kind (string): Kind of plot (default: cumulative distribution).
            ax (matplotlib.axes.Axes): The axes to plot into. If None \
                    (default), a new figure with one axes is created. ax must \
                    not strictly be a matplotlib class, but it must have \
                    common methods such as 'plot' and 'set'.
            tight_layout (bool or dict): Whether to call \
                    matplotlib.pyplot.tight_layout at the end of the \
                    plotting. If it is a dict, pass it unpacked to that \
                    function.
            legend (bool or 2-ple): If True, call ax.legend(). If a 2-ple, \
                    pass the first element as *args  and the second as \
                    **kwargs to ax.legend.
            **kwargs: named arguments passed to the plot function.

        Returns:
            matplotlib.axes.Axes with the axes contaiing the plot.
        '''

        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 8))

        defaults = {
                'linewidth': 2,
                'color': 'darkgrey',
                }
        Plot._update_properties(kwargs, defaults)

        counts = self.dataset.counts
        if features == 'total':
            pass
        elif features == 'mapped':
            counts = counts.exclude_features(spikeins=True, other=True)
        else:
            counts = counts.loc[features]

        if kind == 'cumulative':
            x = counts.values.sum(axis=0)
            x.sort()
            y = 1.0 - np.linspace(0, 1, len(x))
            ax.plot(x, y, **kwargs)
            ax_props = {'ylim': (0, 1)}
        else:
            raise ValueError('Plot kind not understood')

        if not counts._normalized:
            ax_props['xlabel'] = 'Number of reads'
        elif counts._normalized != 'custom':
            ax_props['xlabel'] = counts._normalized.capitalize().replace('_', ' ')

        xmin = np.maximum(x.min(), 0.1)
        xmax = 1.05 * x.max()
        ax_props['xlim'] = (xmin, xmax)
        ax_props['xscale'] = 'log'

        ax.set(**ax_props)

        if legend:
            if np.isscalar(legend):
                ax.legend()
            else:
                legend_args, legend_kwargs = legend
                ax.legend(*legend_args, **legend_kwargs)

        if tight_layout:
            if isinstance(tight_layout, dict):
                plt.tight_layout(**tight_layout)
            else:
                plt.tight_layout()

        return ax
