# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to plot gene expression and phenotypes
# Modules
import numpy as np
import matplotlib as mpl
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
                    'mapped' means all features excluding spikeins and other, \
                    'spikeins' means only spikeins, and 'other' means only \
                    'other' features.
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
            new_axes = True
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 8))
        else:
            new_axes = False

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
        elif features == 'spikeins':
            counts = counts.get_spikeins()
        elif features == 'other':
            counts = counts.get_other_features()
        else:
            counts = counts.loc[features]

        if kind == 'cumulative':
            x = counts.values.sum(axis=0)
            x.sort()
            y = 1.0 - np.linspace(0, 1, len(x))
            ax.plot(x, y, **kwargs)
            ax_props = {
                    'ylim': (-0.05, 1.05),
                    'ylabel': 'Cumulative distribution'}
        else:
            raise ValueError('Plot kind not understood')

        if not counts._normalized:
            ax_props['xlabel'] = 'Number of reads'
        elif counts._normalized != 'custom':
            ax_props['xlabel'] = counts._normalized.capitalize().replace('_', ' ')

        if new_axes:
            xmin = 0.5
            xmax = 1.05 * x.max()
            ax_props['xlim'] = (xmin, xmax)
            ax_props['xscale'] = 'log'
            ax.grid(True)

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

    def gate_features_from_statistics(
            self,
            features='mapped',
            x='mean',
            y='cv',
            **kwargs):
        '''Select features for downstream analysis with a gate.

        Usage: Click with the left mouse button to set the vertices of a \
                polygon. Double left-click closes the shape. Right click \
                resets the plot.

        Args:
            features (list or string): List of features to plot. The string \
                    'mapped' means everything excluding spikeins and other, \
                    'all' means everything including spikeins and other.
            x (string): Statistics to plot on the x axis.
            y (string): Statistics to plot on the y axis.
            **kwargs: named arguments passed to the plot function.

        Returns:
            pd.Index of features within the gate.
        '''
        is_interactive = mpl.is_interactive()
        plt.ioff()

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 8))
        defaults = {
                's': 10,
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

        stats = counts.get_statistics(metrics=(x, y))
        ax_props = {'xlabel': x, 'ylabel': y}
        x = stats.loc[:, x]
        y = stats.loc[:, y]

        ax.scatter(x, y, **kwargs)

        if ax_props['xlabel'] == 'mean':
            xmin = 0.5
            xmax = 1.05 * x.max()
            ax_props['xlim'] = (xmin, xmax)
            ax_props['xscale'] = 'log'
        elif ax_props['ylabel'] == 'mean':
            ymin = 0.5
            ymax = 1.05 * y.max()
            ax_props['ylim'] = (ymin, ymax)
            ax_props['yscale'] = 'log'

        if ax_props['xlabel'] == 'cv':
            xmin = 0
            xmax = 1.05 * x.max()
            ax_props['xlim'] = (xmin, xmax)
        elif ax_props['ylabel'] == 'cv':
            ymin = 0
            ymax = 1.05 * y.max()
            ax_props['ylim'] = (ymin, ymax)

        ax.grid(True)

        ax.set(**ax_props)

        # TODO: event handling
        cids = {'press': None, 'release': None}
        polygon = []
        selected = []
        annotations = []

        def onpress(event):
            if event.button == 1:
                return onpress_left(event)
            elif event.button in (2, 3):
                return onpress_right(event)

        def onpress_left(event):
            xp = event.xdata
            yp = event.ydata
            if len(polygon) == 0:
                h = ax.scatter([xp], [yp], s=50, color='red')
                polygon.append({
                    'x': xp,
                    'y': yp,
                    'handle': h})
            else:
                if len(polygon) == 1:
                    polygon[0]['handle'].remove()
                    polygon[0]['handle'] = None
                xp0 = polygon[-1]['x']
                yp0 = polygon[-1]['y']
                h = ax.plot([xp0, xp], [yp0, yp], lw=2, color='red')[0]
                polygon.append({
                    'x': xp,
                    'y': yp,
                    'handle': h})
            fig.canvas.draw()

            if event.dblclick:
                return ondblclick_left(event)


        def ondblclick_left(event):
            from time import sleep

            # Close the polygon
            xp = polygon[0]['x']
            yp = polygon[0]['y']
            xp0 = polygon[-1]['x']
            yp0 = polygon[-1]['y']
            h = ax.plot([xp0, xp], [yp0, yp], lw=2, color='red')[0]
            polygon[0]['handle'] = h
            fig.canvas.draw()

            # TODO check which features are inside
            from matplotlib import path
            xv = x.values.copy()
            yv = y.values.copy()
            iv = x.index.values
            # A polygon in linear and log is not the same
            xscale = ax.get_xscale()
            yscale = ax.get_yscale()
            if xscale == 'log':
                xv = np.log(xv)
            if yscale == 'log':
                yv = np.log(yv)
            pa = []
            for p in polygon:
                xp = p['x']
                yp = p['y']
                if xscale == 'log':
                    xp = np.log(xp)
                if yscale == 'log':
                    yp = np.log(yp)
                pa.append([xp, yp])
            pa = path.Path(pa)
            points = list(zip(xv, yv))
            ind = pa.contains_points(points).nonzero()[0]
            for ix in ind:
                selected.append(iv[ix])

            # Annotate plot
            for ix in ind:
                h = ax.text(x.iloc[ix], y.iloc[ix],
                        ' '+x.index[ix],
                        ha='left',
                        va='bottom')
                annotations.append(h)
            fig.canvas.draw()

            # Let go of the code flow
            if is_interactive:
                plt.ion()

        def onpress_right(event):
            for elem in polygon:
                h = elem['handle']
                if h is not None:
                    elem['handle'].remove()
            for i in range(len(polygon)):
                del polygon[-1]
            for h in annotations:
                h.remove()
            for i in range(len(annotations)):
                del annotations[-1]
            for i in range(len(selected)):
                del selected[-1]
            fig.canvas.draw()

        def onrelease(event):
            pass
            #for ax, hs in highlights.items():
            #    for h in hs:
            #        h.remove()
            #    highlights[ax] = []
            #fig.canvas.draw()

        def axes_enter(event):
            cids['press'] = fig.canvas.mpl_connect('button_press_event', onpress)
            cids['release'] = fig.canvas.mpl_connect('button_release_event', onrelease)

        def axes_leave(event):
            fig.canvas.mpl_disconnect(cids['press'])
            fig.canvas.mpl_disconnect(cids['release'])
            cids['press'] = None
            cids['release'] = None
            #for ax, hs in highlights.items():
            #    for h in hs:
            #        h.remove()
            #    highlights[ax] = []
            fig.canvas.draw()

        fig.canvas.mpl_connect('axes_enter_event', axes_enter)
        fig.canvas.mpl_connect('axes_leave_event', axes_leave)

        plt.tight_layout()
        plt.show()

        return selected
