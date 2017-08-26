# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to plot gene expression and phenotypes
# Modules
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from ..config import config


try:
    import seaborn as sns
except (ImportError, RuntimeError):
    if 'seaborn_import' not in config['_once_warnings']:
        warnings.warn('Unable to import seaborn: plotting will not work')
        config['_once_warnings'].append('seaborn_import')
    sns = None

try:
    import matplotlib.pyplot as plt
except (ImportError, RuntimeError):
    if 'pyplot_import' not in config['_once_warnings']:
        warnings.warn('Unable to import matplotlib.pyplot: plotting will not work')
        config['_once_warnings'].append('pyplot_import')
    plt = None


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
            legend (bool or dict): If True, call ax.legend(). If a dict, \
                    pass as **kwargs to ax.legend.
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
                ax.legend(**legend)

        if tight_layout:
            if isinstance(tight_layout, dict):
                plt.tight_layout(**tight_layout)
            else:
                plt.tight_layout()

        return ax

    def scatter_statistics(
            self,
            features='mapped',
            x='mean',
            y='cv',
            ax=None,
            tight_layout=True,
            legend=False,
            grid=None,
            **kwargs):
        '''Scatter plot statistics of features.

        Args:
            features (list or string): List of features to plot. The string \
                    'mapped' means everything excluding spikeins and other, \
                    'all' means everything including spikeins and other.
            x (string): Statistics to plot on the x axis.
            y (string): Statistics to plot on the y axis.
            ax (matplotlib.axes.Axes): The axes to plot into. If None \
                    (default), a new figure with one axes is created. ax must \
                    not strictly be a matplotlib class, but it must have \
                    common methods such as 'plot' and 'set'.
            tight_layout (bool or dict): Whether to call \
                    matplotlib.pyplot.tight_layout at the end of the \
                    plotting. If it is a dict, pass it unpacked to that \
                    function.
            legend (bool or dict): If True, call ax.legend(). If a dict, \
                    pass as **kwargs to ax.legend.
            grid (bool or None): Whether to add a grid to the plot. None \
                    defaults to your existing settings.
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
                's': 10,
                'color': 'darkgrey',
                }
        Plot._update_properties(kwargs, defaults)

        counts = self.dataset.counts
        if features == 'total':
            if not counts._otherfeatures.isin(counts.index).all():
                raise ValueError('Other features not found in counts')
            if not counts._spikeins.isin(counts.index).all():
                raise ValueError('Spike-ins not found in counts')
            pass
        elif features == 'mapped':
            counts = counts.exclude_features(
                    spikeins=True, other=True,
                    errors='ignore')
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

        if grid is not None:
            ax.grid(grid)

        ax.set(**ax_props)

        if legend:
            if np.isscalar(legend):
                ax.legend()
            else:
                ax.legend(**legend)

        if tight_layout:
            if isinstance(tight_layout, dict):
                plt.tight_layout(**tight_layout)
            else:
                plt.tight_layout()

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
            if not counts._otherfeatures.isin(counts.index).all():
                raise ValueError('Other features not found in counts')
            if not counts._spikeins.isin(counts.index).all():
                raise ValueError('Spike-ins not found in counts')
            pass
        elif features == 'mapped':
            counts = counts.exclude_features(
                    spikeins=True, other=True,
                    errors='ignore')
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

        # event handling
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
            from matplotlib import path

            # Close the polygon
            xp = polygon[0]['x']
            yp = polygon[0]['y']
            xp0 = polygon[-1]['x']
            yp0 = polygon[-1]['y']
            h = ax.plot([xp0, xp], [yp0, yp], lw=2, color='red')[0]
            polygon[0]['handle'] = h
            fig.canvas.draw()

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
                h = ax.text(
                        x.iloc[ix], y.iloc[ix],
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

        def axes_enter(event):
            cids['press'] = fig.canvas.mpl_connect('button_press_event', onpress)
            cids['release'] = fig.canvas.mpl_connect('button_release_event', onrelease)

        def axes_leave(event):
            fig.canvas.mpl_disconnect(cids['press'])
            fig.canvas.mpl_disconnect(cids['release'])
            cids['press'] = None
            cids['release'] = None
            fig.canvas.draw()

        fig.canvas.mpl_connect('axes_enter_event', axes_enter)
        fig.canvas.mpl_connect('axes_leave_event', axes_leave)

        plt.tight_layout()
        plt.show()

        return selected

    def plot_distributions(
            self,
            features,
            kind='violin',
            ax=None,
            tight_layout=True,
            legend=False,
            orientation='vertical',
            sort=False,
            bottom=0,
            grid=None,
            **kwargs):
        '''Plot distribution of spike-in controls

        Args:
            features (list or string): List of features to plot. If it is the \
                    string 'spikeins', plot all spikeins, if the string \
                    'other', plot other features.
            kind (string): Kind of plot, one of 'violin' (default), 'box', \
                    'swarm'.
            ax (matplotlib.axes.Axes): Axes to plot into. If None (default), \
                    create a new figure and axes.
            tight_layout (bool or dict): Whether to call \
                    matplotlib.pyplot.tight_layout at the end of the \
                    plotting. If it is a dict, pass it unpacked to that \
                    function.
            legend (bool or dict): If True, call ax.legend(). If a dict, \
                    pass as **kwargs to ax.legend.
            orientation (string): 'horizontal' or 'vertical'.
            sort (bool or string): True or 'ascending' sorts the features by \
                    median, 'descending' uses the reverse order.
            bottom (float or string): The value of zero-count features. If \
                    you are using a log axis, you may want to set this to \
                    0.1 or any other small positive number. If a string, it \
                    must be 'pseudocount', then the CountsTable.pseudocount \
                    will be used.
            grid (bool or None): Whether to add a grid to the plot. None \
                    defaults to your existing settings.
            **kwargs: named arguments passed to the plot function.

        Return:
            matplotlib.axes.Axes: The axes with the plot.
        '''
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(18, 8))

        counts = self.dataset.counts

        if features == 'spikeins':
            counts = counts.get_spikeins()
        elif features == 'other':
            counts = counts.get_other_features()
        else:
            counts = counts.loc[features]

        if sort:
            asc = sort != 'descending'
            ind = counts.median(axis=1).sort_values(ascending=asc).index
            counts = counts.loc[ind]

        if bottom == 'pseudocount':
            bottom = counts.pseudocount
        counts = np.maximum(counts, bottom)

        ax_props = {}
        if kind == 'violin':
            defaults = {
                    'scale': 'width',
                    'inner': 'stick',
                    }
            Plot._update_properties(kwargs, defaults)
            sns.violinplot(
                    data=counts.T,
                    orient=orientation,
                    ax=ax,
                    **kwargs)
        elif kind == 'box':
            defaults = {}
            Plot._update_properties(kwargs, defaults)
            sns.boxplot(
                    data=counts.T,
                    orient=orientation,
                    ax=ax,
                    **kwargs)
        elif kind == 'swarm':
            defaults = {}
            Plot._update_properties(kwargs, defaults)
            sns.swarmplot(
                    data=counts.T,
                    orient=orientation,
                    ax=ax,
                    **kwargs)
        else:
            raise ValueError('Plot kind not understood')

        if orientation == 'vertical':
            ax_props['ylim'] = (0.9 * bottom, 1.1 * counts.values.max())
            if not counts._normalized:
                ax_props['ylabel'] = 'Number of reads'
            elif counts._normalized != 'custom':
                ax_props['ylabel'] = counts._normalized.capitalize().replace('_', ' ')
            for label in ax.get_xmajorticklabels():
                label.set_rotation(90)
                label.set_horizontalalignment("center")
            ax.grid(True, 'y')
        elif orientation == 'horizontal':
            ax_props['xlim'] = (0.9 * bottom, 1.1 * counts.values.max())
            if not counts._normalized:
                ax_props['xlabel'] = 'Number of reads'
            elif counts._normalized != 'custom':
                ax_props['xlabel'] = counts._normalized.capitalize().replace('_', ' ')
            ax.grid(True, axis='x')

        ax.set(**ax_props)

        if grid is not None:
            ax.grid(grid)

        if legend:
            if np.isscalar(legend):
                ax.legend()
            else:
                ax.legend(**legend)

        if tight_layout:
            if isinstance(tight_layout, dict):
                plt.tight_layout(**tight_layout)
            else:
                plt.tight_layout()

        return ax

    def scatter_reduced_samples(
            self,
            vectors_reduced,
            color_by=None,
            color_log=None,
            cmap='viridis',
            ax=None,
            tight_layout=True,
            legend=False,
            **kwargs):
        '''Scatter samples after dimensionality reduction.

        Args:
            vectors_reduced (pandas.Dataframe): matrix of coordinates of the \
                    samples after dimensionality reduction. Rows are samples, \
                    columns (typically 2 or 3) are the component in the \
                    low-dimensional embedding.
            color_by (string or None): color sample dots by phenotype or \
                    expression of a certain feature.
            color_log (bool or None): use log of phenotype/expression in the \
                    colormap. Default None only logs expression, but not \
                    phenotypes.
            cmap (string or matplotlib colormap): color map to use for the \
                    sample dots.
            ax (matplotlib.axes.Axes): The axes to plot into. If None \
                    (default), a new figure with one axes is created. ax must \
                    not strictly be a matplotlib class, but it must have \
                    common methods such as 'plot' and 'set'.
            tight_layout (bool or dict): Whether to call \
                    matplotlib.pyplot.tight_layout at the end of the \
                    plotting. If it is a dict, pass it unpacked to that \
                    function.
            legend (bool or dict): If True, call ax.legend(). If a dict, \
                    pass as **kwargs to ax.legend.
            **kwargs: named arguments passed to the plot function.

        Returns:
            matplotlib.axes.Axes with the axes containing the plot.
        '''

        if ax is None:
            new_axes = True
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 8))
        else:
            new_axes = False

        defaults = {
                's': 90,
                }
        Plot._update_properties(kwargs, defaults)

        if color_by is None:
            kwargs['color'] = 'darkgrey'
        else:
            if isinstance(cmap, str):
                cmap = cm.get_cmap(cmap)
            if color_by in self.dataset.samplesheet.columns:
                color_data = self.dataset.samplesheet.loc[:, color_by]
                is_numeric = np.issubdtype(color_data.dtype, np.number)
                color_by_phenotype = True
            elif color_by in self.dataset.counts.index:
                color_data = self.dataset.counts.loc[color_by]
                is_numeric = True
                color_by_phenotype = False
            else:
                raise ValueError(
                'The label '+color_by+' is neither a phenotype nor a feature')

            # Categorical columns get just a list of colors
            if (color_data.dtype.name == 'category') or (not is_numeric):
                cd_unique = list(np.unique(color_data.values))
                c_unique = cmap(np.linspace(0, 1, len(cd_unique)))
                c = c_unique[(cd_unique.index(x) for x in color_data.values)]

            # Non-categorical numeric types are more tricky: check for NaNs
            else:
                if np.isnan(color_data.values).any():
                    unmask = ~np.isnan(color_data.values)
                else:
                    unmask = np.ones(len(color_data), bool)

                cd_min = color_data.values[unmask].min()
                cd_max = color_data.values[unmask].max()

                if color_log:
                    if color_by_phenotype:
                        pc = 0.1 * cd_min
                    else:
                        pc = self.dataset.counts.pseudocount
                    color_data = np.log10(color_data + pc)
                    cd_min = np.log10(cd_min + pc)
                    cd_max = np.log10(cd_max + pc)

                cd_norm = (color_data.values - cd_min) / (cd_max - cd_min)
                c = np.zeros((len(color_data), 4), float)
                c[unmask] = cmap(cd_norm[unmask])
                # Grey-ish semitransparency for NaNs
                c[~unmask] = [0.75] * 3 + [0.3]

            kwargs['c'] = c

        vectors_reduced.plot(
                x=vectors_reduced.columns[0],
                y=vectors_reduced.columns[1],
                kind='scatter',
                ax=ax,
                **kwargs)

        ax.grid(True)

        if legend:
            if np.isscalar(legend):
                ax.legend()
            else:
                ax.legend(**legend)

        if tight_layout:
            if isinstance(tight_layout, dict):
                plt.tight_layout(**tight_layout)
            else:
                plt.tight_layout()

        return ax

    def clustermap(
            self,
            cluster_samples=False,
            cluster_features=False,
            phenotypes_cluster_samples=(),
            phenotypes_cluster_features=(),
            subtract_mean=False,
            divide_std=False,
            orientation='horizontal',
            legend=False,
            **kwargs):
        '''Samples versus features / phenotypes.

        Args:
            cluster_samples (bool or linkage): Whether to cluster samples and \
                    show the dendrogram. Can be either, False, True, or a \
                    linkage from scipy.cluster.hierarchy.linkage.
            cluster_features (bool or linkage): Whether to cluster features \
                    and show the dendrogram. Can be either, False, True, or a \
                    linkage from scipy.cluster.hierarchy.linkage.
            phenotypes_cluster_samples (iterable of strings): Phenotypes to \
                    add to the features for joint clustering of the samples. \
                    If the clustering has been \
                    precomputed including phenotypes and the linkage matrix \
                    is explicitely set as cluster_samples, the *same* \
                    phenotypes must be specified here, in the same order.
            phenotypes_cluster_features (iterable of strings): Phenotypes to \
                    add to the features for joint clustering of the features \
                    and phenotypes. If the clustering has been \
                    precomputed including phenotypes and the linkage matrix \
                    is explicitely set as cluster_features, the *same* \
                    phenotypes must be specified here, in the same order.
            orientation (string): Whether the samples are on the abscissa \
                    ('horizontal') or on the ordinate ('vertical').
            tight_layout (bool or dict): Whether to call \
                    matplotlib.pyplot.tight_layout at the end of the \
                    plotting. If it is a dict, pass it unpacked to that \
                    function.
            legend (bool or dict): If True, call ax.legend(). If a dict, \
                    pass as **kwargs to ax.legend.
            **kwargs: named arguments passed to the plot function.

        Returns:
            A seaborn ClusterGrid instance.
        '''

        if cluster_samples is True:
            cluster_samples = self.dataset.cluster.hierarchical(
                    axis='samples',
                    phenotypes=phenotypes_cluster_samples,
                    )['linkage']
        elif cluster_samples is False:
            cluster_samples = None

        if cluster_features is True:
            cluster_features = self.dataset.cluster.hierarchical(
                    axis='features',
                    phenotypes=phenotypes_cluster_features,
                    )['linkage']
        elif cluster_features is False:
            cluster_features = None

        data = self.dataset.counts.copy()
        # FIXME: add phenotypes

        #if subtract_mean:
        #    data -= data.mean(axis=1)

        #    if divide_std:
        #        data /= (1e-10 + data.std(axis=1))

        if orientation == 'horizontal':
            row_linkage = cluster_features
            col_linkage = cluster_samples
        elif orientation == 'vertical':
            data = data.T
            row_linkage = cluster_samples
            col_linkage = cluster_features
        else:
            raise ValueError('Orientation must be "horizontal" or "vertical".')

        defaults = {
                'yticklabels': True,
                'xticklabels': True,
                'row_linkage': row_linkage,
                'col_linkage': col_linkage,
                }
        Plot._update_properties(kwargs, defaults)

        g = sns.clustermap(
                data=data,
                **kwargs)

        ax = g.ax_heatmap
        for label in ax.get_xmajorticklabels():
            label.set_rotation(90)
            label.set_horizontalalignment("center")
        for label in ax.get_ymajorticklabels():
            label.set_rotation(0)
            label.set_verticalalignment("center")

        if legend:
            # TODO
            pass

        # TODO: reimplement some heuristic tight_layout

        return g
