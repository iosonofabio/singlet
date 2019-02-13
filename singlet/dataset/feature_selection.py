# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to feature selection
# Modules
import numpy as np
import pandas as pd

from .plugins import Plugin


# Classes / functions
class FeatureSelection(Plugin):
    '''Plot gene expression and phenotype in single cells'''

    def unique(
            self,
            inplace=False):
        '''Select features with unique ids

        Args:
            inplace (bool): Whether to change the feature list in place.

        Returns:
            pd.Index of selected features if not inplace, else None.
        '''
        from collections import Counter
        d = Counter(self.dataset._featuresheet.index)
        features = [f for f, count in d.items() if count == 1]

        if inplace:
            self.dataset.counts = self.dataset._counts.loc[features]
        else:
            return pd.Index(features, name=self.dataset._featuresheet.index.name)

    def expressed(
            self,
            n_samples,
            exp_min,
            inplace=False):
        '''Select features that are expressed in at least some samples.

        Args:
            n_samples (int): Minimum number of samples the features should be
                expressed in.
            exp_min (float): Minimum level of expression of the features.
            inplace (bool): Whether to change the feature list in place.

        Returns:
            pd.Index of selected features if not inplace, else None.
        '''
        ind = (self.dataset.counts >= exp_min).sum(axis=1) >= n_samples
        if inplace:
            self.dataset.counts = self.dataset.counts.loc[ind]
        else:
            return self.dataset.featurenames[ind]

    def overdispersed_strata(
            self, bins=10,
            n_features_per_stratum=50,
            inplace=False):
        '''Select overdispersed features in strata of increasing expression.

        Args:
            bins (int or list): Bin edges determining the strata. If this is
                a number, split the expression in this many equally spaced bins
                between minimal and maximal expression.
            n_features_per_stratum (int): Number of features per stratum to
                select.

        Returns:
            pd.Index of selected features if not inplace, else None.

        Notice that the number of selected features may be smaller than
        expected if some strata have no dispersion (e.g. only dropouts).
        Because of this, it is recommended you restrict the counts to
        expressed features before using this function.
        '''

        stats = self.dataset.counts.get_statistics(metrics=('mean', 'cv'))
        mean = stats.loc[:, 'mean']

        if np.isscalar(bins):
            exp_min, exp_max = mean.values.min(), mean.values.max()
            bins = np.linspace(exp_min, exp_max, bins+1)

        features = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                cvi = stats.loc[mean >= bins[i], 'cv']
            else:
                cvi = stats.loc[(mean >= bins[i]) & (mean < bins[i+1]), 'cv']
            features.append(cvi.nlargest(n_features_per_stratum).index)
        features = pd.Index(np.concatenate(features), name=cvi.index.name)

        if inplace:
            self.dataset.counts = self.dataset.counts.loc[features]
        else:
            return features

    def sam(self, k=None, distance='correlation', *args, **kwargs):
        '''Calculate feature weights via self-assembling manifolds

        Args:
            k (int or None): The number of nearest neighbors for each sample
            distance (str): The distance matrix
            *args, **kwargs: Arguments to SAM.run

        Returns:
            SAM instance containing SAM.output_vars['gene_weights']

        See also: https://github.com/atarashansky/self-assembling-manifold
        '''
        import SAM

        sam = SAM.SAM(
                counts=self.dataset.counts.T,
                k=k,
                distance=distance)
        sam.run(*args, **kwargs)

        return sam

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
        import matplotlib as mpl
        import matplotlib.pyplot as plt

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
