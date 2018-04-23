# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to plot gene expression and phenotypes
# Modules
import numpy as np
import pandas as pd
import xarray as xr


# Classes / functions
class Fit():
    '''Fit gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Fit gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    def fit_single(
            self,
            xs,
            ys,
            model,
            method='least-squares',
            handle_nans='ignore',
            **kwargs):
        '''Fit feature expression or phenotypes against other.

        Args:
            xs (list or string): Features and/or phenotypes to use as
                abscissa (independent variable). The string
                'total' means all features including spikeins and other,
                'mapped' means all features excluding spikeins and other,
                'spikeins' means only spikeins, and 'other' means only
                'other' features.
            ys (list or string): Features and/or phenotypes to use as
                ordinate (dependent variable). The string
                'total' means all features including spikeins and other,
                'mapped' means all features excluding spikeins and other,
                'spikeins' means only spikeins, and 'other' means only
                'other' features.
            model (string or function): The model to use for fitting. If a
                string, it must be one of 'linear', 'threshold-linear',
                'logistic'. If a function, it must accept an array
                as first argument (the x) and the parameters as
                additional arguments (like scipy.optimize.curve_fit).
            method (string or function): The minimization algorithm. For now,
                only 'least-squares' is accepted. In this case, the
                goodness of fit is the sum of the squared residues.
            handle_nans (string): How to deal with Not a Numbers, typically
                in the phenotypes. Must be either 'ignore' (default), in
                which case only the non-NaN samples will be used for
                fitting, or 'raise', in which case NaNs will stop the fit.
            **kwargs: Passed to the fit function. For nonlinear
                least-squares, this is scipy.optimize.curve_fit. Linear
                least-squares is analytical so it ignores **kwargs.

        Returns:
            A 3-dimensional xarray with the xs, ys as first two axes. The
            third axis, called 'results', contains the parameters
            and an assessment of the fit quality. If method is
            least-squres, it is the sum of squared residuals.

        NOTE: This function fits every combination of x and y independently,
            interactions are not considered.
        '''
        datad = {'x': {'names': xs, 'data': []},
                 'y': {'names': ys, 'data': []}}

        counts = self.dataset.counts
        sheet = self.dataset.samplesheet.T

        for key, dic in datad.items():
            args = dic['names']
            if args == 'total':
                counts_k = counts
                pheno_k = []
            elif args == 'mapped':
                counts_k = counts.exclude_features(
                        spikeins=True, other=True,
                        errors='ignore')
                pheno_k = []
            elif args == 'spikeins':
                counts_k = counts.get_spikeins()
                pheno_k = []
            elif args == 'other':
                counts_k = counts.get_other_features()
                pheno_k = []
            else:
                counts_k = counts.loc[counts.index.isin(args)]
                pheno_k = sheet.loc[sheet.index.isin(args)]

            if len(counts_k):
                dic['data'].append(counts_k)
            if len(pheno_k):
                dic['data'].append(pheno_k)

            if len(dic['data']) == 0:
                raise ValueError('Data for {:} are empty!'.format(key))
            elif len(dic['data']) == 1:
                dic['data'] = dic['data'][0]
            else:
                dic['data'] = pd.concat(dic['data'], axis=0)

        # Prepare output structure
        if model == 'linear':
            n_pars = 2
            coords_par = ['slope', 'intercept']
        elif model == 'threshold-linear':
            n_pars = 3
            coords_par = ['baseline', 'intercept', 'slope']
        elif model == 'logistic':
            n_pars = 4
            coords_par = ['initial', 'final', 'mid-point', 'slope']
        elif isinstance(model, str):
            raise ValueError('model format not accepted')
        else:
            from inspect import signature
            sig = signature(model)
            n_pars = len(sig.parameters) - 1
            coords_par = [p.name for p in sig.parameters[1:]]

        if method == 'least-squares':
            gof_name = 'sum_squared_residuals'
        else:
            raise ValueError('Only least-squares is implemented')

        n_x = datad['x']['data'].shape[0]
        n_y = datad['y']['data'].shape[0]
        coords_x = datad['x']['data'].index.tolist()
        coords_y = datad['y']['data'].index.tolist()

        res = xr.DataArray(
                np.zeros((n_x, n_y, n_pars + 1)),
                dims=['x', 'y', 'parameters'],
                coords={'x': coords_x, 'y': coords_y,
                        'parameters': coords_par + [gof_name]})

        # Set up fitting environment
        if method == 'least-squares':
            from scipy.optimize import curve_fit
            from scipy.stats import linregress

            if model == 'linear':
                def fun(x, s, i):
                    return i + s * x
            elif model == 'threshold-linear':
                def fun(x, b, i, s):
                    y = i + s * x
                    t = (b - i) / s
                    y[x <= t] = b
                    return y
            elif model == 'logistic':
                def fun(x, i, f, mp, s):
                    return i + (f - i) / (1 + np.exp(-s * (x - mp)))
            else:
                fun = model

            # Fit
            for key_x, data_x in datad['x']['data'].iterrows():
                for key_y, data_y in datad['y']['data'].iterrows():
                    x = data_x.values.astype(float)
                    y = data_y.values.astype(float)

                    # Mask NaNs
                    isnan = np.zeros_like(x, bool)
                    isnan |= np.isnan(x)
                    isnan |= np.isnan(y)
                    if isnan.sum() != 0:
                        if handle_nans == 'raise':
                            raise ValueError('NaNs found in data.')
                        else:
                            x = x[~isnan]
                            y = y[~isnan]

                    if model == 'linear':
                        popt = linregress(x, y)[:2]
                    else:
                        popt, pcov = curve_fit(
                                fun,
                                xdata=x,
                                ydata=y,
                                **kwargs)

                    gof = ((y - fun(x, *popt))**2).sum()

                    res_i = np.append(popt, gof)

                    res.loc[{'x': key_x, 'y': key_y}] = res_i
        else:
            raise ValueError('Only least-squares is implemented')

        return res
