# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to reduce dimensionality of gene expression
#             and phenotypes.
# Modules
import numpy as np
import pandas as pd

from ..utils.cache import method_caches


# Classes / functions
class DimensionalityReduction():
    '''Reduce dimensionality of gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Reduce dimensionality of gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset

    @method_caches
    def pca(self,
            n_dims=2,
            transform='log10',
            robust=True,
            random_state=None):
        '''Principal component analysis

        Args:
            n_dims (int): Number of dimensions (2+).
            transform (string or None): Whether to preprocess the data.
            robust (bool): Whether to use Principal Component Pursuit to
                exclude outliers.

        Returns:
            dict of the left eigenvectors (vs), right eigenvectors (us)
                of the singular value decomposition, eigenvalues
                (lambdas), the transform, and the whiten function (for
                plotting).
        '''
        from sklearn.decomposition import PCA

        X = self.dataset.counts.copy()
        pco = self.dataset.counts.pseudocount
        if transform == 'log10':
            X = np.log10(X + pco)
        elif transform == 'log2':
            X = np.log2(X + pco)
        elif transform == 'log':
            X = np.log(X + pco)

        whiten = lambda x: ((x.T - X.mean(axis=1)) / X.std(axis=1, ddof=0)).T
        Xnorm = whiten(X)
        # NaN (e.g. features that do not vary i.e. dropout)
        Xnorm[np.isnan(Xnorm)] = 0

        if robust:
            #from numpy.linalg import matrix_rank
            #rank = matrix_rank(Xnorm.values)

            # Principal Component Pursuit (PSP)
            rpca = _RPCA(Xnorm.values)
            # L is low-rank, S is sparse (outliers)
            L, S = rpca.fit(max_iter=1000, iter_print=None)
            L = pd.DataFrame(L, index=X.index, columns=X.columns)
            whiten = lambda x: ((x.T - L.mean(axis=1)) / L.std(axis=1)).T
            Xnorm = whiten(L)
            Xnorm[np.isnan(Xnorm)] = 0
            #print('rPCA: original rank:', rank,
            #      'reduced rank:', matrix_rank(L),
            #      'sparse rank:', matrix_rank(S))

        pca = PCA(n_components=n_dims, random_state=random_state)
        vs = pd.DataFrame(
                pca.fit_transform(Xnorm.values.T),
                columns=['PC'+str(i+1) for i in range(pca.n_components)],
                index=X.columns)
        us = pd.DataFrame(
                pca.components_,
                index=vs.columns,
                columns=X.index).T

        return {
                'vs': vs,
                'us': us,
                'eigenvalues': pca.explained_variance_ * Xnorm.shape[1],
                'transform': pca.transform,
                'whiten': whiten,
                }

    @method_caches
    def tsne(
            self,
            n_dims=2,
            perplexity=30,
            theta=0.5,
            rand_seed=0,
            **kwargs):
        '''t-SNE algorithm.

        Args:
            n_dims (int): Number of dimensions to use.
            perplexity (float): Perplexity of the algorithm.
            theta (float): A number between 0 and 1. Higher is faster but
                less accurate (via the Barnes-Hut approximation).
            rand_seed (int): Random seed. -1 randomizes each run.
            **kwargs: Named arguments passed to the t-SNE algorithm.

        Returns:
        '''
        # scikit-learn's <0.19 has a bug
        import sklearn
        vmaj, vmin, vrel = sklearn.__version__.split('.')
        if (int(vmaj) == 0) and (int(vmin) < 19):
            from bhtsne import tsne
            use_bhtsne = True
        else:
            from sklearn.manifold import TSNE
            use_bhtsne = False

        n = self.dataset.n_samples
        if(n - 1 < 3 * perplexity):
            raise ValueError('Perplexity too high, reduce to <= {:}'.format((n - 1.)/3))

        X = self.dataset.counts.copy()

        if use_bhtsne:
            # this version does not require pre-whitening
            Y = tsne(
                    data=X.values.T,
                    dimensions=n_dims,
                    perplexity=perplexity,
                    theta=theta,
                    rand_seed=rand_seed,
                    **kwargs)
        else:
            Y = TSNE(
                    n_components=n_dims,
                    perplexity=perplexity,
                    method='barnes_hut' if theta > 0 else 'exact',
                    angle=theta,
                    random_state=rand_seed,
                    ).fit_transform(
                    X.values.T
                    )

        vs = pd.DataFrame(
                Y,
                index=X.columns,
                columns=['dimension '+str(i+1) for i in range(n_dims)])
        return vs


# Supplementary classes
class _RPCA:
    '''from: https://github.com/dganguli/robust-pca'''

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=None):
        ite = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and ite < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            ite += 1
            if iter_print is None:
                continue
            if (ite % iter_print) == 0 or ite == 1 or ite > max_iter or err <= _tol:
                print('iteration: {0}, error: {1}'.format(ite, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):
        import matplotlib.pyplot as plt
        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')
