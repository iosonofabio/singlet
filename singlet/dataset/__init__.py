# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Dataset that combines feature counts with metadata.
# Modules
import numpy as np
import pandas as pd

from ..samplesheet import SampleSheet
from ..counts_table import CountsTable
from ..featuresheet import FeatureSheet


# Classes / functions
class Dataset():
    '''Collection of cells, with feature counts and metadata'''

    def __init__(
            self,
            counts_table=None,
            samplesheet=None,
            featuresheet=None,
            plugins=None):
        '''Collection of cells, with feature counts and metadata

        Args:
            counts_table (string): Name of the counts table (to load from a
                config file) or instance of CountsTable
            samplesheet (string or None): Name of the samplesheet (to load from
                a config file) or instance of SampleSheet
            featuresheet (string or None): Name of the samplesheet (to load
                from a config file) or instance of FeatureSheet
            plugins (dict): Dictionary of classes that take the Dataset
                instance as only argument for __init__, to expand the
                possibilities of Dataset operations.

        NOTE: All samples in the counts_table must also be in the
            samplesheet, but the latter can have additional samples. If
            that is the case, the samplesheet is sliced down to the
            samples present in the counts_table.
        '''
        from .correlations import Correlation
        from .plot import Plot
        from .dimensionality import DimensionalityReduction
        from .cluster import Cluster
        from .fit import Fit
        from .feature_selection import FeatureSelection
        from .graph import Graph

        # In general this class should be used for gene counts and phenotypes,
        # but we have to cover the corner cases that no counts or no phenotypes
        # are provided
        if counts_table is None:
            if samplesheet is None:
                raise ValueError('At least samplesheet or counts_table must be present')
            elif not isinstance(samplesheet, SampleSheet):
                self._samplesheet = SampleSheet.from_sheetname(samplesheet)
            else:
                self._samplesheet = samplesheet
            self._counts = CountsTable(data=[], index=[], columns=self._samplesheet.index)
        else:
            if not isinstance(counts_table, CountsTable):
                counts_table = CountsTable.from_tablename(counts_table)
            self._counts = counts_table

            if samplesheet is None:
                self._samplesheet = SampleSheet(data=[], index=self._counts.columns)
                self._samplesheet.sheetname = None
            elif not isinstance(samplesheet, SampleSheet):
                self._samplesheet = SampleSheet.from_sheetname(samplesheet)
            else:
                self._samplesheet = samplesheet

        if featuresheet is None:
            self._featuresheet = FeatureSheet(data=[], index=self._counts.index)
            self._featuresheet.sheetname = None
        elif not isinstance(featuresheet, FeatureSheet):
            self._featuresheet = FeatureSheet.from_sheetname(featuresheet)
        else:
            self._featuresheet = featuresheet

        # Uniform axes across data and metadata
        assert(self._counts.columns.isin(self._samplesheet.index).all())
        self._samplesheet = self._samplesheet.loc[self._counts.columns]
        # FIXME: this is very slow
        #assert(self._counts.index.isin(self._featuresheet.index).all())
        #self._featuresheet = self._featuresheet.loc[self._counts.index]

        # Inject yourself into counts_table
        self.counts.dataset = self

        # Plugins
        self.correlation = Correlation(self)
        self.plot = Plot(self)
        self.dimensionality = DimensionalityReduction(self)
        self.cluster = Cluster(self)
        self.fit = Fit(self)
        self.feature_selection = FeatureSelection(self)
        self.graph = Graph(self)
        if (plugins is not None) and len(plugins):
            self._plugins = dict(plugins)
            for key, val in plugins:
                setattr(self, key, val(self))
        else:
            self._plugins = {}

    def __str__(self):
        return '{:} with {:} samples and {:} features'.format(
                self.__class__.__name__,
                self.n_samples,
                self.n_features)

    def __repr__(self):
        return '<{:}: {:} samples, {:} features>'.format(
                self.__class__.__name__,
                self.n_samples,
                self.n_features)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        # FIXME: fillna(0) is sloppy but not so bad
        ss = (self._samplesheet.fillna(0) == other._samplesheet.fillna(0)).values.all()
        fs = (self._featuresheet.fillna(0) == other._featuresheet.fillna(0)).values.all()
        ct = (self._counts == other._counts).values.all()
        return ss and fs and ct

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        '''Merge two Datasets.

        For samples with the same names, counts will be added and metadata of
            one of the Datasets used. For new samples, the new counts and
            metadata will be used.

        NOTE: metadata and gene names must be aligned for this operation to
            succeed. If one of the two Datasets has more metadata or
            features than the other, they cannot be added.
        '''
        selfcopy = self.copy()
        selfcopy += other
        return selfcopy

    def __iadd__(self, other):
        '''Merge two Datasets.

        For samples with the same names, counts will be added and metadata of
            one of the Datasets used. For new samples, the new counts and
            metadata will be used.

        NOTE: metadata and gene names must be aligned for this operation to
            succeed. If one of the two Datasets has more metadata or
            features than the other, they cannot be added.
        '''
        if set(self.samplemetadatanames) != set(other.samplemetadatanames):
            raise IndexError('The Datasets have different sample metadata')
        if set(self.featurenames) != set(other.featurenames):
            raise IndexError('The Datasets have different features')
        if set(self.featuremetadatanames) != set(other.featuremetadatanames):
            raise IndexError('The Datasets have different feature metadata')

        snames = self.samplenames
        for samplename, meta in other.samplesheet.iterrows():
            if samplename not in snames:
                self.samplesheet.loc[samplename] = meta
                self.counts.loc[:, samplename] = other.counts.loc[:, samplename]
            else:
                self.counts.loc[:, samplename] += other.counts.loc[:, samplename]

    def split(self, phenotypes, copy=True):
        '''Split Dataset based on one or more categorical phenotypes

        Args:
            phenotypes (string or list of strings): one or more phenotypes to
                use for the split. Unique values of combinations of these
                determine the split Datasets.

        Returns:
            dict of Datasets: the keys are either unique values of the
                phenotype chosen or, if more than one, tuples of unique
                combinations.
        '''
        from itertools import product

        if isinstance(phenotypes, str):
            phenotypes = [phenotypes]

        phenos_uniques = [tuple(set(self.samplesheet.loc[:, p])) for p in phenotypes]
        dss = {}
        for comb in product(*phenos_uniques):
            ind = np.ones(self.n_samples, bool)
            for p, val in zip(phenotypes, comb):
                ind &= self.samplesheet.loc[:, p] == val
            if ind.sum():
                samplesheet = self.samplesheet.loc[ind]
                counts = self.counts.loc[:, ind]

                if copy:
                    samplesheet = samplesheet.copy()
                    counts = counts.copy()

                if len(phenotypes) == 1:
                    label = comb[0]
                else:
                    label = comb

                dss[label] = self.__class__(
                        samplesheet=samplesheet,
                        counts_table=counts,
                        featuresheet=self.featuresheet,
                        )
        return dss

    @property
    def n_samples(self):
        '''Number of samples'''
        if self._samplesheet is not None:
            return self._samplesheet.shape[0]
        elif self._counts is not None:
            return self._counts.shape[1]
        else:
            return 0

    @property
    def n_features(self):
        '''Number of features'''
        if self._counts is not None:
            return self._counts.shape[0]
        else:
            return 0

    @property
    def samplenames(self):
        '''pandas.Index of sample names'''
        return self._samplesheet.index.copy()

    @property
    def featurenames(self):
        '''pandas.Index of feature names'''
        return self._counts.index.copy()

    @property
    def samplemetadatanames(self):
        '''pandas.Index of sample metadata column names'''
        return self._samplesheet.columns.copy()

    @property
    def featuremetadatanames(self):
        '''pandas.Index of feature metadata column names'''
        return self._featuresheet.columns.copy()

    @property
    def samplesheet(self):
        '''Matrix of sample metadata.

        Rows are samples, columns are metadata (e.g. phenotypes).
        '''
        return self._samplesheet

    @samplesheet.setter
    def samplesheet(self, value):
        self._counts = self._counts.loc[:, value.index]
        self._samplesheet = value

    @property
    def counts(self):
        '''Matrix of gene expression counts.

        Rows are features, columns are samples.

        Notice: If you reset this matrix with features that are not in the
            featuresheet or samples that are not in the samplesheet,
            those tables will be reset to empty.
        '''
        return self._counts

    @counts.setter
    def counts(self, value):
        try:
            self._samplesheet = self._samplesheet.loc[value.columns]
        except KeyError:
            self._samplesheet = SampleSheet(data=[], index=value.columns)

        try:
            self._featuresheet = self._featuresheet.loc[value.index]
        except KeyError:
            self._featuresheet = FeatureSheet(data=[], index=value.index)

        self._counts = value
        self._counts.dataset = self

    @property
    def featuresheet(self):
        '''Matrix of feature metadata.

        Rows are features, columns are metadata (e.g. Gene Ontologies).
        '''
        return self._featuresheet

    @featuresheet.setter
    def featuresheet(self, value):
        self._counts = self._counts.loc[value.index, :]
        self._featuresheet = value

    def copy(self):
        '''Copy of the Dataset'''
        return self.__class__(
                counts_table=self._counts.copy(),
                samplesheet=self._samplesheet.copy(),
                featuresheet=self.featuresheet.copy(),
                plugins=self._plugins)

    def query_samples_by_metadata(
            self,
            expression,
            inplace=False,
            local_dict=None):
        '''Select samples based on metadata.

        Args:
            expression (string): An expression compatible with
                pandas.DataFrame.query.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
            local_dict (dict): A dictionary of local variables, useful if you
                are using @var assignments in your expression. By far the
                most common usage of this argument is to set
                local_dict=locals().

        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''
        if inplace:
            self._samplesheet.query(
                    expression, inplace=True,
                    local_dict=local_dict)
            self._counts = self._counts.loc[:, self._samplesheet.index]
        else:
            samplesheet = self._samplesheet.query(
                    expression, inplace=False,
                    local_dict=local_dict)
            counts_table = self._counts.loc[:, samplesheet.index].copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table,
                    featuresheet=self._featuresheet.copy(),
                    )

    def query_features_by_metadata(
            self,
            expression,
            inplace=False,
            local_dict=None):
        '''Select features based on metadata.

        Args:
            expression (string): An expression compatible with
                pandas.DataFrame.query.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
            local_dict (dict): A dictionary of local variables, useful if you
                are using @var assignments in your expression. By far the
                most common usage of this argument is to set
                local_dict=locals().
        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''
        if inplace:
            self._featuresheet.query(
                    expression, inplace=True,
                    local_dict=local_dict)
            self._counts = self._counts.loc[self._featuresheet.index]
        else:
            featuresheet = self._featuresheet.query(
                    expression, inplace=False,
                    local_dict=local_dict)
            counts_table = self._counts.loc[featuresheet.index].copy()
            samplesheet = self._samplesheet.copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table,
                    featuresheet=featuresheet)

    def query_samples_by_counts(
            self, expression, inplace=False,
            local_dict=None):
        '''Select samples based on gene expression.

        Args:
            expression (string): An expression compatible with
                pandas.DataFrame.query.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
            local_dict (dict): A dictionary of local variables, useful if you
                are using @var assignments in your expression. By far the most
                common usage of this argument is to set local_dict=locals().
        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''
        counts = self._counts.copy()
        drop = []
        if ('total' in expression) and ('total' not in counts.index):
            counts.loc['total'] = counts.sum(axis=0)
            drop.append('total')
        if ('mapped' in expression) and ('mapped' not in counts.index):
            counts.loc['mapped'] = counts.exclude_features(spikeins=True, other=True).sum(axis=0)
            drop.append('mapped')

        counts_table = counts.T.query(
                expression, inplace=False,
                local_dict=local_dict).T
        if drop:
            counts_table.drop(drop, axis=0, inplace=True)

        if inplace:
            self.counts = counts_table
        else:
            samplesheet = self._samplesheet.loc[counts_table.columns].copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table,
                    featuresheet=self.featuresheet.copy())

    def query_features_by_counts(
            self, expression, inplace=False,
            local_dict=None):
        '''Select features based on their expression.

        Args:
            expression (string): An expression compatible with
                pandas.DataFrame.query.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
            local_dict (dict): A dictionary of local variables, useful if you
                are using @var assignments in your expression. By far the
                most common usage of this argument is to set
                local_dict=locals().
        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''
        if inplace:
            self._counts.query(
                    expression, inplace=True,
                    local_dict=local_dict)
            self._featuresheet = self._featuresheet.loc[self._counts.index]
        else:
            counts_table = self._counts.query(
                    expression, inplace=False,
                    local_dict=local_dict)
            samplesheet = self._samplesheet.copy()
            featuresheet = self._featuresheet.loc[counts_table.index].copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table,
                    featuresheet=featuresheet)

    def rename(
            self,
            axis,
            column,
            inplace=False):
        '''Rename samples or features

        Args:
            axis (string): Must be 'samples' or 'features'.
            column (string): Must be a column of the samplesheet (for
                axis='samples') or of the featuresheet (for axis='features')
                with unique names of samples or features.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
        '''
        if axis not in ('samples', 'features'):
            raise ValueError('axis must be "samples" or "features"')

        if inplace:
            if axis == 'samples':
                self._samplesheet.index = self._samplesheet.loc[:, column]
                self._counts.columns = self._samplesheet.loc[:, column]
            else:
                self._featuresheet.index = self._featuresheet.loc[:, column]
                self._counts.index = self._featuresheet.loc[:, column]
        else:
            other = self.copy()
            other.rename(
                    axis=axis,
                    column=column,
                    inplace=True)
            return other

    def compare(
            self,
            other,
            features='mapped',
            phenotypes=(),
            method='kolmogorov-smirnov'):
        '''Statistically compare with another Dataset.

        Args:
            other (Dataset): The Dataset to compare with.
            features (list, string, or None): Features to compare. The string
                'total' means all features including spikeins and other,
                'mapped' means all features excluding spikeins and other,
                'spikeins' means only spikeins, and 'other' means only
                'other' features. If empty list or None, do not compare
                features (useful for phenotypic comparison).
            phenotypes (list of strings): Phenotypes to compare.
            method (string or function): Statistical test to use for the
                comparison. If a string it must be one of
                'kolmogorov-smirnov' or 'mann-whitney'. If a function, it
                must accept two arrays as arguments (one for each
                dataset, running over the samples) and return a P-value
                for the comparison.
        Return:
            A pandas.DataFrame containing the P-values of the comparisons for
                all features and phenotypes.
        '''

        pvalues = []
        if features:
            counts = self.counts
            counts_other = other.counts
            if features == 'total':
                pass
            elif features == 'mapped':
                counts = counts.exclude_features(
                        spikeins=True, other=True, errors='ignore')
                counts_other = counts_other.exclude_features(
                        spikeins=True, other=True, errors='ignore')
            elif features == 'spikeins':
                counts = counts.get_spikeins()
                counts_other = counts_other.get_spikeins()
            elif features == 'other':
                counts = counts.get_other_features()
                counts_other = counts_other.get_other_features()
            else:
                counts = counts.loc[features]
                counts_other = counts_other.loc[features]

            if method == 'kolmogorov-smirnov':
                from scipy.stats import ks_2samp
                for (fea, co1), (_, co2) in zip(
                        counts.iterrows(),
                        counts_other.iterrows()):
                    pvalues.append([fea, ks_2samp(co1.values, co2.values)[1]])

            elif method == 'mann-whitney':
                from scipy.stats import mannwhitneyu
                for (fea, co1), (_, co2) in zip(
                        counts.iterrows(),
                        counts_other.iterrows()):
                    # Mann-Whitney U has issues with ties, so we handle a few
                    # corner cases separately
                    is_degenerate = False
                    # 1. no samples
                    if (len(co1.values) == 0) or (len(co2.values) == 0):
                        is_degenerate = True
                    # 2. if there is only one value over the board
                    tmp1 = np.unique(co1.values)
                    tmp2 = np.unique(co2.values)
                    if ((len(tmp1) == 1) and (len(tmp2) == 1) and
                       (tmp1[0] == tmp2[0])):
                        is_degenerate = True
                    # 3. if the arrays are the exact same
                    elif ((len(co1) == len(co2)) and
                          (np.sort(co1.values) == np.sort(co2.values)).all()):
                        is_degenerate = True
                    if is_degenerate:
                        pvalues.append([fea, 1])
                        continue
                    pvalues.append([fea, mannwhitneyu(
                        co1.values, co2.values,
                        alternative='two-sided')[1]])
            else:
                for (fea, co1) in counts.iterrows():
                    co2 = counts_other.loc[fea]
                    pvalues.append([fea, method(co1.values, co2.values)])

        if phenotypes:
            pheno = self.samplesheet.loc[:, phenotypes].T
            pheno_other = other.samplesheet.loc[:, phenotypes].T

            if method == 'kolmogorov-smirnov':
                from scipy.stats import ks_2samp
                for phe, val1 in pheno.iterrows():
                    val2 = pheno_other.loc[phe]
                    pvalues.append([phe, ks_2samp(val1.values, val2.values)[1]])
            elif method == 'mann-whitney':
                from scipy.stats import mannwhitneyu
                for phe, val1 in pheno.iterrows():
                    val2 = pheno_other.loc[phe]
                    # Mann-Whitney U has issues with ties
                    is_degenerate = False
                    if ((len(np.unique(val1.values)) == 1) or
                       (len(np.unique(val2.values)) == 1)):
                        is_degenerate = True
                    elif ((len(val1) == len(val2)) and
                          (np.sort(val1.values) == np.sort(val2.values)).all()):
                        is_degenerate = True
                    if is_degenerate:
                        pvalues.append([phe, 1])
                        continue
                    pvalues.append([phe, mannwhitneyu(
                        val1.values, val2.values,
                        alternative='two-sided')[1]])
            else:
                for phe, val1 in pheno.iterrows():
                    val2 = pheno_other.loc[phe]
                    pvalues.append([phe, method(val1.values, val2.values)])

        df = pd.DataFrame(pvalues, columns=['name', 'P-value'])
        df.set_index('name', drop=True, inplace=True)
        return df

    def bootstrap(self, groupby=None):
        '''Resample with replacement, aka bootstrap dataset

            Args:
                groupby (str or list of str or None): If None, bootstrap random
                samples disregarding sample metadata. If a string or a list of
                strings, boostrap over groups of samples with consistent
                entries for that/those columns.

            Returns:
                A Dataset with the resampled samples.
        '''
        n = self.n_samples
        if groupby is None:
            ind = np.random.randint(n, size=n)
        else:
            meta = self.samplesheet.loc[:, groupby]
            meta_unique = meta.drop_duplicates().values
            n_groups = meta_unique.shape[0]
            ind_groups = np.random.randint(n_groups, size=n_groups)
            ind = []
            for i in ind_groups:
                indi = (meta == meta_unique[i]).values
                if indi.ndim > 1:
                    indi = indi.all(axis=1)
                indi = indi.nonzero()[0]
                ind.extend(list(indi))
            ind = np.array(ind)

        snames = self.samplenames
        from collections import Counter
        tmp = Counter()
        index = []
        for i in ind:
            tmp[i] += 1
            index.append(snames[i]+'--sampling_'+str(tmp[i]))
        index = pd.Index(index, name=self.samplenames.name)

        ss = self.samplesheet.__class__(
            self.samplesheet.values[ind],
            index=index,
            columns=self.samplesheet.columns,
            )
        ct = self.counts.__class__(
            self.counts.values[:, ind],
            index=self.counts.index,
            columns=index,
            )
        fs = self.featuresheet.copy()
        plugins = {key: val.__class__ for key, val in self._plugins}

        return self.__class__(
            counts_table=ct,
            samplesheet=ss,
            featuresheet=fs,
            plugins=plugins,
            )

    def average(self, axis, column):
        '''Average samples or features based on metadata


        Args:
            axis (string): Must be 'samples' or 'features'.
            column (string): Must be a column of the samplesheet (for
                axis='samples') or of the featuresheet (for axis='features').
                Samples or features with a common value in this column are
                averaged over.
        Returns:
            A Dataset with the averaged counts.

        Note: if you average over samples, you get an empty samplesheet.
        Simlarly, if you average over features, you get an empty featuresheet.

        '''
        if axis not in ('samples', 'features'):
            raise ValueError('axis must be "samples" or "features"')

        if axis == 'samples':
            if column not in self.samplesheet.columns:
                raise ValueError(
                    '{:} is not a column of the SampleSheet'.format(column))

            vals = pd.Index(np.unique(self.samplesheet[column]), name=column)
            n_conditions = len(vals)
            counts = np.zeros(
                    (self.n_features, n_conditions),
                    dtype=self.counts.values.dtype)
            for i, val in enumerate(vals):
                ind = self.samplesheet[column] == val
                counts[:, i] = self.counts.loc[:, ind].values.mean(axis=1)
            counts = self.counts.__class__(
                    pd.DataFrame(
                        counts,
                        index=self.counts.index,
                        columns=vals))

            featuresheet = self._featuresheet.copy()

            return Dataset(
                    counts_table=counts,
                    featuresheet=featuresheet)

        elif axis == 'features':
            if column not in self.featuresheet.columns:
                raise ValueError(
                    '{:} is not a column of the FeatureSheet'.format(column))

            vals = pd.Index(np.unique(self.featuresheet[column]), name=column)
            n_conditions = len(vals)
            counts = np.zeros(
                    (n_conditions, self.n_samples),
                    dtype=self.counts.values.dtype)
            for i, val in enumerate(vals):
                ind = self.featuresheet[column] == val
                counts[i] = self.counts.loc[ind].values.mean(axis=0)
            counts = self.counts.__class__(
                    pd.DataFrame(
                        counts,
                        index=vals,
                        columns=self.counts.columns))

            samplesheet = self._samplesheet.copy()

            return Dataset(
                    counts_table=counts,
                    samplesheet=samplesheet)
