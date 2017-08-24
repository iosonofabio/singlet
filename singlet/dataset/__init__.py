# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Dataset that combines feature counts with metadata.
# Modules
import numpy as np


# Classes / functions
class Dataset():
    '''Collection of cells, with feature counts and metadata'''

    def __init__(self, samplesheet, counts_table):
        '''Collection of cells, with feature counts and metadata

        Args:
            samplesheet (string): Name of the samplesheet (to load from a \
                    config file) or instance of SampleSheet
            counts_table (string): Name of the counts table (to load from a \
                    config file) or instance of CountsTable

        NOTE: All samples in the counts_table must also be in the \
                samplesheet, but the latter can have additional samples. If \
                that is the case, the samplesheet is sliced down to the \
                samples present in the counts_table.
        '''
        from ..samplesheet import SampleSheet
        from ..counts_table import CountsTable
        from .correlations import Correlation
        from .plot import Plot
        from .dimensionality import DimensionalityReduction
        from .cluster import Cluster

        if not isinstance(samplesheet, SampleSheet):
            samplesheet = SampleSheet.from_sheetname(samplesheet)
        self._samplesheet = samplesheet

        if not isinstance(counts_table, CountsTable):
            counts_table = CountsTable.from_tablename(counts_table)
        self._counts = counts_table

        assert(self._counts.columns.isin(self._samplesheet.index).all())
        self._samplesheet = self._samplesheet.loc[self._counts.columns]

        # Plugins
        self.correlation = Correlation(self)
        self.plot = Plot(self)
        self.dimensionality = DimensionalityReduction(self)
        self.cluster = Cluster(self)

    def __str__(self):
        return '{:} with {:} samples and {:} features'.format(
                self.__class__.__name__,
                self.n_samples,
                self.n_features)

    def __repr__(self):
        return '{:}("{:}", "{:}")'.format(
                self.__class__.__name__,
                self._samplesheet.sheetname,
                self._counts.name)

    def __eq__(self, other):
        if type(other) is not type(self):
            return False
        # FIXME: fillna(0) is sloppy but not so bad
        ss = (self._samplesheet.fillna(0) == other._samplesheet.fillna(0)).values.all()
        ct = (self._counts == other._counts).values.all()
        return ss and ct

    def __ne__(self, other):
        return not self == other

    def __add__(self, other):
        '''Merge two Datasets.

        For samples with the same names, counts will be added and metadata of \
                one of the Datasets used. For new samples, the new counts and \
                metadata will be used.

        NOTE: metadata and gene names must be aligned for this operation to \
                succeed. If one of the two Datasets has more metadata or \
                features than the other, they cannot be added.
        '''
        selfcopy = self.copy()
        selfcopy += other
        return selfcopy

    def __iadd__(self, other):
        '''Merge two Datasets.

        For samples with the same names, counts will be added and metadata of \
                one of the Datasets used. For new samples, the new counts and \
                metadata will be used.

        NOTE: metadata and gene names must be aligned for this operation to \
                succeed. If one of the two Datasets has more metadata or \
                features than the other, they cannot be added.
        '''
        if set(self.metadatanames) != set(other.metadatanames):
            raise IndexError('The Datasets have different metadata')
        if set(self.featurenames) != set(other.featurenames):
            raise IndexError('The Datasets have different features')

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
            phenotypes (string or list of strings): one or more phenotypes to \
                    use for the split. Unique values of combinations of these \
                    determine the split Datasets.

        Returns:
            dict of Datasets: the keys are either unique values of the \
                    phenotype chosen or, if more than one, tuples of unique \
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
                        )
        return dss

    @property
    def n_samples(self):
        '''Number of samples'''
        return self._samplesheet.shape[0]

    @property
    def n_features(self):
        '''Number of features'''
        return self._counts.shape[0]

    @property
    def samplenames(self):
        '''pandas.Index of sample names'''
        return self._samplesheet.index.copy()

    @property
    def featurenames(self):
        '''pandas.Index of feature names'''
        return self._counts.index.copy()

    @property
    def metadatanames(self):
        '''pandas.Index of metadata column names'''
        return self._samplesheet.columns.copy()

    @property
    def samplesheet(self):
        '''Matrix of metadata.

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
        '''
        return self._counts

    @counts.setter
    def counts(self, value):
        self._samplesheet = self._samplesheet.loc[value.columns]
        self._counts = value

    def copy(self):
        '''Copy of the Dataset including a new SampleSheet and CountsTable'''
        return self.__class__(
                self._samplesheet.copy(),
                self._counts.copy())

    def query_samples_by_counts(self, expression, inplace=False):
        '''Select samples based on gene expression.

        Args:
            expression (string): An expression compatible with pandas.DataFrame.query.
            inplace (bool): Whether to change the Dataset in place or return a new one.

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

        counts_table = counts.T.query(expression, inplace=False).T
        if drop:
            counts_table.drop(drop, axis=0, inplace=True)

        if inplace:
            self.counts = counts_table
        else:
            samplesheet = self._samplesheet.loc[counts_table.columns].copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table)

    def query_features(self, expression, inplace=False):
        '''Select features based on their expression.

        Args:
            expression (string): An expression compatible with pandas.DataFrame.query.
            inplace (bool): Whether to change the Dataset in place or return a new one.

        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''
        if inplace:
            self._counts.query(expression, inplace=True)
        else:
            counts_table = self._counts.query(expression, inplace=False)
            samplesheet = self._samplesheet.copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table)
