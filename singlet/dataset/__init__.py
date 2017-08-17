# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Dataset that combines feature counts with metadata.
# Modules


# Classes / functions
class Dataset():
    '''Collection of cells, with feature counts and metadata'''

    def __init__(self, samplesheet, counts_table):
        '''Collection of cells, with feature counts and metadata

        Args:
            samplesheet (string): Name of the samplesheet (from a config file)
            counts_table (string): Name of the counts table (from a config file)
        '''
        from ..samplesheet import SampleSheet
        from ..counts_table import CountsTable
        from .correlations import Correlation

        if not isinstance(samplesheet, SampleSheet):
            samplesheet = SampleSheet.from_sheetname(samplesheet)
        self._samplesheet = samplesheet

        if not isinstance(counts_table, CountsTable):
            counts_table = CountsTable.from_tablename(counts_table)
        self._counts = counts_table

        # Allow sorting of the counts columns
        assert(set(self._samplesheet.index) == set(self._counts.columns))
        self._counts = self._counts.loc[:, self._samplesheet.index]

        # Plugins
        self.correlation = Correlation(self)


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
