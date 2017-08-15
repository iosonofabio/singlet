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
        from .samplesheet import SampleSheet
        from .counts_table import CountsTable

        if not isinstance(samplesheet, SampleSheet):
            samplesheet = SampleSheet.from_sheetname(samplesheet)
        self._samplesheet = samplesheet

        if not isinstance(counts_table, CountsTable):
            counts_table = CountsTable.from_tablename(counts_table)
        self._counts = counts_table

        # Allow sorting of the counts columns
        assert(set(self._samplesheet.index) == set(self._counts.columns))
        self._counts = self._counts.loc[:, self._samplesheet.index]

    @property
    def samplenames(self):
        return self._samplesheet.index.copy()

    @property
    def featurenames(self):
        return self._counts.index.copy()

    @property
    def metadatanames(self):
        return self._samplesheet.columns.copy()

    @property
    def samplesheet(self):
        '''Matrix of metadata.

        Rows are samples, columns are metadata (e.g. phenotypes).
        '''
        return self._samplesheet.copy()

    @samplesheet.setter
    def samplesheet(self, value):
        self._counts = self._counts.loc[:, value.index]
        self._samplesheet = value

    @property
    def counts(self):
        '''Matrix of gene expression counts.

        Rows are features, columns are samples.
        '''
        return self._counts.copy()

    @counts.setter
    def counts(self, value):
        self._samplesheet = self._samplesheet.loc[value.columns]
        self._counts = value

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
            print(counts.loc['mapped'])
            drop.append('mapped')

        counts_table = counts.T.query(expression, inplace=False).T
        if drop:
            counts_table.drop(drop, axis=0, inplace=True)

        if inplace:
            self.counts = counts_table
        else:
            samplesheet = self.samplesheet.loc[counts_table.columns]
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
