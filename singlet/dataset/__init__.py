# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Dataset that combines feature counts with metadata.
# Modules
import numpy as np
import pandas as pd

from ..samplesheet import SampleSheet
from ..counts_table import CountsTable
from ..counts_table import CountsTableSparse
from ..featuresheet import FeatureSheet
from .plugins import Plugin
from .utils import concatenate


# Classes / functions
class Dataset():
    '''Collection of cells, with feature counts and metadata'''

    def __init__(
            self,
            counts_table=None,
            samplesheet=None,
            featuresheet=None,
            dataset=None,
            plugins=None):
        '''Collection of cells, with feature counts and metadata

        Args:
            counts_table (string): Name of the counts table (to load from a
                config file) or instance of CountsTable or CountsTableSparse
            samplesheet (string or None): Name of the samplesheet (to load from
                a config file) or instance of SampleSheet
            featuresheet (string or None): Name of the samplesheet (to load
                from a config file) or instance of FeatureSheet
            dataset (string or dict or None): Name of the Dataset (to load from
                a config file) or dict with the config settings themselves.
            plugins (dict): Dictionary of classes that take the Dataset
                instance as only argument for __init__, to expand the
                possibilities of Dataset operations.

        NOTE: you can set *either* a dataset or a combination of counts_table,
            samplesheet, and featuresheet. Setting both will raise an error.

        NOTE: All samples in the counts_table must also be in the
            samplesheet, but the latter can have additional samples. If
            that is the case, the samplesheet is sliced down to the
            samples present in the counts_table.
        '''

        if ((dataset is not None) and
           ((counts_table is not None) or (samplesheet is not None) or
           (featuresheet is not None))):
            raise ValueError('Set a dataset or a counts_table/samplesheet/featuresheet, but not both')

        if (dataset is None) and (samplesheet is None) and (counts_table is None):
            raise ValueError('A dataset, samplesheet or counts_table must be present')

        if dataset is not None:
            self._from_dataset(dataset)
        else:
            self._from_datastructures(
                counts_table=counts_table,
                samplesheet=samplesheet,
                featuresheet=featuresheet,
                )

        # Inject yourself into counts_table
        self.counts.dataset = self

        # Plugins
        self._set_plugins(plugins=plugins)

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
        return concatenate([self, other])

    def __iadd__(self, other):
        '''Merge two Datasets.

        For samples with the same names, counts will be added and metadata of
            one of the Datasets used. For new samples, the new counts and
            metadata will be used.

        NOTE: metadata and gene names must be aligned for this operation to
            succeed. If one of the two Datasets has more metadata or
            features than the other, they cannot be added.
        '''
        newself = concatenate([self, other])
        self._counts = newself._counts
        self._samplesheet = newself._samplesheet
        return self

    def _set_plugins(self, plugins=None):
        '''Set plugins according to user's request'''
        from .correlations import Correlation
        from .plot import Plot
        from .dimensionality import DimensionalityReduction
        from .cluster import Cluster
        from .fit import Fit
        from .feature_selection import FeatureSelection
        from .graph import Graph

        self.correlation = Correlation(self)
        self.plot = Plot(self)
        self.dimensionality = DimensionalityReduction(self)
        self.cluster = Cluster(self)
        self.fit = Fit(self)
        self.feature_selection = FeatureSelection(self)
        self.graph = Graph(self)
        if (plugins is not None) and len(plugins):
            self._plugins = dict(plugins)
            for key, val in plugins.items():
                setattr(self, key, val(self))
        else:
            self._plugins = {}

    def _from_datastructures(
            self,
            counts_table=None,
            samplesheet=None,
            featuresheet=None):
        '''Set main data structures'''
        from ..config import config

        if counts_table is None:
            if (isinstance(samplesheet, SampleSheet) or
               isinstance(samplesheet, pd.DataFrame)):
                self._counts = CountsTable(
                        data=[],
                        index=[],
                        columns=samplesheet.index)
        elif isinstance(counts_table, CountsTable):
            self._counts = counts_table
        elif isinstance(counts_table, CountsTableSparse):
            self._counts = counts_table
        elif isinstance(counts_table, pd.DataFrame):
            self._counts = CountsTable(counts_table)
        else:
            config_table = config['io']['count_tables'][counts_table]
            if config_table.get('sparse', False):
                self._counts = CountsTableSparse.from_tablename(counts_table)
            else:
                self._counts = CountsTable.from_tablename(counts_table)

        if samplesheet is None:
            self._samplesheet = SampleSheet(
                data=[],
                index=self._counts.columns)
            self._samplesheet.sheetname = None
        elif isinstance(samplesheet, SampleSheet):
            self._samplesheet = samplesheet
        elif isinstance(samplesheet, pd.DataFrame):
            self._samplesheet = SampleSheet(samplesheet)
        else:
            self._samplesheet = SampleSheet.from_sheetname(samplesheet)

        # This is the catchall for counts
        if not hasattr(self, '_counts'):
            self._counts = CountsTable(
                data=[],
                index=[],
                columns=self._samplesheet.index)

        if featuresheet is None:
            self._featuresheet = FeatureSheet(data=[], index=self._counts.index)
            self._featuresheet.sheetname = None
        elif isinstance(featuresheet, FeatureSheet):
            self._featuresheet = featuresheet
        elif isinstance(featuresheet, pd.DataFrame):
            self._featuresheet = FeatureSheet(featuresheet)
        else:
            self._featuresheet = FeatureSheet.from_sheetname(featuresheet)

        # Uniform axes across data and metadata
        # TODO: this runs into a bug if cell names are boolean (e.g. after
        # averaging), hence we make a patchup catch
        if set(self._counts.columns) != set([False, True]):
            self._samplesheet = self._samplesheet.loc[self._counts.columns]
        #self._featuresheet = self._featuresheet.loc[self._counts.index]

    def _from_dataset(self, dataset):
        '''Load from config file using a dataset name or config

        Args:
            dataset (str or dict): if a string, a dataset with this name is
            searched for in the config file. If a dict, it is interpreted as
            the dataset config itself.
        '''
        from ..config import config, _normalize_dataset
        from ..io import parse_dataset, integrated_dataset_formats

        datasetname = None
        if isinstance(dataset, str):
            datasetname = dataset
            dataset = config['io']['datasets'][datasetname]
            dataset['datasetname'] = datasetname
        else:
            dataset = _normalize_dataset(dataset)

        if ('format' in dataset) and (dataset['format'] in integrated_dataset_formats):
            d = parse_dataset(dataset)
            self._counts = CountsTable(d['counts'])
            self._samplesheet = SampleSheet(d['samplesheet'])
            self._featuresheet = FeatureSheet(d['featuresheet'])
        else:
            if ('samplesheet' not in dataset) and ('counts_table' not in dataset):
                raise ValueError('Your dataset config must include a counts_table or a samplesheet')

            if 'samplesheet' in dataset:
                self._samplesheet = SampleSheet.from_datasetname(datasetname)
            if 'counts_table' in dataset:
                config_table = dataset['counts_table']
                if config_table.get('sparse', False):
                    counts_table = CountsTableSparse.from_datasetname(datasetname)
                else:
                    counts_table = CountsTable.from_datasetname(datasetname)
                self._counts = counts_table

            if not hasattr(self, '_samplesheet'):
                self._samplesheet = SampleSheet(
                    data=[],
                    index=self._counts.columns)
                self._samplesheet.sheetname = None
            elif not hasattr(self, '_counts'):
                self._counts = CountsTable(
                        data=[],
                        index=[],
                        columns=self._samplesheet.index)

            if 'featuresheet' in dataset:
                self._featuresheet = FeatureSheet.from_datasetname(datasetname)
            else:
                self._featuresheet = FeatureSheet(data=[], index=self._counts.index)

    def to_dataset_file(self, filename, fmt=None, **kwargs):
        '''Store dataset into an integrated dataset file

        Args:
            filename (str): path of the file to write to.
            fmt (str or None): file format. If None, infer from the file
            extension.
            **kwargs (keyword arguments): depend on the format.

        '''
        if fmt is None:
            fmt = filename.split('.')[-1]

        if fmt == 'loom':
            import loompy

            matrix = self.counts.values
            row_attrs = {col: self.featuresheet[col].values for col in self.featuresheet}
            col_attrs = {col: self.samplesheet[col].values for col in self.samplesheet}

            # Add attributes for the indices no matter what
            if self.featuresheet.index.name is not None:
                row_attrs[self.featuresheet.index.name] = self.featuresheet.index.values
            else:
                row_attrs['_index'] = self.featuresheet.index.values
            if self.samplesheet.index.name is not None:
                col_attrs[self.samplesheet.index.name] = self.samplesheet.index.values
            else:
                col_attrs['_index'] = self.samplesheet.index.values

            loompy.create(filename, matrix, row_attrs, col_attrs)

        elif fmt == 'h5ad':
            adata = self.to_AnnData()
            adata.write(filename)

        else:
            raise ValueError('File format not supported')

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
    def shape(self):
        return (self.n_features, self.n_samples)

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

    def query_samples_by_name(
            self,
            samplenames,
            inplace=False,
            ignore_missing=False,
            ):
        '''Select samples by name.

        Args:
            samplenames: names of the samples to keep.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
            ignore_missing (bool): Whether to silently skip missing samples.
        '''
        if ignore_missing:
            snall = self.samplenames
            samplenames = [fn for fn in samplenames if fn in snall]

        if inplace:
            self._samplesheet = self._samplesheet.loc[samplenames]
            self._counts = self._counts.loc[:, samplenames]
        else:
            samplesheet = self._samplesheet.loc[samplenames].copy()
            counts_table = self._counts.loc[:, samplenames].copy()
            featuresheet = self._featuresheet.copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table,
                    featuresheet=featuresheet)

    def query_features_by_name(
            self,
            featurenames,
            inplace=False,
            ignore_missing=False,
            ):
        '''Select features by name.

        Args:
            featurenames: names of the features to keep.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
            ignore_missing (bool): Whether to silently skip missing features.
        '''
        if ignore_missing:
            fnall = self.featurenames
            featurenames = [fn for fn in featurenames if fn in fnall]

        if inplace:
            self._featuresheet = self._featuresheet.loc[featurenames]
            self._counts = self._counts.loc[featurenames]
        else:
            featuresheet = self._featuresheet.loc[featurenames].copy()
            counts_table = self._counts.loc[featurenames].copy()
            samplesheet = self._samplesheet.copy()
            return self.__class__(
                    samplesheet=samplesheet,
                    counts_table=counts_table,
                    featuresheet=featuresheet)

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

    def exclude_samples_by_name(
            self,
            samplenames,
            inplace=False):
        '''Exclude samples

        Args:
            samplenames (list): Names of samples to exclude
            inplace (bool): Whether to change the Dataset in place or return a
                new one.

        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''
        if inplace:
            ind = ~self._samplesheet.index.isin(samplenames)
            self._counts = self._counts.loc[:, ind]
            self._samplesheet = self._samplesheet.loc[ind]
        else:
            ds = self.cop()
            ds.exclude_samples_by_name(samplenames, inplace=True)
            return ds

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

        DEPRECATED: use `reindex` instead.
        '''
        return self.reindex(axis, column, inplace=inplace)

    def reindex(
            self,
            axis,
            column,
            drop=False,
            inplace=False):
        '''Reindex samples or features from a metadata column

        Args:
            axis (string): Must be 'samples' or 'features'.
            column (string): Must be a column of the samplesheet (for
                axis='samples') or of the featuresheet (for axis='features')
                with unique names of samples or features.
            drop (bool): Whether to drop the column from the metadata table.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.

        Returns:
            If inplace==True, None. Otherwise, a new Dataset.
        '''
        if axis not in ('samples', 'features'):
            raise ValueError('axis must be "samples" or "features"')

        if inplace:
            if axis == 'samples':
                self._samplesheet.index = self._samplesheet.loc[:, column]
                self._counts.columns = self._samplesheet.loc[:, column]
                if drop:
                    del self._samplesheet[column]
            else:
                self._featuresheet.index = self._featuresheet.loc[:, column]
                self._counts.index = self._featuresheet.loc[:, column]
                if drop:
                    del self._featuresheet[column]
        else:
            other = self.copy()
            other.reindex(
                    axis=axis,
                    column=column,
                    drop=drop,
                    inplace=True)
            return other

    def merge_duplicates(
            self,
            axis,
            column,
            keep='first',
            inplace=False,
            ):
        '''Merge duplicate features or samples, adding together their counts

        Args:
            axis (string): Must be 'samples' or 'features'.
            column (string): Must be a column of the samplesheet (for
                axis='samples') or of the featuresheet (for axis='features')
                with potentially duplicate names of samples or features.
            keep (str): Which of the metadata rows to keep. Must be 'first',
                'last', or 'random'.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.

        Returns:
            If inplace==True, None. Otherwise, a new Dataset.
        '''
        from collections import Counter, defaultdict

        if axis not in ('samples', 'features'):
            raise ValueError('axis must be "samples" or "features"')

        cou = Counter()
        tra = defaultdict(list)

        if axis == 'features':
            for idx, val in self.featuresheet[column].items():
                tra[val].append(idx)
                cou[val] += 1
            index_new = []
            counts = np.zeros(
                        (len(cou), self.n_samples),
                        dtype=self.counts.values.dtype,
                        )
            n = 0
            todo = []
            for idx, val in self.featuresheet[column].items():
                if cou[val] == 1:
                    index_new.append(idx)
                    counts[n] += self.counts.loc[idx]
                    n += 1
                else:
                    trai = tra[val]
                    if keep == 'first':
                        jdx = trai[0]
                    elif keep == 'last':
                        jdx = trai[-1]
                    else:
                        jdx = trai[np.random.randint(len(trai))]

                    # New row
                    if idx == jdx:
                        index_new.append(idx)
                        counts[n] += self.counts.loc[idx]
                        n += 1

                    else:
                        todo.append((idx, jdx))
            counts = self.counts.__class__(
                counts,
                index=pd.Index(
                    index_new, name=self.featuresheet.index.name,
                    ),
                columns=self.samplenames,
                )

            for idx, jdx in todo:
                counts.loc[jdx] += self.counts.loc[idx]
            del todo

            if inplace:
                self._counts = counts
                self._featuresheet = self._featuresheet.loc[self._counts.index]
            else:
                return self.__class__(
                    counts=counts,
                    featuresheet=self._featuresheet.loc[counts.index].copy(),
                    samplesheet=self._samplesheet.copy(),
                    )

        else:
            for idx, val in self.samplesheet[column].items():
                tra[val].append(idx)
                cou[val] += 1
            index_new = []
            counts = np.zeros(
                        (self.n_features, len(cou)),
                        dtype=self.counts.values.dtype,
                        )
            n = 0
            todo = []
            for idx, val in self.samplesheet[column].items():
                if cou[val] == 1:
                    index_new.append(idx)
                    counts[:, n] += self.counts.loc[:, idx]
                    n += 1
                else:
                    trai = tra[val]
                    if keep == 'first':
                        jdx = trai[0]
                    elif keep == 'last':
                        jdx = trai[-1]
                    else:
                        jdx = trai[np.random.randint(len(trai))]

                    # New row
                    if idx == jdx:
                        index_new.append(idx)
                        counts[:, n] += self.counts.loc[:, idx]
                        n += 1

                    else:
                        todo.append((idx, jdx))
            counts = self.counts.__class__(
                counts,
                index=self.featurenames,
                columns=pd.Index(
                    index_new, name=self.samplesheet.index.name),
                )

            for idx, jdx in todo:
                counts.loc[:, jdx] += self.counts.loc[:, idx]
            del todo

            if inplace:
                self._counts = counts
                self._samplesheet = self._sampleesheet.loc[self._counts.columns]
            else:
                return self.__class__(
                    counts=counts,
                    samplesheet=self._samplesheet.loc[counts.columns].copy(),
                    featuresheet=self._featuresheet.copy(),
                    )

    def compare(
            self,
            other,
            features='mapped',
            phenotypes=(),
            method='kolmogorov-smirnov',
            additional_attributes=('log2_fold_change', 'avg_self', 'avg_other'),
            ):
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
                dataset, running over the samples) and return a pair
                (statistic, P-value) for the comparison.
            attitional_attributes (list/tuple of str): a list of additional
                attributes about the comparison. At the moment thse can be:
                'log2_fold_change', 'avg_self', 'avg_other'.
        Return:
            A pandas.DataFrame containing the statistic and P-values of the
            comparisons for all features and phenotypes.
        '''
        res = []

        additional_ordered = []
        if method == 'kolmogorov-smirnov-rich':
            additional_ordered.append('KS_xmax')
        if 'log2_fold_change' in additional_attributes:
            additional_ordered.append('log2_fold_change')
        if 'avg_self' in additional_attributes:
            additional_ordered.append('avg_self')
        if 'avg_other' in additional_attributes:
            additional_ordered.append('avg_other')

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
                    tmp = ks_2samp(co1.values, co2.values)
                    res.append([fea, tmp[0], tmp[1]])

            elif method == 'kolmogorov-smirnov-rich':
                from .utils import ks_2samp
                for (fea, co1), (_, co2) in zip(
                        counts.iterrows(),
                        counts_other.iterrows()):
                    tmp = ks_2samp(co1.values, co2.values)
                    res.append([fea, tmp[0], tmp[1], tmp[2]])

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
                        res.append([fea, 0, 1])
                        continue
                    tmp = mannwhitneyu(
                        co1.values, co2.values,
                        alternative='two-sided')
                    res.append([fea, tmp[0], tmp[1]])
            else:
                for (fea, co1) in counts.iterrows():
                    co2 = counts_other.loc[fea]
                    tmp = method(co1.values, co2.values)
                    res.append([fea, tmp[0], tmp[1]])

            if len(additional_attributes):
                i = 0
                for (fea, co1), (_, co2) in zip(
                        counts.iterrows(),
                        counts_other.iterrows()):

                    avg_self, avg_other = None, None
                    if 'log2_fold_change' in additional_attributes:
                        avg_self = co1.values.mean()
                        avg_other = co2.values.mean()
                        log2fc = np.log2(0.1 + avg_self) - np.log2(0.1 + avg_other)
                        res[i].append(log2fc)

                    if 'avg_self' in additional_attributes:
                        if avg_self is None:
                            avg_self = co1.values.mean()
                        res[i].append(avg_self)

                    if 'avg_other' in additional_attributes:
                        if avg_other is None:
                            avg_other = co1.values.mean()
                        res[i].append(avg_other)

                    i += 1

        if phenotypes:
            pheno = self.samplesheet.loc[:, phenotypes].T
            pheno_other = other.samplesheet.loc[:, phenotypes].T

            i_pheno = len(res)

            if method == 'kolmogorov-smirnov':
                from scipy.stats import ks_2samp
                for phe, val1 in pheno.iterrows():
                    val2 = pheno_other.loc[phe]
                    tmp = ks_2samp(val1.values, val2.values)
                    res.append([phe, tmp[0], tmp[1]])

            if method == 'kolmogorov-smirnov-rich':
                from .utils import ks_2samp
                for phe, val1 in pheno.iterrows():
                    val2 = pheno_other.loc[phe]
                    tmp = ks_2samp(val1.values, val2.values)
                    res.append([phe, tmp[0], tmp[1], tmp[2]])

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
                        res.append([phe, 0, 1])
                        continue
                    tmp = mannwhitneyu(
                        val1.values, val2.values,
                        alternative='two-sided')
                    res.append([phe, tmp[0], tmp[1]])
            else:
                for phe, val1 in pheno.iterrows():
                    val2 = pheno_other.loc[phe]
                    tmp = method(val1.values, val2.values)
                    res.append([phe, tmp[0], tmp[1]])

            if len(additional_attributes):
                for (fea, co1), (_, co2) in zip(
                        counts.iterrows(),
                        counts_other.iterrows()):

                    avg_self, avg_other = None, None
                    if 'log2_fold_change' in additional_attributes:
                        avg_self = co1.values.mean()
                        avg_other = co2.values.mean()
                        log2fc = np.log2(0.1 + avg_self) - np.log2(0.1 + avg_other)
                        res[i_pheno].append(log2fc)

                    if 'avg_self' in additional_attributes:
                        if avg_self is None:
                            avg_self = co1.values.mean()
                        res[i_pheno].append(avg_self)

                    if 'avg_other' in additional_attributes:
                        if avg_other is None:
                            avg_other = co1.values.mean()
                        res[i_pheno].append(avg_other)

                    i_pheno += 1

        df = pd.DataFrame(res, columns=['name', 'statistic', 'P-value'] + additional_ordered)
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

    def average(self, axis, by):
        '''Average samples or features based on metadata


        Args:
            axis (string): Must be 'samples' or 'features'.
            by (string or list): Must be one or more column of the samplesheet
                (for axis='samples') or of the featuresheet (for
                axis='features'). Samples or features with a common value in
                these columns are averaged over.
        Returns:
            A Dataset with the averaged counts.

        Note: if you average over samples, you get an empty samplesheet.
        Simlarly, if you average over features, you get an empty featuresheet.

        '''
        if axis not in ('samples', 'features'):
            raise ValueError('axis must be "samples" or "features"')

        by_string = isinstance(by, str)
        if by_string:
            by = [by]
        else:
            by = list(by)

        if axis == 'samples':
            for column in by:
                if column not in self.samplesheet.columns:
                    raise ValueError(
                        '{:} is not a column of the SampleSheet'.format(column))

            if by_string:
                vals = pd.Index(
                        self.samplesheet[by[0]].drop_duplicates(),
                        name=by[0])
            else:
                vals = pd.Index(self.samplesheet[by].drop_duplicates())
            n_conditions = len(vals)
            counts = np.zeros(
                    (self.n_features, n_conditions),
                    dtype=self.counts.values.dtype)
            for i, val in enumerate(vals):
                ind = (self.samplesheet[by] == val).all(axis=1)
                counts[:, i] = self.counts.loc[:, ind].values.mean(axis=1)

            if by_string:
                samplesheet = None
            else:
                vals = pd.MultiIndex.from_tuples(vals, names=by)
                samplesheet = self._samplesheet.__class__(
                        vals.to_frame(index=True),
                        )

            counts = self.counts.__class__(
                    pd.DataFrame(
                        counts,
                        index=self.counts.index,
                        columns=vals),
                    )

            featuresheet = self._featuresheet.copy()

            return Dataset(
                    counts_table=counts,
                    featuresheet=featuresheet,
                    samplesheet=samplesheet,
                    )

        elif axis == 'features':
            if column not in self.featuresheet.columns:
                raise ValueError(
                    '{:} is not a column of the FeatureSheet'.format(column))

            if by_string:
                vals = pd.Index(
                        self.featuresheet[by[0]].drop_duplicates(),
                        name=by[0])
            else:
                vals = pd.Index(self.featuresheet[by].drop_duplicates())

            n_conditions = len(vals)
            counts = np.zeros(
                    (n_conditions, self.n_samples),
                    dtype=self.counts.values.dtype)
            for i, val in enumerate(vals):
                ind = (self.featuresheet[column] == val).all(axis=1)
                counts[i] = self.counts.loc[ind].values.mean(axis=0)

            if by_string:
                featuresheet = None
            else:
                vals = pd.MultiIndex.from_tuples(vals, names=by)
                featuresheet = self._featuresheet.__class__(
                        vals.to_frame(index=True),
                        )

            counts = self.counts.__class__(
                    pd.DataFrame(
                        counts,
                        index=vals,
                        columns=self.counts.columns),
                    )

            samplesheet = self._samplesheet.copy()

            return Dataset(
                    counts_table=counts,
                    samplesheet=samplesheet,
                    featuresheet=featuresheet,
                    )


    @classmethod
    def from_AnnData(cls, adata, convert_obsm=None):
        '''Load from AnnData object

        Args:
            adata (anndata.AnnData): object to load from
            convert_obsm (list or None): if not None, a list of multidimensional
                'obsm' to convert to samplesheet columns
        '''
        shape = adata.shape
        counts = np.zeros(shape, np.float32)
        counts[:, :] = adata.X
        samplesheet = adata.obs
        featuresheet = adata.var
        count_table = CountsTable(
                counts.T,
                index=featuresheet.index,
                columns=samplesheet.index,
                )
        self = cls(
            counts_table=count_table,
            samplesheet=samplesheet,
            featuresheet=featuresheet,
            )

        if convert_obsm is not None:
            for col in convert_obsm:
                nc = adata.obsm[col].shape[1]
                for i in range(nc, 1):
                    self.samplesheet[f'{col}_{i}'] = adata.obsm[col][:, i-1]

        return self

    def to_AnnData(self):
        '''Convert to AnnData object'''
        import anndata

        X = self.counts.values.T
        obs = self.samplesheet
        var = self.featuresheet

        adata = anndata.AnnData(
            X,
            obs=obs,
            var=var,
            )
        return adata

    def subsample(
            self,
            n,
            axis='samples',
            within_metadata=None,
            with_replacement=False,
            inplace=False):
        '''Average samples or features based on metadata

        Args:
            n (int): number of samples or features to take in the subsample.
            axis (string): Must be 'samples' or 'features'.
            within_metadata (None or str): if None, subsample from the whole
                dataset. If a column of sample/featuresheet, subsample n within
                each unique value of that column.
            with_replacement (bool): whether to sample with replacement or not.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
        Returns:
            If inplace is True, None. Else, a Dataset with the subsample.
        '''
        import copy

        if axis not in ('samples', 'features'):
            raise ValueError('axis must be "samples" or "features"')

        if axis == 'samples':
            if within_metadata is None:
                if with_replacement is False:
                    ind = np.arange(self.n_samples)
                    np.random.shuffle(ind)
                    ind = ind[:n]
                    samplenames = self.samplenames[ind]
                else:
                    ind = np.random.randint(self.n_samples, size=n)
                    samplenames = self.samplenames[ind]
            else:
                samplenames = []
                meta = self.samplesheet[[within_metadata]].copy()
                meta['ind'] = np.arange(len(meta))
                metau = np.unique(meta[within_metadata])
                for mu in metau:
                    ind = meta.loc[meta[within_metadata] == mu, 'ind'].values
                    if with_replacement is False:
                        np.random.shuffle(ind)
                        ind = ind[:n]
                        samplenames.extend(self.samplenames[ind].tolist())
                    else:
                        ii = np.random.randint(len(ind), size=n)
                        samplenames.extend(self.samplenames[ind[ii]].tolist())

            if with_replacement is False:
                samplenames_new = list(samplenames)
            else:
                samplenames_new = [sn+'_'+str(i+1) for i, sn in enumerate(samplenames)]

            # Set counts
            counts = self.counts.__class__(
                self.counts.loc[:, samplenames].values,
                index=self.counts.index,
                columns=samplenames_new,
                )
            # Shallow copy of metadata
            for prop in counts._metadata:
                # dataset if special, to avoid infinite loops
                if prop == 'dataset':
                    counts.dataset = None
                elif not hasattr(self.counts, prop):
                    continue
                else:
                    setattr(counts, prop,
                            copy.copy(getattr(self.counts, prop)))
            counts._normalized = self.counts._normalized

            # Set samplesheet
            samplesheet = self.samplesheet.loc[samplenames].copy()
            samplesheet.index = samplenames_new

            # Set featuresheet
            if inplace:
                featuresheet = self.featuresheet
            else:
                featuresheet = self.featuresheet.copy()

        elif axis == 'features':
            if within_metadata is None:
                if with_replacement is False:
                    ind = np.arange(self.n_features)
                    np.random.shuffle(ind)
                    ind = ind[:n]
                    featurenames = self.featurenames[ind]
                    featurenames_new = self.featurenames[ind]
                else:
                    ind = np.random.randint(self.n_features, size=n)
                    featurenames = self.featurenames[ind]
                    featurenames_new = [sn+'_'+str(i+1) for i, sn in enumerate(self.featurenames[ind])]
            else:
                raise NotImplementedError('Subsampling within groups of features not implemented yet!')

            counts = self.counts.loc[featurenames].copy()
            counts.index = featurenames_new
            featuresheet = self.featuresheet.loc[featurenames].copy()
            featuresheet.index = featurenames_new
            if inplace:
                samplesheet = self.samplesheet
            else:
                samplesheet = self.samplesheet.copy()

        if inplace:
            self._counts = counts
            self._samplesheet = samplesheet
            self._featuresheet = featuresheet
        else:
            return Dataset(
                    counts_table=counts,
                    samplesheet=samplesheet,
                    featuresheet=featuresheet,
                    )

    def sort_by_metadata(
            self,
            by,
            axis='samples',
            ascending=True,
            inplace=False,
            ):
        '''Sort samples by one or more metadata columns

        Args:
            by (string or list): column(s) to use for sorting
            axis (string): 'samples' or 'features'
            ascending (bool or list of bools): whether to sort low to high
                values. It can be a list of the same length as 'by' if the
                latter is a list.
            inplace (bool): Whether to change the Dataset in place or return a
                new one.
        Returns:
            If `inplace` is True, None. Else, a Dataset.
        '''

        if axis == 'samples':
            samplenames = (self.samplesheet
                               .sort_values(by, ascending=ascending)
                               .index)
            return self.query_samples_by_name(samplenames, inplace=inplace)
        elif axis == 'features':
            featurenames = (self.featuresheet
                                .sort_values(by, ascending=ascending)
                                .index)
            return self.query_features_by_name(featurenames, inplace=inplace)
        else:
            raise ValueError('axis must be "samples" or "features"')
