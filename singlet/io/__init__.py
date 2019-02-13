# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Parse sample sheets.
# Modules
import numpy as np
import pandas as pd
from singlet.config import config


integrated_dataset_formats = ['loom']


# Parser
def parse_samplesheet(dictionary):
    from .csv import parse_samplesheet as parse_csv, csv_formats
    from .googleapi import parse_samplesheet as parse_googleapi

    if 'sheetname' in dictionary:
        sheet = config['io']['samplesheets'][dictionary['sheetname']]
    elif 'datasetname' in dictionary:
        sheet = config['io']['datasets'][dictionary['datasetname']]['samplesheet']
    else:
        raise ValueError('Please specify a samplesheet or a dataset')

    if ('format' in sheet) and (sheet['format'] in csv_formats):
        table = parse_csv(sheet['path'], sheet['format'])
    elif 'url' in sheet:
        table = parse_googleapi(sheet)
    else:
        raise ValueError('samplesheet format not recognized')

    if ('cells' in sheet) and (sheet['cells'] != 'rows'):
        table = table.T

    if 'index' in sheet:
        index_col = sheet['index']
    else:
        index_col = 'name'

    table.set_index(index_col, inplace=True, drop=True)

    return table


def parse_featuresheet(dictionary):
    from .csv import parse_featuresheet as parse_csv, csv_formats

    if 'sheetname' in dictionary:
        sheet = config['io']['featuresheets'][dictionary['sheetname']]
    elif 'datasetname' in dictionary:
        sheet = config['io']['datasets'][dictionary['datasetname']]['featuresheet']
    else:
        raise ValueError('Please specify a featuresheet or a dataset')

    if sheet['format'] in csv_formats:
        table = parse_csv(sheet['path'], sheet['format'])
    else:
        raise ValueError('samplesheet format not recognized')

    if ('features' in sheet) and (sheet['features'] != 'rows'):
        table = table.T

    if 'index' in sheet:
        index_col = sheet['index']
    else:
        index_col = 'name'

    table.set_index(index_col, inplace=True, drop=True)

    return table


def parse_counts_table(dictionary):
    from .csv import parse_counts_table as parse_csv, csv_formats
    from .pickle import parse_counts_table as parse_pickle

    if 'countsname' in dictionary:
        sheet = config['io']['count_tables'][dictionary['countsname']]
    elif 'datasetname' in dictionary:
        sheet = config['io']['datasets'][dictionary['datasetname']]['counts_table']
    else:
        raise ValueError('Please specify a counts_table or a dataset')

    paths = sheet['path']
    fmts = sheet['format']
    if isinstance(paths, str):
        paths = [paths]
        fmts = [fmts]

    tables = []
    for path, fmt in zip(paths, fmts):
        if fmt in csv_formats:
            parse = parse_csv
        elif fmt == 'pickle':
            parse = parse_pickle
        else:
            raise ValueError('Format not understood')

        table = parse(path, fmt)
        if ('cells' in sheet) and (sheet['cells'] != 'columns'):
            table = table.T

        if 'index' in sheet:
            table.set_index(sheet['index'], inplace=True, drop=True)
        elif not table.index.name:
            table.set_index(table.columns[0], inplace=True, drop=True)

        # Feature names are strings
        table.index = table.index.astype(str)

        # Counts are floats
        if sheet['bit_precision'] == 64:
            table = table.astype(np.float64)
        elif sheet['bit_precision'] == 128:
            table = table.astype(np.float128)
        elif sheet['bit_precision'] == 32:
            table = table.astype(np.float32)
        elif sheet['bit_precision'] == 16:
            table = table.astype(np.float16)
        else:
            raise ValueError('Bit precision must be one of 16, 32, 64, or 128')

        tables.append(table)

    if len(tables) == 1:
        table = tables[0]
    else:
        table = pd.concat(tables, axis=1)
    return table


def parse_counts_table_sparse(dictionary):
    from .npz import parse_counts_table_sparse as parse_npz

    if 'countsname' in dictionary:
        sheet = config['io']['count_tables'][dictionary['countsname']]
    elif 'datasetname' in dictionary:
        sheet = config['io']['datasets'][dictionary['datasetname']]['counts_table']
    else:
        raise ValueError('Please specify a counts_table or a dataset')

    paths = sheet['path']
    fmts = sheet['format']
    if isinstance(paths, str):
        paths = [paths]
        fmts = [fmts]

    tables = []
    for path, fmt in zip(paths, fmts):
        if fmt == 'npz':
            parse = parse_npz
        else:
            raise ValueError('Format not understood')

        table = parse(path, fmt)
        if ('cells' in sheet) and (sheet['cells'] != 'columns'):
            table = table.T

        if 'index' in sheet:
            table.set_index(sheet['index'], inplace=True, drop=True)
        elif not table.index.name:
            table.set_index(table.columns[0], inplace=True, drop=True)

        tables.append(table)

    if len(tables) == 1:
        table = tables[0]
    else:
        table = pd.concat(tables, axis=1)
    return table


def parse_dataset(dictionary):
    from .loom import parse_dataset as parse_loom

    if 'datasetname' in dictionary:
        dataset = config['io']['datasets'][dictionary['datasetname']]
    else:
        raise ValueError('A datasetname is required')

    if dataset['format'] == 'loom':
        return parse_loom(
                dataset['path'],
                dataset['axis_samples'],
                dataset['index_samples'],
                dataset['index_features'],
                bit_precision=dataset['bit_precision'],
                )
    else:
        raise ValueError('Integrated dataset parsing supports the following formats: {:}'.format(
            ', '.join(integrated_dataset_formats)))
