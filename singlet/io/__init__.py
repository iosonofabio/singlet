# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Parse sample sheets.
# Modules
import numpy as np
import pandas as pd
from singlet.config import config


# Parser
def parse_samplesheet(dictionary):
    from .csv import parse_samplesheet as parse_csv
    from .googleapi import parse_samplesheet as parse_googleapi

    if 'sheetname' in dictionary:
        sheet = config['io']['samplesheets'][dictionary['sheetname']]
    elif 'datasetname' in dictionary:
        sheet = config['io']['datasets'][dictionary['datasetname']]['samplesheet']
    else:
        raise ValueError('Please specify a samplesheet or a dataset')

    if 'path' in sheet:
        table = parse_csv(sheet['path'], sheet['format'])
    elif 'url' in sheet:
        table = parse_googleapi(sheet)

    if ('cells' in sheet) and (sheet['cells'] != 'rows'):
        table = table.T

    if 'index' in sheet:
        index_col = sheet['index']
    else:
        index_col = 'name'

    table.set_index(index_col, inplace=True, drop=True)

    return table


def parse_featuresheet(dictionary):
    from .csv import parse_featuresheet as parse_csv

    if 'sheetname' in dictionary:
        sheet = config['io']['featuresheets'][dictionary['sheetname']]
    elif 'datasetname' in dictionary:
        sheet = config['io']['datasets'][dictionary['datasetname']]['featuresheet']
    else:
        raise ValueError('Please specify a featuresheet or a dataset')

    table = parse_csv(sheet['path'], sheet['format'])

    if ('features' in sheet) and (sheet['features'] != 'rows'):
        table = table.T

    if 'index' in sheet:
        index_col = sheet['index']
    else:
        index_col = 'name'

    table.set_index(index_col, inplace=True, drop=True)

    return table


def parse_counts_table(dictionary):
    from .csv import parse_counts_table as parse_csv
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
        if fmt == 'tsv':
            parse = parse_csv
        elif fmt == 'csv':
            parse = parse_csv
        elif fmt == 'pickle':
            parse = parse_pickle
        elif fmt == 'csv.gz':
            parse = parse_csv
        elif fmt == 'tsv.gz':
            parse = parse_csv
        else:
            raise ValueError('Format not understood')

        table = parse(path, fmt)
        if ('cells' in sheet) and (sheet['cells'] != 'columns'):
            table = table.T

        if 'index' in sheet:
            table.set_index(sheet['index'], inplace=True, drop=True)
        elif not table.index.name:
            table.set_index(table.columns[0], inplace=True, drop=True)

        # Feature names are strings, but counts are floats
        table.index = table.index.astype(str)
        table = table.astype(float)

        tables.append(table)

    if len(tables) == 1:
        table = tables[0]
    else:
        table = pd.concat(tables, axis=1)
    return table


def parse_counts_table_sparse(tablename):
    from .npz import parse_counts_table_sparse as parse_npz
    # FIXME: check this function!!
    sheet = config['io']['count_tables'][tablename]
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
