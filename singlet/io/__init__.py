# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Parse sample sheets.
# Modules
import pandas as pd
from singlet.config import config


# Parser
def parse_samplesheet(sheetname):
    from .csv import parse_samplesheet as parse_csv
    from .googleapi import parse_samplesheet as parse_googleapi

    sheet = config['io']['samplesheets'][sheetname]
    if 'path' in sheet:
        table = parse_csv(sheet['path'], sheet['format'])
    elif 'url' in sheet:
        table = parse_googleapi(sheetname)

    if ('cells' in sheet) and (sheet['cells'] != 'rows'):
        table = table.T

    if 'index_column' in sheet:
        index_col = sheet['index_column']
    else:
        index_col = 'name'

    table.set_index(index_col, inplace=True, drop=True)

    return table


def parse_featuresheet(sheetname):
    from .csv import parse_featuresheet as parse_csv

    sheet = config['io']['featuresheets'][sheetname]
    table = parse_csv(sheet['path'], sheet['format'])

    if ('features' in sheet) and (sheet['features'] != 'rows'):
        table = table.T

    if 'index_column' in sheet:
        index_col = sheet['index_column']
    else:
        index_col = 'name'

    table.set_index(index_col, inplace=True, drop=True)

    return table


def parse_counts_table(tablename):
    from .csv import parse_counts_table as parse_csv
    from .pickle import parse_counts_table as parse_pickle

    sheet = config['io']['count_tables'][tablename]
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
        else:
            raise ValueError('Format not understood')

        table = parse(path, fmt)
        if ('cells' in sheet) and (sheet['cells'] != 'columns'):
            table = table.T

        tables.append(table)

    if len(tables) == 1:
        table = tables[0]
    else:
        table = pd.concat(tables, axis=1)
    return table
