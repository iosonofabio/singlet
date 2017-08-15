# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to CSV/TSV files.
# Modules
import os
import yaml

from singlet.config import config


# Process config
for sheetname, sheet in config['io']['samplesheets'].items():
    if ('format' not in sheet) and ('path' in sheet):
        path = sheet['path']
        config['io']['samplesheets'][sheetname]['format'] = path.split('.')[-1].lower()
for tablename, sheet in config['io']['count_tables'].items():
    if ('format' not in sheet) and ('path' in sheet):
        path = sheet['path']
        if isinstance(path, str):
            fmt = path.split('.')[-1].lower()
        else:
            fmt = [p.split('.')[-1].lower() for p in path]
        config['io']['count_tables'][tablename]['format'] = fmt


# Parser
def parse_samplesheet(sheetname):
    import pandas as pd

    sheet = config['io']['samplesheets'][sheetname]
    fmt = sheet['format']

    if fmt == 'tsv':
        sep = '\t'
    elif fmt == 'csv':
        sep = ','
    else:
        raise ValueError('Format not understood')

    table = pd.read_csv(sheet['path'], sep=sep, index_col='name')

    if ('cells' in sheet) and (sheet['cells'] != 'rows'):
        table = table.T

    return table


def parse_counts_table(tablename):
    import pandas as pd

    sheet = config['io']['count_tables'][tablename]
    paths = sheet['path']
    fmts = sheet['format']
    if isinstance(paths, str):
        paths = [paths]
        fmts = [fmts]

    tables = []
    for path, fmt in zip(paths, fmts):
        if fmt == 'tsv':
            sep = '\t'
        elif fmt == 'csv':
            sep = ','
        else:
            raise ValueError('Format not understood')

        table = pd.read_csv(path, sep=sep, index_col=0)

        if ('cells' in sheet) and (sheet['cells'] != 'columns'):
            table = table.T

        tables.append(table)

    if len(tables) == 1:
        table = tables[0]
    else:
        table = pd.concat(tables, axis=1)
    return table
