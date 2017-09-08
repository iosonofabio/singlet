# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to CSV/TSV files.
# Modules
from singlet.config import config


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

    table = pd.read_csv(sheet['path'], sep=sep, index_col=False)

    return table


def parse_featuresheet(sheetname):
    import pandas as pd

    if 'featuresheets' not in config['io']:
        raise ValueError('Config file has no featuresheets section')

    sheet = config['io']['featuresheets'][sheetname]
    fmt = sheet['format']

    if fmt == 'tsv':
        sep = '\t'
    elif fmt == 'csv':
        sep = ','
    else:
        raise ValueError('Format not understood')

    table = pd.read_csv(sheet['path'], sep=sep, index_col='name')

    if ('features' in sheet) and (sheet['features'] != 'rows'):
        table = table.T

    return table


def parse_counts_table(path, fmt):
    import pandas as pd

    if fmt == 'tsv':
        sep = '\t'
    elif fmt == 'csv':
        sep = ','
    else:
        raise ValueError('Format not understood')

    table = pd.read_csv(path, sep=sep, index_col=0)

    return table
