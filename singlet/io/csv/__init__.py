# vim: fdm=indent
'''
author:     Fabio Zanini
date:       02/08/17
content:    Support module for filenames related to CSV/TSV files.
'''
# Modules
import os
import yaml

from singlet.config import config


# Process config
for sheetname, sheet in config['io']['samplesheets'].items():
    if ('format' not in sheet) and ('path' in sheet):
        path = sheet['path']
        config['io']['samplesheets'][sheetname]['format'] = path.split('.')[-1].lower()


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

    table = pd.read_csv(sheet['path'], sep=sep)

    if ('cells' in sheet) and (sheet['cells'] != 'rows'):
        table = table.T

    return table
