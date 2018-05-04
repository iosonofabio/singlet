# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to CSV/TSV files.
# Modules
import numpy as np
from singlet.config import config


# Parser
def parse_samplesheet(path, fmt):
    import pandas as pd

    if fmt == 'tsv':
        sep = '\t'
    elif fmt == 'csv':
        sep = ','
    else:
        raise ValueError('Format not understood')

    table = pd.read_csv(path, sep=sep, index_col=False)
    return table


def parse_featuresheet(path, fmt):
    import pandas as pd

    if fmt == 'tsv':
        sep = '\t'
    elif fmt == 'csv':
        sep = ','
    else:
        raise ValueError('Format not understood')

    table = pd.read_csv(path, sep=sep, index_col=False)
    return table


def parse_counts_table(path, fmt):
    import pandas as pd

    if fmt in ('tsv', 'tsv.gz'):
        sep = '\t'
    elif fmt in ('csv', 'csv.gz'):
        sep = ','
    else:
        raise ValueError('Format not understood')

    table = pd.read_csv(path, sep=sep, index_col=False)
    return table
