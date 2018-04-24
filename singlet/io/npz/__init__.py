# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to pickle files.
# Modules
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


# Parser
def parse_counts_table_sparse(path, fmt):
    tmp = np.load(path)
    matrix = coo_matrix(
            (tmp['data'], (tmp['row'], tmp['col'])),
            shape=tmp['shape'])
    index = pd.Index(tmp['index'], name=tmp['indexname'])
    columns = pd.Index(tmp['columns'], name='samplename')
    table = pd.SparseDataFrame(
            matrix,
            index=index,
            columns=columns,
            dtype=float)
    return table
