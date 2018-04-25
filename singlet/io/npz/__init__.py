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
    index = pd.Index(tmp['index'], name=tmp['indexname'][0])
    columns = pd.Index(tmp['columns'], name='samplename')
    table = pd.SparseDataFrame(
            matrix,
            index=index,
            columns=columns,
            dtype=float,
            default_fill_value=0)
    return table


def to_counts_table_sparse(counts, path):
    coo = counts.to_coo()

    index = counts.index.values
    indexname = np.array([counts.index.name])
    columns = counts.columns.values
    shape = counts.shape
    data = coo.data
    row = coo.row
    col = coo.col

    np.savez(
        path,
        row=row,
        col=col,
        data=data,
        shape=shape,
        index=index,
        columns=columns,
        indexname=indexname)
