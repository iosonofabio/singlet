# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to pickle files.
# Modules


# Parser
def parse_counts_table(path, fmt):
    import pandas as pd
    return pd.read_pickle(path)
