# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to pickle files.
# Modules
import pandas as pd


# Parser
def parse_counts_table(path, fmt):
    table = pd.read_pickle(path)
    return table
