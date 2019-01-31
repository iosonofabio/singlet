# vim: fdm=indent
'''
author:     Fabio Zanini
date:       31/01/19
content:    Test I/O of sparse counts table via npz files.
'''
import os


def test_parse__save_npz():
    print('Parsing example NPZ count table')
    from singlet.io import parse_counts_table_sparse
    table = parse_counts_table_sparse({'countsname': 'example_PBMC_sparse'})
    print('Done!')

    print('Saving NPZ count table')
    from singlet.io.npz import to_counts_table_sparse
    to_counts_table_sparse(table, 'example_data/example_PBMC_sparse_backup.npz')
    os.remove('example_data/example_PBMC_sparse_backup.npz')
    print('Done!')
