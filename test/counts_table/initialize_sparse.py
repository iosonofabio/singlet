#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       15/08/17
content:    Test CountsTableSparse class.
'''
# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    print('Instantiating CountsTableSparse')
    from singlet.counts_table_sparse import CountsTableSparse
    ct = CountsTableSparse.from_tablename('example_PBMC_sparse')
    print('Done!')

    print('log CountsTableSparse')
    ctlog = ct.log(base=10)
    print('Done!')

    print('unlog CountsTableSparse')
    ctunlog = ctlog.unlog(base=10)
    print('Done!')

