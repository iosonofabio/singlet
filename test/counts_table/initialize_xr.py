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
    print('Instantiating CountsTableXR')
    from singlet.counts_table import CountsTableXR
    ct = CountsTableXR.from_tablename('example_table_tsv')
    print('Done!')

    print('log CountsTableXR')
    ctlog = ct.log(base=10)
    print('Done!')

    print('unlog CountsTableXR')
    ctunlog = ctlog.unlog(base=10)
    print('Done!')
