#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Tests for the library.
'''
# Modules
import os
import sys
import subprocess as sp


# Controlled sp environment
def run(script, **kwargs):
    sp.run(
        script,
        env='SINGLET_CONFIG_FILENAME=example_data/config_example.yml',
        check=True,
        **kwargs)



# Script
if __name__ == '__main__':

    # Config

    # IO
    run('test/io/csv_parser.py')
