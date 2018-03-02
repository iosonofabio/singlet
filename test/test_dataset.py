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
import argparse
import subprocess as sp


# Controlled sp environment
def run(script, where=None, **kwargs):
    import platform

    if where == 'local' and platform.node() != 'X260':
        return
    if where == 'remote' and platform.node() == 'X260':
        return

    env = os.environ.copy()
    env['SINGLET_CONFIG_FILENAME'] = 'example_data/config_example.yml'

    # Include local tests
    if platform.node() == 'X260':
        singlet_path = os.path.dirname(os.path.dirname(__file__))
        env['PYTHONPATH'] = singlet_path+':'+env['PYTHONPATH']
        print(singlet_path)

    return sp.check_call(
        script,
        env=env,
        shell=True,
        **kwargs)


# Script
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test dataset.')
    parser.add_argument('--tests', nargs='+', default=None,
                        help='Only perform these tests')
    args = parser.parse_args()
    tests = args.tests
    if tests is None:
        tests = [fn.split('.')[0] for fn in os.listdir('test/dataset')]

    for te in tests:
        te_fn = 'test/dataset/{:}.py'.format(te)

        # TODO: one could set up PNG comparisons like Matplotlib itself does
        kwargs = {}
        if te in ('plot', 'graph'):
            kwargs['where'] = 'local'

        run(te_fn, **kwargs)
