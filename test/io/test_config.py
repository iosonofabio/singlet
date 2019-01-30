#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test YAML parser for config files.
'''
def test_config():
    print('Parsing config file YAML')
    from singlet.config import config
    print('Done!')


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    test_config()
