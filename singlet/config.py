# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to the Google Sheet APIs.
# Modules
import os
import yaml


# Globals
config_filename = os.getenv(
        'SINGLET_CONFIG_FILENAME',
        os.getenv('HOME') + '/.singlet/config.yml')
with open(config_filename) as stream:
    config = yaml.load(stream)

# Warnings that should be seen only once
config['_once_warnings'] = []
