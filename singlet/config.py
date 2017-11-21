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
try:
    with open(config_filename) as stream:
        config = yaml.load(stream)
    if config is None:
        config = {}
except IOError:
    config = {}

# Warnings that should be seen only once
config['_once_warnings'] = []


# Process config
if 'io' in config:
    if 'count_tables' in config['io']:
        for tablename, sheet in config['io']['count_tables'].items():
            if ('format' not in sheet) and ('path' in sheet):
                path = sheet['path']
                if isinstance(path, str):
                    fmt = path.split('.')[-1].lower()
                else:
                    fmt = [p.split('.')[-1].lower() for p in path]
                config['io']['count_tables'][tablename]['format'] = fmt

    if 'samplesheets' in config['io']:
        for sheetname, sheet in config['io']['samplesheets'].items():
            if ('format' not in sheet) and ('path' in sheet):
                path = sheet['path']
                config['io']['samplesheets'][sheetname]['format'] = path.split('.')[-1].lower()

            elif ('google_id' not in sheet) and ('url' in sheet):
                url = sheet['url']
                config['io']['samplesheets'][sheetname]['google_id'] = url.split('/')[-1]

    if 'featuresheets' in config['io']:
        for sheetname, sheet in config['io']['featuresheets'].items():
            if ('format' not in sheet) and ('path' in sheet):
                path = sheet['path']
                config['io']['featuresheets'][sheetname]['format'] = path.split('.')[-1].lower()
