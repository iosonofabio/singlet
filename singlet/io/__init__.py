# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Parse sample sheets.
# Modules
from singlet.config import config


# Parser
def parse_samplesheet(sheetname):
    from .csv import parse_samplesheet as parse_csv
    from .googleapi import parse_samplesheet as parse_googleapi

    sheet = config['io']['samplesheets'][sheetname]
    if 'path' in sheet:
        return parse_csv(sheetname)
    elif 'url' in sheet:
        return parse_googleapi(sheetname)
