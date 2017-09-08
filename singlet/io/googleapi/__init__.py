# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to the Google Sheet APIs.
# Modules


# Parser
def parse_samplesheet(sheetname):
    from .samplesheet import SampleSheet
    return SampleSheet(sheetname).get_table(fmt='pandas')
