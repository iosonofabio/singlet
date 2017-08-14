# vim: fdm=indent
'''
author:     Fabio Zanini
date:       14/08/17
content:    Dataset that combines feature counts with metadata.
'''
# Modules


# Classes / functions
class Dataset():
    def __init__(self, samplesheet, counts_table):
        from .samplesheet import SampleSheet
        from .counts_table import CountsTable

        self.samplesheet = SampleSheet.from_sheetname(samplesheet)
        self.counts = CountsTable.from_tablename(counts_table)
