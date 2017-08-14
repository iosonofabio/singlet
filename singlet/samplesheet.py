# vim: fdm=indent
'''
author:     Fabio Zanini
date:       14/08/17
content:    Samplesheet with metadata.
'''
# Modules
import pandas as pd


# Classes / functions
class SampleSheet(pd.DataFrame):
    _metadata = ['sheet']

    @property
    def _constructor(self):
        return SampleSheet

    @classmethod
    def from_sheetname(cls, sheetname):
        from .config import config
        from .io import parse_samplesheet

        self = cls(parse_samplesheet(sheetname))
        self.name = sheetname
        self.sheet = config['io']['samplesheets'][sheetname]

        return self
