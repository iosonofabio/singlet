# vim: fdm=indent
# author:     Fabio Zanini
# date:       14/08/17
# content:    Samplesheet with metadata.
# Modules
import pandas as pd


# Classes / functions
class SampleSheet(pd.DataFrame):
    _metadata = ['sheetname']

    @property
    def _constructor(self):
        return SampleSheet

    @classmethod
    def from_sheetname(cls, sheetname):
        from .io import parse_samplesheet

        self = cls(parse_samplesheet({'sheetname': sheetname}))
        self.sheetname = sheetname

        return self

    @classmethod
    def from_datasetname(cls, datasetname):
        from .io import parse_samplesheet

        self = cls(parse_samplesheet({'datasetname': datasetname}))
        self.sheetname = datasetname

        return self
