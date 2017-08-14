# vim: fdm=indent
'''
author:     Fabio Zanini
date:       09/08/17
content:    Table of gene counts
'''
# Modules
import pandas as pd


# Classes / functions
class CountsTable(pd.DataFrame):
    _metadata = ['sheet']

    @property
    def _constructor(self):
        return CountsTable

    @classmethod
    def from_tablename(cls, tablename):
        from .config import config
        from .io.csv import parse_counts_table

        self = cls(parse_counts_table(tablename))
        self.name = tablename
        self.sheet = config['io']['count_tables'][self.tablename]

        return self

    def exclude_features(self, spikeins=True, other=True):
        import re
        index = self.index

        if spikeins:
            reg = re.compile('^'+self.sheet['spikein-regex'])
            index = [ix for ix in index if not reg.match(ix)]

        if other:
            reg = re.compile('^'+self.sheet['other-regex'])
            index = [ix for ix in index if not reg.match(ix)]

        return self.loc[index]
