# vim: fdm=indent
'''
author:     Fabio Zanini
date:       20/01/17
content:    Access the Nextera indices sheet.
'''
# Modules
import os
import sys
import numpy as np
import pandas as pd
import argparse

from singlecell.googleapi.googleapi import GoogleAPI


# Classes / functions
class NexteraIndices(GoogleAPI):
    def __init__(self):
        sId = '1arKUciQ016fpHRyjMZi4PqEizdhd7DVCmBIdXl_Mgjs'
        super().__init__(sId)

    def get_index_matrix(self, range_names):
        named_ranges = self.get_named_ranges(
                sheetname='Plates',
                fmt='dict',
                convert=True)
        ranges = [named_ranges[k] for k in range_names]
        data = self.get_data('Plates', ranges=[r.split('!')[-1] for r in ranges])
        return np.array(data)

    def get_index_info_samplesheet(self, index_groups):
        named_ranges = self.get_named_ranges(
                sheetname='List',
                fmt='dict',
                convert=True)
        ranges = ['A1:G1', named_ranges['listAll'].split('!')[-1]]
        header, data = self.get_data('List', ranges=ranges)
        data = pd.DataFrame(data, columns=header[0]).set_index('Index Name')
        return [data.loc[indices] for indices in index_groups]


# Script
if __name__ == '__main__':

    nx = NexteraIndices()

    data = nx.get_index_matrix(['plateEx1', 'plateEx2', 'plateEx3', 'plateEx4'])
    info = nx.get_index_info_samplesheet(['S521'])
