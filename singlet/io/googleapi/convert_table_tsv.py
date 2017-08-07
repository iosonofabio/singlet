# vim: fdm=indent
'''
author:     Fabio Zanini
date:       09/01/17
content:    Test quickstart to access google sheets via the API.
'''
# Modules
import os
import sys
import numpy as np
import pandas as pd

from singlecell.googleapi.googleapi import GoogleAPI
from singlecell.filenames import sample_googlesheet_ids


# Script
if __name__ == '__main__':

    # Samples playground
    spreadsheetId = sample_googlesheet_ids['sandbox']

    ga = GoogleAPI(spreadsheetId)

    shape = ga.get_sheet_shape(sheetname='sequenced')

    print(ga.get_last_column(sheetname='sequenced'))

    table = ga.get_table(sheetname='sequenced')
    ga.update_tsv_table(sheetname='sequenced', sandbox=True)
