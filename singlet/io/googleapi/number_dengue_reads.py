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

from singlecell.googleapi.samplesheet import SampleSheet


# Script
if __name__ == '__main__':

    # Samples playground
    ga = SampleSheet(sandbox=True)
    ndr = ga.get_number_dengue_reads()

    sys.exit()

    new_data = pd.Series(
            [11, 13, 12],
            index=['test_uninfected_1', 'test_uninfected_2', 'test_uninfected_bulk100'])

    result = ga.set_number_dengue_reads(new_data)
