#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       07/08/17
content:    Test Dataset class.
'''
import matplotlib.pyplot as plt
import numpy as np


# Script
if __name__ == '__main__':

    # NOTE: an env variable for the config file needs to be set when
    # calling this script
    from singlet.dataset import Dataset
    ds = Dataset(samplesheet='example_sheet_tsv', counts_table='example_table_tsv')

    print('Correlation features to phenotypes')
    ax = ds.plot.plot_coverage(color='blue', lw=3)
    ax = ds.plot.plot_coverage(
            features='other', color='red', linewidth=1,
            ax=ax)
    plt.show()
    print('Done!')
