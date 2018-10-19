# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to LOOM files.
# Modules
import numpy as np
import pandas as pd
from singlet.config import config


# Parser
def parse_dataset(path, axis_samples, index_samples, index_features):
    import loompy

    with loompy.connect(path) as ds:
        if axis_samples == 'columns':
            samplesheet = pd.DataFrame(data=ds.ca)
            featuresheet = pd.DataFrame(data=ds.ra)
        else:
            samplesheet = pd.DataFrame(data=ds.ra)
            featuresheet = pd.DataFrame(data=ds.ca)

        samplesheet.set_index(index_samples, drop=False)
        featuresheet.set_index(index_features, drop=False)

        if axis_samples == 'columns':
            counts_table = pd.DataFrame(
                data=ds,
                index=featuresheet.index,
                columns=samplesheet.index,
                )
        else:
            counts_table = pd.DataFrame(
                data=ds.T,
                index=samplesheet.index,
                columns=featuresheet.index,
                )

    return {
        'counts_table': counts_table,
        'samplesheet': samplesheet,
        'featuresheet': featuresheet,
        }
