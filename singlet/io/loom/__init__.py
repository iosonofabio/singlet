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
            n_features, n_samples = ds.shape

            samplesheet = pd.DataFrame(data=[])
            for key, val in ds.ca.items():
                samplesheet[key] = val

            featuresheet = pd.DataFrame(data=[])
            for key, val in ds.ra.items():
                featuresheet[key] = val
        else:
            n_samples, n_features = ds.shape

            samplesheet = pd.DataFrame(data=[])
            for key, val in ds.ra.items():
                samplesheet[key] = val

            featuresheet = pd.DataFrame(data=[])
            for key, val in ds.ca.items():
                featuresheet[key] = val

        samplesheet.set_index(index_samples, drop=False, inplace=True)
        featuresheet.set_index(index_features, drop=False, inplace=True)

        if axis_samples == 'columns':
            counts_table = pd.DataFrame(
                data=ds[:, :],
                index=featuresheet.index,
                columns=samplesheet.index,
                )
        else:
            counts_table = pd.DataFrame(
                data=ds[:, :].T,
                index=samplesheet.index,
                columns=featuresheet.index,
                )

    return {
        'counts': counts_table,
        'samplesheet': samplesheet,
        'featuresheet': featuresheet,
        }
