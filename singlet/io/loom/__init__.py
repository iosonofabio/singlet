# vim: fdm=indent
# author:     Fabio Zanini
# date:       02/08/17
# content:    Support module for filenames related to LOOM files.
# Modules
import numpy as np
import pandas as pd
from singlet.config import config


# Parser
def parse_dataset(
        path,
        axis_samples,
        index_samples,
        index_features,
        bit_precision=64):
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

        if index_samples is not None:
            samplesheet.set_index(index_samples, drop=False, inplace=True)
        if index_features is not None:
            featuresheet.set_index(index_features, drop=False, inplace=True)

        # Parse counts
        count_mat = ds[:, :]
        dtypes = {
            16: np.float16,
            32: np.float32,
            64: np.float64,
            128: np.float128,
            }
        dtype_tgt = dtypes[bit_precision]
        if np.dtype(ds[0, 0]) != dtype_tgt:
            count_mat = count_mat.astype(dtype_tgt)

        if axis_samples == 'columns':
            counts_table = pd.DataFrame(
                data=count_mat,
                index=featuresheet.index,
                columns=samplesheet.index,
                )
        else:
            counts_table = pd.DataFrame(
                data=count_mat.T,
                index=samplesheet.index,
                columns=featuresheet.index,
                )

    return {
        'counts': counts_table,
        'samplesheet': samplesheet,
        'featuresheet': featuresheet,
        }
