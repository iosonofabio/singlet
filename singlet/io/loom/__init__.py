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
        index_samples,
        index_features,
        bit_precision=64):
    import loompy

    with loompy.connect(path) as ds:
        n_features, n_samples = ds.shape

        samplesheet = pd.DataFrame(data=[])
        for key, val in ds.ca.items():
            # 2D tables can be a single ds.ca/ra object
            if val.ndim == 2:
                for i in range(val.shape[1]):
                    samplesheet[key+'-'+str(i+1)] = val[:, i]
            elif val.ndim == 1:
                samplesheet[key] = val
            else:
                print('WARNING: 3+D metadata, skipped: {:}'.format(key))

        featuresheet = pd.DataFrame(data=[])
        for key, val in ds.ra.items():
            # 2D tables can be a single ds.ca/ra object
            if val.ndim == 2:
                for i in range(val.shape[1]):
                    featuresheet[key+'-'+str(i+1)] = val[:, i]
            elif val.ndim == 1:
                featuresheet[key] = val
            else:
                print('WARNING: 3+D metadata, skipped: {:}'.format(key))

        if index_samples is not None:
            if index_samples not in samplesheet.columns:
                raise IndexError('index_samples not found: {:}'.format(
                    ', '.join(samplesheet.columns)))
            samplesheet.set_index(index_samples, drop=False, inplace=True)
        if index_features is not None:
            if index_features not in featuresheet.columns:
                raise IndexError('index_features not found: {:}'.format(
                    ', '.join(featuresheet.columns)))
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

        counts_table = pd.DataFrame(
            data=count_mat,
            index=featuresheet.index,
            columns=samplesheet.index,
            )

    return {
        'counts': counts_table,
        'samplesheet': samplesheet,
        'featuresheet': featuresheet,
        }
