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
        obsm_keys=None,
        ):
    import anndata

    adata = anndata.read_h5ad(path)

    samplesheet = adata.obs.copy()

    # Add obsm (e.g. PCA, embeddings)
    for key, array in adata.obsm.items():
        if key.startswith('X_'):
            newkey = key[2:]
            if (obsm_keys is not None) and (newkey not in obsm_keys):
                continue
            for j, col in enumerate(array.T, 1):
                samplesheet[newkey+'_'+str(j)] = col

    featuresheet = adata.var.copy()

    count_mat = adata.X.toarray().T

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
