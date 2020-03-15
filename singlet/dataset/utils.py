# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/03/20
# content:    Dataset that combines feature counts with metadata.
# Modules
import numpy as np
import pandas as pd


def concatenate(datasets):
    '''Concatenate a list of datasets

    Args:
        datasets (list): list of singlet.Dataset objects to be concatenated.
            The function runs over it more than once so a consuming lazy iterator
            is not a valid input.

    Returns:
        concatenated singlet.Dataset
    '''
    if len(datasets) == 0:
        raise ValueError('Cannot concatenate empty list')

    ns = []
    ms = []
    for ds in datasets:
        ns.append(ds.n_samples)
        ms.append(ds.n_features)

    for m in ms:
        if m != ms[0]:
            raise ValueError(
                    'Cannot concatenate datasets with different features')

    m = ms[0]
    ntot = sum(ns)
    mat = np.zeros((m, ntot), dtype=datasets[0].counts.values.dtype)
    cells = np.zeros(ntot, 'O')
    i = 0
    for ii, (ds, n) in enumerate(zip(datasets, ns)):
        mat[:, i: i+n] = ds.counts.values
        cells[i: i+n] = [x+'-'+str(ii+1) for x in ds.samplenames]
        i += n
    counts = datasets[0].counts.__class__(
            mat,
            index=datasets[0].featurenames,
            columns=cells,
            )
    samplesheet = pd.concat([ds.samplesheet for ds in datasets], axis=0)
    samplesheet.index = cells
    featuresheet = datasets[0].featuresheet

    return datasets[0].__class__(
        counts_table=counts,
        samplesheet=samplesheet,
        featuresheet=featuresheet,
        )
