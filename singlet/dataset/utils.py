# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/03/20
# content:    Dataset that combines feature counts with metadata.
# Modules
from collections import Counter
import numpy as np
import pandas as pd


def concatenate(datasets, missing='intersect'):
    '''Concatenate a list of datasets

    Args:
        datasets (list): list of singlet.Dataset objects to be concatenated.
            The function runs over it more than once so a consuming lazy iterator
            is not a valid input.
        missing (str): What to do with genes that are missing from any of the
            datasets. 'pad' means pad the missing genes with zeros, 'intersect'
            means take only the intersection

    Returns:
        concatenated singlet.Dataset
    '''
    if len(datasets) == 0:
        raise ValueError('Cannot concatenate empty list')

    ns = []
    feas = []
    for ds in datasets:
        ns.append(ds.n_samples)
        feas.append(ds.featurenames)

    if missing == 'intersect':
        features = feas[0]
        if len(datasets) > 1:
            for fea in feas[1:]:
                features = np.intersect1d(features, fea)
    elif missing == 'pad':
        features = feas[0]
        if len(datasets) > 1:
            for fea in feas[1:]:
                features = np.union1d(features, fea)
        features = np.sort(features)
    else:
        raise ValueError('missing must be "pad" or "intersect"')

    samplesheet = pd.concat([ds.samplesheet for ds in datasets], axis=0)
    if Counter(samplesheet.index.values).most_common(1)[0][1] > 1:
        raise ValueError(
                'Samples cannot share names, make sure they are unique',
                )

    fea_vect = pd.Series(np.arange(len(features)), index=features)
    m = len(features)
    ntot = sum(ns)
    mat = np.zeros((m, ntot), dtype=datasets[0].counts.values.dtype)
    i = 0
    for ii, (ds, n) in enumerate(zip(datasets, ns)):
        # Shortcut if the features are alright
        if (len(feas[ii]) == m) and (feas[ii] == features).all():
            mat[:, i: i+n] = ds.counts.values
        elif missing == 'intersect':
            mat[:, i: i+n] = ds.counts.loc[features].values
        else:
            js = fea_vect.loc[feas[ii]].values
            mat[js, i: i+n] = ds.counts.values

        i += n

    counts = datasets[0].counts.__class__(
            mat,
            index=features,
            columns=samplesheet.index,
            )
    if missing == 'intersect':
        featuresheet = datasets[0].featuresheet.loc[features].copy()
    else:
        featuresheet = None

    return datasets[0].__class__(
        counts_table=counts,
        samplesheet=samplesheet,
        featuresheet=featuresheet,
        )
