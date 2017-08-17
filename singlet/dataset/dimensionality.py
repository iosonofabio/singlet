# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to reduce dimensionality of gene expression
#             and phenotypes.
# Modules


# Classes / functions
class DimensionalityReduction():
    '''Reduce dimensionality of gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Reduce dimensionality of gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset
