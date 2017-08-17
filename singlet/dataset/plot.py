# vim: fdm=indent
# author:     Fabio Zanini
# date:       16/08/17
# content:    Dataset functions to plot gene expression and phenotypes
# Modules


# Classes / functions
class Plot():
    '''Plot gene expression and phenotype in single cells'''
    def __init__(self, dataset):
        '''Plot gene expression and phenotype in single cells

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset
