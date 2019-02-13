# vim: fdm=indent
# author:     Fabio Zanini
# date:       13/02/19
# content:    Plugin template and infrastructure.


# Classes / functions
class Plugin():
    '''Plugin for singlet Dataset'''

    def __init__(self, dataset):
        '''Set the dataset to self.dataset

        Args:
            dataset (Dataset): the dataset to analyze.
        '''
        self.dataset = dataset
