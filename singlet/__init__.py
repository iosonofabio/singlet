# vim: fdm=indent
# author:     Fabio Zanini
# date:       15/08/17
# content:    Main singlet module.
# Module exporting
from .samplesheet import SampleSheet
from .featuresheet import FeatureSheet
from .counts_table import CountsTable, CountsTableXR
from .dataset import Dataset, concatenate
from ._version import version

# Deprecated
CountsTableSparse = None
