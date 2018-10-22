Example: Loom file
=========================
Loom files are becoming a common way of sharing single cell transcriptomic data. In a loom file, a counts table, a samplesheet, and a featuresheet are kept together inside a single file with an extension ``.loom``. ``singlet`` supports reading from loom files via config files.

Your ``singlet.yml`` must contain a section such as:

.. code-block:: yaml

     datasets:
      ds1:
        path: xxx.loom
        format: loom
        axis_samples: columns
        index_samples: Cell
        index_features: Gene 

Then you can load you ``Dataset`` easily:

.. code-block:: python

  from singlet.dataset import Dataset
  ds = Dataset(dataset='ds1')

To export a ``Dataset`` to a loom file, you can use the method ``to_dataset_file``:

.. code-block:: python

  from singlet.dataset import Dataset
  ds = Dataset(dataset='ds1')
  ds.to_dataset_file('xxx.loom')
