Example: Split and compare
==========================
Singlet allows you to split a dataset based on metadata in a single line. Moreover, it is easy to perform statistical comparisons between two datasets, comparing feature expression and/or phenotypes with any statistical test you like.

.. code-block:: python

  from singlet.dataset import Dataset
  ds = Dataset(
          samplesheet='example_sheet_tsv',
          counts_table='example_table_tsv')

  ds.counts.normalize('counts_per_million', inplace=True)

  # Split dataset based on metadata
  dataset_dict = ds.split('experiment')

  # Statistical comparison of features between datasets
  dataset_dict['test_pipeline'].compare(
         dataset_dict['exp1'],
         method='mann-whitney')

.. note::
  Mann-Whitney's U test and two sample Kolmogorov-Smirnov's test are built-ins, but you can just set `method` to any function you want that calculates the P-values.
