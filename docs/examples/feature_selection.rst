Example: Feature Selection
==========================
It is typical in scRNA-Seq experiments to filter out features that are not expressed in any sample, or at low levels in very few samples. Moreover, of all remaining features, it is customary to select highly variable features for some applications such as dimensionality reduction.

.. code-block:: python

  from singlet.dataset import Dataset
  ds = Dataset(
          samplesheet='example_sheet_tsv',
          counts_table='example_table_tsv')

  ds.counts.normalize('counts_per_million', inplace=True)
  
  # This selects only genes that are present at >= 5 counts per million in at least 2 samples
  ds.feature_selection.expressed(
         n_samples=2,
         exp_min=5,
         inplace=True)

  # This selects highly variable features
  ds.feature_selection.overdispersed_strata(
         inplace=True)
