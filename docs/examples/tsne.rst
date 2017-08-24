Example: t-SNE
==============
t-SNE [tsne]_ is a commonly used algorithm to reduce dimensionality in single cell data.

.. code-block:: python

  from singlet.dataset import Dataset
  ds = Dataset(
          samplesheet='example_sheet_tsv',
          counts_table='example_table_tsv')

  ds.counts.normalize('counts_per_million', inplace=True)
  ds.counts = ds.counts.iloc[:200]

  print('Calculate t-SNE')
  vs = ds.dimensionality.tsne(
          n_dims=2,
          transform='log10',
          theta=0.5,
          perplexity=0.8)

  print('Plot t-SNE')
  ax = ds.plot.scatter_reduced_samples(
          vs,
          color_by='quantitative_phenotype_1_[A.U.]')

  plt.show()

You should get figures similar to the following ones:

.. image:: ../_static/example_tsne.png
   :width: 600
   :alt: tsne

.. [tsne] L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008.
