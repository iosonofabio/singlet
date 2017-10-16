API reference
=============
Singlet analysis is centered around the `Dataset` class, which describes a set of samples (usually single cells). Each `Dataset` has two main properties:

- a `CountsTable` with the counts of genomic features, typically transcripts
- a `SampleSheet` with the metdata and phenotypic information.

Moreover, a `Dataset` has a number of "action properties" that perform operations on the data:

- `Dataset.correlations`: correlate feature expressions and phenotypes
- `Dataset.dimensionality`: reduce dimensionality of the data including phenotypes
- `Dataset.cluster`: cluster samples, features, and phenotypes
- `Dataset.plot`: plot the results of various analyses

Supporting modules are useful for particular purposes or internal use only:

- `config`
- `utils`
- `io`

singlet\.counts\_table module
-----------------------------

.. automodule:: singlet.counts_table
    :members:
    :undoc-members:
    :show-inheritance:

singlet\.samplesheet module
---------------------------

.. automodule:: singlet.samplesheet
    :members:
    :undoc-members:
    :show-inheritance:

singlet\.dataset module
---------------------------
.. automodule:: singlet.dataset
    :members:
    :undoc-members:
    :show-inheritance:

`Dataset` action properties
---------------------------

singlet\.dataset\.correlations module
-------------------------------------

.. automodule:: singlet.dataset.correlations
    :members:
    :undoc-members:
    :show-inheritance:


singlet\.dataset\.feature_selection module
---------------------------------------

.. automodule:: singlet.dataset.feature_selection
    :members:
    :undoc-members:
    :show-inheritance:


singlet\.dataset\.dimensionality module
---------------------------------------

.. automodule:: singlet.dataset.dimensionality
    :members:
    :undoc-members:
    :show-inheritance:


singlet\.dataset\.cluster module
-------------------------------------

.. automodule:: singlet.dataset.cluster
    :members:
    :undoc-members:
    :show-inheritance:


singlet\.dataset\.fit module
-------------------------------------

.. automodule:: singlet.dataset.fit
    :members:
    :undoc-members:
    :show-inheritance:


singlet\.dataset\.plot module
-------------------------------------

.. automodule:: singlet.dataset.plot
    :members:
    :undoc-members:
    :show-inheritance:
