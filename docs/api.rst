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

singlet\.dataset module
---------------------------
.. automodule:: singlet.dataset
    :members:
    :undoc-members:
    :show-inheritance:

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
