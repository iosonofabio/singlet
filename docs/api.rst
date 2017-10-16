API reference
=============
Singlet analysis is centered around the `Dataset` class, which describes a set of samples (usually single cells). Each `Dataset` has three main properties:

- a :doc:`CountsTable <api/counts_table>` with the counts of genomic features, typically transcripts
- a :doc:`SampleSheet <api/samplesheet>` with the sample metdata and phenotypic information.
- a :doc:`FeatureSheet <api/featuresheet>` with the feature metdata, for instance alternative names and Gene Ontology terms.

At least one of the three properties must be present. In fact, you are perfectly free to set only the feature counts or even, although may be not so useful, only the sample metadata. Moreover, a `Dataset` has a number of "action properties" that perform operations on the data:

- `Dataset.correlations`: correlate feature expressions and phenotypes
- `Dataset.feature_selection`: select features based on expression patterns
- `Dataset.dimensionality`: reduce dimensionality of the data including phenotypes
- `Dataset.cluster`: cluster samples, features, and phenotypes
- `Dataset.fit`: fit (regress on) feature expression and metadata
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
