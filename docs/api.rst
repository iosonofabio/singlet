API reference
=============
Singlet analysis is centered around the :doc:`Dataset <api/dataset>` class, which describes a set of samples (usually single cells). Each `Dataset` has three main properties:

- a :doc:`CountsTable <api/counts_table>` with the counts of genomic features, typically transcripts
- a :doc:`SampleSheet <api/samplesheet>` with the sample metdata and phenotypic information.
- a :doc:`FeatureSheet <api/featuresheet>` with the feature metdata, for instance alternative names and Gene Ontology terms.

At least one of the three properties must be present. In fact, you are perfectly free to set only the feature counts or even, although may be not so useful, only the sample metadata. Moreover, a `Dataset` has a number of "action properties" that perform operations on the data:

- :doc:`Dataset.correlations <api/correlations>`: correlate feature expressions and phenotypes
- :doc:`Dataset.feature_selection <api/feature_selection>`: select features based on expression patterns
- :doc:`Dataset.dimensionality <api/dimensionality>`: reduce dimensionality of the data including phenotypes
- :doc:`Dataset.cluster <api/cluster>`: cluster samples, features, and phenotypes
- :doc:`Dataset.fit <api/fit>`: fit (regress on) feature expression and metadata
- :doc:`Dataset.plot <api/plot>`: plot the results of various analyses

Supporting modules are useful for particular purposes or internal use only:

- `config`
- `utils`
- `io`
