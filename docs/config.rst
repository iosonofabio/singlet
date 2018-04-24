Configuration
=============
Singlet is designed to work on separate projects at once. To keep projects tidy and independent, there are two layers of configuration.

The ``SINGLET_CONFIG_FILENAME`` environment variable
-----------------------------------------------------
If you set the environment variable ``SINGLET_CONFIG_FILENAME`` to point at a YAML file, singlet will use it to configure your current session. To use separate sessions in parallel, just prepend your scripts with::

  import os
  os.environ['SINGLET_CONFIG_FILENAME'] = '<full path to config file>'

and each will work totally independently.

The configuration file
----------------------
Singlet loads a configuration file in `YAML <http://www.yaml.org/start.html>`_ format when you import the ``singlet`` module. If you have not specified the location of this file with the ``SINGLET_CONFIG_FILENAME`` environment variable, it defaults to::

  <your home folder>/.singlet/config.yml

so the software will look there. An example configuration file is `online <https://github.com/iosonofabio/singlet/blob/master/example_data/config_example.yml>`_. If you are not familiar with YAML syntax, it is a bit like Python dictionaries without brackets... or like JSON.

Before going into the specifics, here's a schematic example of the configuration file:

.. code-block:: yaml

  io:
    samplesheets:
      ss1:
        path: xxx.csv
        index: samplename
    featuresheets:
      fs1:
        path: yyy.csv
        index: EnsemblGeneID
    count_tables:
      ct1:
        path: zzz.csv
        normalized: no
        spikeins:
          - ERCC-00002
          - ERCC-00003
        other:
          - __alignment_not_unique
          - __not_aligned

Now for the full specification, the root key value pairs are:

 - ``io``: for input/output specifications. At the moment this key is the only master key and is required.

There are no root lists.

io
~~~~~~~~
The ``io`` section has the following key value pairs:

 - ``samplesheets``: samplesheet files or Google Documents (for sample metadata).
 - ``featuresheets``: featuresheet files (for feature/gene annotations or metadata). 
 - ``count_tables``: count table files.

samplesheets
_______________
The ``samplesheets`` section contains an arbitrary number of key value pairs and no lists. Each entry describes a samplesheet and has the following format:
 - the key determines the id of the samplesheet: this id is used in the contstructor of ``Dataset``.
 - the value is a series of key value pairs, no lists.

Singlet can source samplesheets either from a local file or from an online Google Sheet. If you want to use a local file, use the following key value pairs:
 - ``path``: a filename on disk containing the samplesheet, usually in CSV/TSV format.
 - ``format``: a file format of the samplesheet (optional). If missing, it is inferred from the ``path`` filename.

If you prefer to source an online Google Sheet, use the following key value pairs:
 - ``url``: the URL of the spreadsheet, e.g. 'https://docs.google.com/spreadsheets/d/15OKOC48WZYFUQvYl9E7qEsR6AjqE4_BW7qcCsjJAD6w' for the example sheet.
 - ``client_id_filename``: a local filename (initially empty) where your login information for OAUTH2 is stored. This is a JSON file so this variable typically ends with ``.json``
 - ``client_secret_filename``: a local filename (initially empry) where your secret information for OAUTH2 is stored. This is a JSON file so this variable typically ends with ``.json``
 - ``sheet``: the name of the sheet with the data within the spreadsheet.

Whichever way you are using to source the data, the following key value pairs are available:
 - ``description``: a description of the sample sheet (optional).
 - ``cells``: one of ``rows`` or ``columns``. If each row in the samplesheet is a sample, use ``rows``, else use ``columns``. Notice that singlet samplesheets have samples as **rows**.
 -  ``index``: the name of the column/row of the samplesheet containing the sample names. This defaults to ``name`` (optional).

count_tables
_____________________
The ``count_tables`` section contains an arbitrary number of key value pairs and no lists. Each entry describes a counts table and has the following format:
 - the key determines the id of the counts table: this id is used in the contstructor of ``Dataset``.
 - the value is a series of key value pairs, no lists.

The following key value pairs are available:
 - ``description``: a description of the counts table (optional).
 - ``path``: a filename on disk containing the counts table, usually in CSV/TSV format.
 - ``format``: a file format of the counts table (optional). If missing, it is inferred from the ``path`` filename.
 - ``cells``: one of ``rows`` or ``columns``. If each row in the counts table is a sample, use ``rows``, else use ``columns``.
 - ``normalized``: either ``yes`` or ``no``. If data is not notmalized, you can normalize it with singlet by using the ``CountsTable.normalize`` method.
 - ``spikeins``: a YAML list of features that appear in the counts table and represent spike-in controls as opposed to real features. Spikeins can be excluded from the counts table using ``CountsTable.exclude_features``.
 - ``other``: a YAML list of features that are neither biological features nor spike-in controls. This list typically includes ambiguous alignments, multiple-aligned reads, reads outside features, etc. Other features can be excluded from the counts table using ``CountsTable.exclude_features``.

 The first column/row of the counts table must be the list of samples.


featuresheets
________________
The ``featuresheets`` section contains an arbitrary number of key value pairs and no lists. Each entry describes a featuresheet, i.e. a table with metadata for the features. A typical usage of featuresheets is to connect feature ids (e.g. ``EnsemblGeneID``) with human-readable names, Gene Ontology terms, species information, pathways, cellular localization, etc.Each entry has the following format:
 - the key is the id of the featuresheet: this id is used in the constructor of ``Dataset``.
 - the value is a series of key value pairs, no lists.

The following key value pairs are available:
 - ``description``: a description of the featuresheet (optional).
 - ``path``: a filename on disk containing the featuresheet, usually in CSV/TSV format.
 - ``format``: a file format of the featuresheet (optional). If missing, it is inferred from the ``path`` filename.
 - ``features``: one of ``rows`` or ``columns``. If each feature in the featuresheet is a feature, use ``rows``, otherwise use ``columns``.
 -  ``index``: the name of the column/row of the featuresheet containing the feature names. This defaults to ``name`` (optional).

