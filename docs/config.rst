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

The root key value pairs are:

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
The ``samplesheets`` section contains an arbitrary number of key value pairs and no lists. Each entry has the following format:
 - the key determines the id of a samplesheet: this id is used in the contstructor of ``Dataset``.
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
 - ``cells``: one of ``rows`` or ``columns``. If each row in the samplesheet is a sample, use ``rows``, else use ``columns``.

count_tables
_____________________



featuresheets
________________
