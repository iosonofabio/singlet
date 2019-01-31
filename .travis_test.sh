#!/bin/bash
if [ "$TRAVIS_OS_NAME" == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  PYTHON="$HOME/miniconda/bin/python$PYTHON_VERSION"
  PYTEST="$HOME/miniconda/bin/pytest"
else
  PYTHON=${PYTHON:-python}
  PYTEST=${PYTEST:-"pytest -rxXs --cov=singlet/"}
fi

export SINGLET_CONFIG_FILENAME='example_data/config_example.yml'

echo "python: ${PYTHON}"

echo 'Running pytests...'
# LOCAL TESTING:
# PYTHONPATH=$(pwd):PYTHONPATH SINGLET_CONFIG_FILENAME='example_data/config_example.yml' pytest -rxXs test

echo 'IO tests...'
${PYTEST} "test/io"
if [ $? != 0 ]; then
    exit 1
fi
echo 'done!'

echo 'Samplesheet tests...'
${PYTEST} "test/samplesheet"
if [ $? != 0 ]; then
    exit 1
fi
echo 'done!'

echo 'Counts table tests...'
${PYTEST} "test/counts_table"
if [ $? != 0 ]; then
    exit 1
fi
echo 'done!'

echo 'Dataset tests...'
${PYTEST} "test/dataset"
if [ $? != 0 ]; then
    exit 1
fi
echo 'done!'
