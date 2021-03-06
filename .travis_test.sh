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

${PYTEST} "test"
