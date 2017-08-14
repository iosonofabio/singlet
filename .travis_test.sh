#!/bin/bash
if [ "$TRAVIS_OS_NAME" == 'osx' ]; then
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  PYTHON="$HOME/miniconda/bin/python$PYTHON_VERSION"
else
  PYTHON=${PYTHON:-python}
fi

echo "python: ${PYTHON}"

echo 'Running tests...'

echo 'IO tests...'
${PYTHON} "test/test_io.py"
if [ $? != 0 ]; then
    exit 1
fi
echo 'done!'

echo 'Dataset tests...'
${PYTHON} "test/test_dataset.py"
if [ $? != 0 ]; then
    exit 1
fi
echo 'done!'

