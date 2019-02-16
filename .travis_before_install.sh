#!/bin/bash
echo "Prepare interpreter"
if [ $TRAVIS_OS_NAME == 'linux' ]; then
  echo "Installing deps for linux"
  #sudo add-apt-repository ppa:nschloe/swig-backports -y
  #sudo apt-get -qq update
  #sudo apt-get install -y swig3.0
elif [ $TRAVIS_OS_NAME == 'osx' ]; then
  echo "Installing deps for OSX"
  if [ $PYTHON_VERSION == "2.7" ]; then
    CONDA_VER='2'
  elif [ $PYTHON_VERSION == "3.7" ]; then
    CONDA_VER='3'
  else
    echo "Miniconda only supports 2.7 and 3.7"
  fi
  curl "https://repo.continuum.io/miniconda/Miniconda${CONDA_VER}-latest-MacOSX-x86_64.sh" -o "miniconda.sh"
  bash "miniconda.sh" -b -p $HOME/miniconda
  echo "$PATH"
  export PATH="$HOME/miniconda/bin:$PATH"
  source $HOME/miniconda/bin/activate
  # Use pip from conda
  conda install -y pip
  pip --version

else
  echo "OS not recognized: $TRAVIS_OS_NAME"
  exit 1
fi

echo "Installing dependencies"
## setuptools < 18.0 has issues with Cython as a dependency
#pip install Cython
#if [ $? != 0 ]; then
#    exit 1
#fi

# deps #FIXME: do better
pip install pytest
pip install pytest-cov
pip install coveralls

pip install pyyaml
pip install numpy
pip install scipy
pip install pandas
pip install xarray
pip install loompy
pip install scikit-learn
pip install matplotlib
pip install seaborn
# NOTE: one day they shall fix this (sigh!)
pip install Cython
#pip install bhtsne
# Google API tests are only local anyway
#pip install google-api-python-client
#pip install numba
#pip install umap-learn
pip install sam-algorithm

# NOTE: lshknn requires eigen which is purely headers. It finds the headers via pkgconfig
# which is the Python wrapper of pkg-config. Let's see how far we get with this on OSX
# TODO: install eigen happens in the .travis.yml
pip install pkgconfig
pip install pybind11
pip install lshknn
