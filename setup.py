#!/usr/bin/env python
# vim: fdm=indent
'''
author:     Fabio Zanini
date:       08/08/17
content:    Setup script for singlet
'''
import sys
import os
from distutils.log import INFO as logINFO

if ((sys.version_info[0] < 3) or
   (sys.version_info[0] == 3 and sys.version_info[1] < 4)):
    sys.stderr.write("Error in setup script for singlet:\n")
    sys.stderr.write("Singlet supports Python 3.4+.")
    sys.exit(1)


# Setuptools but not distutils support build/runtime/optional dependencies
try:
    from setuptools import setup, Extension, find_packages
    from setuptools.command.build_py import build_py
    from setuptools import Command
    kwargs = dict(
        setup_requires=[
            'PyYAML',
            'numpy',
            'pandas',
            'matplotlib',
        ],
        install_requires=[
            'PyYAML',
            'numpy',
            'pandas',
            'matplotlib',
        ],
      )
except ImportError:
    sys.stderr.write("Could not import 'setuptools'," +
                     " falling back to 'distutils'.\n")
    from distutils.core import setup, Extension, find_packages
    from distutils.command.build_py import build_py
    from distutils.cmd import Command
    kwargs = dict(
        requires=[
            'PyYAML',
            'numpy',
            'pandas',
            'matplotlib',
            ]
    )


# Get version
with open('singlet/_version.py') as fversion:
    version = fversion.readline().rstrip()


# Setup function
setup(name='singlet',
      version=version,
      author='Fabio Zanini',
      author_email='fabio.zanini@stanford.edu',
      maintainer='Fabio Zanini',
      maintainer_email='fabio.zanini@stanford.edu',
      url='https://github.com/iosonofabio/singlet',
      description="Single cell RNA Seq analysis",
      long_description="""
      Single cell RNA Seq analysis.

      **Development**: https://github.com/iosonofabio/singlet

      **Documentation**: http://singlet.readthedocs.io""",
      license='GPL3',
      classifiers=[
         'Development Status :: 3 - Alpha',
         'Topic :: Scientific/Engineering :: Bio-Informatics',
         'Intended Audience :: Developers',
         'Intended Audience :: Science/Research',
         'License :: OSI Approved :: GNU General Public License (GPL)',
         'Operating System :: POSIX',
         'Programming Language :: Python'
      ],
      packages=['singlet'] + ['singlet.' + s for s in find_packages(where='singlet')],
      **kwargs
      )
