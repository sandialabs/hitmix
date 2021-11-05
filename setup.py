# Copyright 2021 National Technology & Engineering Solutions of Sandia,
# LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the
# U.S. Government retains certain rights in this software.

from setuptools import setup

setup(
    name='hitmix',
    version='1.0.0',
    packages=['hitmix'],
    package_dir={'': '.'},
    url='',
    license='BSD-3',
    author='Danny Dunlavy',
    author_email='dmdunla@sandia.gov',
    description='Python Implementation of HITMIX',
    install_requires=[
        "numpy",
        "pytest",
        "sphinx_rtd_theme"
    ]
)
