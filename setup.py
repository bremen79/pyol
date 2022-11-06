# Always prefer setuptools over distutils
from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open
from os import path

# The directory containing this file
HERE = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(HERE, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# This call to setup() does all the work
setup(
    name="PyOL",
    version="0.0.2",
    description="A Python library for online learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bremen79/pyol",
    author="Francesco Orabona",
    author_email="francesco@orabona.com",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        "Operating System :: OS Independent"
    ],
    packages=["pyol"],
    include_package_data=True,
    install_requires=["numpy"]
)
