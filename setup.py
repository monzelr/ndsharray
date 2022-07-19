#!/usr/bin/env python

"""The setup script."""

import sys
import os
import io
import os
import re

from setuptools import setup

SOFTWARE_NAME = "ndsharray"


def read(*names, **kwargs):
    """Python 2 and Python 3 compatible text file reading.
    Required for single-sourcing the version string.
    """
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version_author_email(*file_paths):
    """
    Search the file for a specific string.
    file_path contain string path components.
    Reads the supplied Python module as text without importing it.
    """
    _version = _author = _email = ""
    file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", file, re.M)
    author_match = re.search(r"^__author__ = ['\"]([^'\"]*)['\"]", file, re.M)
    email_match = re.search(r"^__email__ = ['\"]([^'\"]*)['\"]", file, re.M)
    if version_match:
        _version = version_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")
    if author_match:
        _author = author_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")
    if email_match:
        _email = email_match.group(1)
    else:
        raise RuntimeError("Unable to find version string.")
    return _version, _author, _email


version, author, author_email = find_version_author_email(SOFTWARE_NAME, '__init__.py')


# this is only necessary when not using setuptools/distribute
from sphinx.setup_command import BuildDoc


requirements = ["numpy>=1.13.0",
                "sphinx"]

test_requirements = ["numpy>=1.13.0"]

python_requires = '>=3.6'
classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Natural Language :: English',
    'License :: OSI Approved :: BSD License',
    'Operating System :: POSIX',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
]
description = "Sharing numpy ndarray with a simple interface between different (sub)processes."
install_requires = requirements
long_description = ""
keywords = 'sharing numpy arrays, inter process communication, subprocess, multiple process sharing one ndarray'
name = SOFTWARE_NAME
test_suite = 'tests'
url = 'https://gitlab.com/monzelr/' + SOFTWARE_NAME
zip_safe = False

setup(
    author=author,
    author_email=author_email,
    python_requires=python_requires,
    classifiers=classifiers,
    description=description,
    install_requires=requirements,
    keywords=keywords,
    name=name,
    packages=[SOFTWARE_NAME],
    cmdclass={'build_sphinx': BuildDoc},
    test_suite=test_suite,
    tests_require=test_requirements,
    url=url,
    license="BSD 3-Clause License",
    version=version,
    zip_safe=zip_safe,
    command_options = {
                      'build_sphinx': {
                          'project': ('setup.py', name),
                          'version': ('setup.py', version),
                          'release': ('setup.py', version),
                          'source_dir': ('setup.py', 'documentation')}}
)

