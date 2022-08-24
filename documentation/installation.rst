============
Installation
============

Via pip
-------

This module can be found on pipy: `https://pypi.org/project/ndsharray/`_

For installation, first activate your environment, and then install ndsharray:

.. code-block:: console

    pip install ndsharray

To uninstall the python package, type in this command:

.. code-block:: console

    pip uninstall ndsharray


From sources
------------

The sources for ndsharray can be found in the github `github repository`_.

Clone the repository:

.. code-block:: console

    git clone git://gitlab.com/monzelr/ndsharray

Best practice is just to link the source to to your python environment via the develop command:

.. code-block:: console

    cd ndsharray
    python setup.py develop

Of course, you can also install the package with python normally:

.. code-block:: console

    $ python setup.py install

To uninstall the python package, type in this command:

.. code-block:: console

    pip uninstall ndsharray



Distribution
------------
If you want to distribute the package, please build a python wheel which can be distributed:

.. code-block:: console

    python setup.py bdist_wheel

The wheel can be installed with the pip command.

.. _github repository: https://github.com/monzelr/ndsharray
.. _https://pypi.org/project/ndsharray/: https://pypi.org/project/ndsharray/
