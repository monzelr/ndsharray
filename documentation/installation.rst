============
Installation
============


From sources
------------

The sources for ndsharray can be found in the github `github repo`_.

You should clone the private repository:

.. code-block:: console

    $ git clone git://gitlab.com/monzelr/ndsharray

Best practice is just to link the source doe to your python environment via the develop command:

.. code-block:: console

    $ cd ndsharray
    $ pip install ndsharray

To uninstall the python package, type in this command:

.. code-block:: console

    $ pip uninstall ndsharray

Of course, you can also install the package with python:

.. code-block:: console

    $ python setup.py install

For deployment
--------------
If you want to distribute the package, please build a python wheel which can be distributed:

.. code-block:: console

    $ python setup.py build_ext
    $ python setup.py bdist_wheel

The wheel contains compiled machine code which is not readable for humans. Thus it can be deployed savely.
The wheel can be installed with the pip command.

.. _github repo: https://github.com/monzelr/ndsharray
