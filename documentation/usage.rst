=====
Usage
=====

Basics
------

NdShArray shall be used in different python processes. You can open another python process with python's subprocess
module or with python's multiprocessing module.

NdShArray uses shared memory, thus different CPUs can access the same numpy array. Note that NdShArray does
a memory view - exactly it uses the numpy function :code:`frombuffer`, while using the :code:`read` function.
However, the :code:`write` function does a copy into the shared memory.

To understand the libray open two different python console, and copy line for line the following code.

first console
_____________
In the first python console we create a random numpy array and create a unique identifier tag, which is important for
the reading ndsharray.

.. code-block:: python

    import numpy as np
    import ndsharray

    my_array = np.random.random((4, 2))  # array size can be changed during runtime

    # array must not be specified at instantiation
    shared_array = ndsharray.NdShArray("my_unique_tag123", array=my_array, r_w='w')

    print("name of the ndsharray: %s" % shared_array.name)  # prints the unique identifier (uuid4)
    print("access of the ndsharray: %s" % shared_array.access)  # prints access: 'r' or 'w'
    print("current ndarray name: %s" % shared_array.ndarray_mmap_name)  # prints the current mmap name of the ndarray
    print(my_array)


second console
______________
Open the second python console;  and copy the following code:

.. code-block:: python

    import numpy as np
    import ndsharray

    _shared_array = ndsharray.NdShArray("my_unique_tag123", r_w='r')

    status, my_array = _shared_array.read()

    print("status: %s" % status)  # check if the read is valid with the status!
    # check the read time: the elapsed time between write- and read-function
    print("elapsed write-read time: %s ms" % _shared_array.read_time_ms)
    print(my_array)

first console again
___________________
Ok, we just saw a successful transfer of the numpy array into the second process. But what if the numpy array size
change in dimension, dtype or shape? The class NdShArray carries about this and creates silently a new shared memory,
this can be seen by the property :code:`ndarray_mmap_name`.

.. note::
    The silently re-creation of a shared memory takes its time, so if you have 3 different-shaped numpy arrays which
    just updates its values over time, it is the best way to create 3 NdShArrays!

Go to the first console and enter the following code snippet:

.. code-block:: python

    my_int_array = (255 * np.random.random((6, 3))).astype(np.uint8)
    shared_array.write(my_int_array)

    # the name of the ndarray mmap have been changed now - it is using a different uuid4 identifier:
    print("current ndarray name: %s" % shared_array.ndarray_mmap_name)
    print(my_int_array)

second console again
____________________
Let's go to the second console and check the numpy array:

.. code-block:: python

    status, my_array_2 = _shared_array.read()

    print("status: %s" % status)  # check if the read is valid with the status!
    # check the read time: the elapsed time between write- and read-function
    print("elapsed write-read time: %s ms" % _shared_array.read_time_ms)
    print("current ndarray name: %s" % _shared_array.ndarray_mmap_name)
    print(my_array_2)

Supported numpy types
_____________________
To check the supported numpy types just take a look into :code:`ndsharray.supported_types`:

.. code-block:: python

    import ndsharray

    print("supported numpy types:")
    for _dtype in ndsharray.supported_types:
        print(_dtype)

As you can see, currently not all numpy dtypes are supported (e. g. :code:`bytes`, :code:`str` and :code:`object` data
types are missing).


More Example Code
_________________
More examples can be found on the github project in the example folder:
https://github.com/monzelr/ndsharray/tree/main/examples
