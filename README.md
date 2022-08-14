ndsharray
=========
ndarray + sharing = ndsharray

Overview
--------
This python module let you share numpy ndarray with difference processes (either via python's multiprocessing or sharing between different python instances). The library behind this package is the lib [mmap](https://docs.python.org/3/library/mmap.html) from official python - no extra library is needed. The mmap approach is much faster than the pickle approach - you can even do a video streaming on a Raspberry Pi / Jetson Nano over multiple python processes. \
This library is eas to use, just intialize the shared array with unique tag and write/read! On a Windows OS, you can even change the numpy array size/shape/dtype during runtime - the mmap will be automatically rebuild if there is a change in the numpy array size/shape/dtype.

Small Example Code:
```python
import numpy as np
from ndsharray import ndsharray
    
shared_array = ndsharray("my_unique_tag", r_w='w')  # r_w must be specified on posix (linx/mac) operation systems

my_array = np.random.random((400, 800))
shared_array.write(my_array)

print(my_array)
```

That's all for writing into shared memory. How to read? Open a second python instance:
```python
import numpy as np
from ndsharray import ndsharray

shared_array = ndsharray("my_unique_tag", r_w='r')  # r_w must be specified on posix (linux/mac) operation systems

status, my_array = shared_array.read()

print(my_array)
```

Some technical notes
--------------------
This library shall be an easy to use library and also shall be faster than pickling numpy arrays to another process. Please note that the python's provided [shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) is maybe a faster approach than ndsharray! However, shared_memory is available since python 3.8 and not supported for python 3.6.
The pefomance of this library is good enough for video streaming (see also example)!

Example Code
------------
In folder ndsharray/examples is an example. This example shows a simple Server / Client connection sharing a numpy array every 5 seconds. 
```
python server.py
```
Note: Always start the ndsharray-writer first! If you start the reader first, you may will get an **Access Denied Error**!

Requirements
------------ 
- Python ≥ 3.6
- numpy

Tested with example codes on 
- Windows 10, amd64, Python 3.6 / 3.8
- Ubuntu 20, amd64, Python 3.6 /3.8
- Ubuntu 18.04, aarch64 (NVIDIA Jetson Nano), Python 3.6


Installation
------------
Make sure to have git, python and pip in your environment path or activate your python environment.\
To install enter in cmd/shell:

    git clone https://github.com/monzelr/ndsharray.git

    cd ndsharray

    pip install .

Alternative with python:

    python setup.py install


To Dos
------
- Unit Tests of all numpy arrays
- Documentation
- implement the faster approach with [shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
