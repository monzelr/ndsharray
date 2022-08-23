ndsharray
=========
ndarray + sharing = ndsharray

Overview
--------
This python module let you share a numpy ndarray within different processes (either via python's multiprocessing or sharing between different python instances). The library behind this package is the lib [mmap](https://docs.python.org/3/library/mmap.html) from official python - no extra library, except numpy, is needed. The mmap approach is much faster than the pickle approach - you can even do a video streaming on a Raspberry Pi / Jetson Nano over multiple python processes. \
This library is eas to use, just initialize the shared array with a unique tag and write/read! You can even change the numpy array size/shape/dtype during runtime - the mmap will be silently rebuild if there is a change in the numpy array size/shape/dtype.

Small Example Code:
```python
import numpy as np
from ndsharray import NdShArray
    
shared_array = NdShArray("my_unique_tag", r_w='w')  # r_w must be specified

my_array = np.random.random((400, 800))
shared_array.write(my_array)

print(my_array)
```

That's all for writing into shared memory. How to read? Open a second python instance:
```python
import numpy as np
from ndsharray import NdShArray

shared_array = NdShArray("my_unique_tag", r_w='r')  # r_w must be specified

status, my_array = shared_array.read()

print(my_array)
```

Some technical notes
--------------------
This library shall be an easy-to-use library and also shall be faster than pickling numpy arrays to another process. Please note that the python's provided [shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) does the same as ndsharray, but is using byte array instead of numpy array! However, shared_memory is available since python 3.8 and not supported for python 3.6.
The performance of this library is good enough for video streaming (see also example)!

Example Code
------------
In folder ndsharray/examples are 2 different examples. Both examples show a simple Server / Client connection sharing a numpy array every 5 seconds. 
```
python server.py
```
Note: Always start the ndsharray-writer first! If you start the reader first, you will get a **ValueError**!

Requirements
------------ 
- Python ≥ 3.6
- numpy

Tested with example codes on 
- Windows 10, amd64, Python 3.6 / 3.8
- Ubuntu 20, amd64, Python 3.6 /3.8
- NVIDIA Jetson Nano with Ubuntu 18.04, ARM64-bit (aarch64), Python 3.6


Installation
------------
Make sure to have git, python and pip in your environment path or activate your python environment.\
To install enter in cmd/shell:

    git clone https://github.com/monzelr/ndsharray.git

    cd ndsharray

    pip install .

Alternative with python:

    python setup.py install

