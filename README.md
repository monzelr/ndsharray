ndsharray
=========
ndarray + sharing = ndsharray

Overview
--------
This python module let you share numpy ndarray with difference processes (either via python's multiprocessing or sharing between different python instances). The library behind this package is the lib [mmap](https://docs.python.org/3/library/mmap.html) from official python - no extra library is needed.
This library is eas to use, just intialize the shared array with unique tag:

```python
import numpy as np
from ndsharray import ndsharray
    
shared_array = ndsharray("my_unique_tag")

my_array = np.random.random((400, 800))
shared_array.write(my_array)

print(my_array)
```

That's all for writing into shared memory. How to read? Open a second python instance:
```python
import numpy as np
from ndsharray import ndsharray

shared_array = ndsharray("my_unique_tag")

status, my_array = shared_array.read()

print(my_array)
```

Some technical notes
--------------------
This library shall be an easy to use library and also shall be faster than pickling numpy arrays to another process. Please note that the python's provided [shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html) is a faster approach than ndsharray! However, shared_memory is available since python 3.8 and not supported for python 3.6. 


Requirements
------------ 
- Python ≥ 3.6
- numpy
- currently only for Windows amd64

Note: Tested on Windows 10, amd64, Python 3.6 / 3.8 \
**Will currently not work on Linux, MacOS and on AARCH64 devices (ARM devices like Raspberry PI).**


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
- Unit Test
- Documentation
- implementation for Linux
- im for the faster approach with [shared_memory](https://docs.python.org/3/library/multiprocessing.shared_memory.html)
