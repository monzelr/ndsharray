import numpy as np
import ndsharray

_shared_array = ndsharray.NdShArray("my_unique_tag123", r_w='r')

status, my_array = _shared_array.read()

print("status: %s" % status)  # check if the read is valid with the status!
# check the read time: the elapsed time between write- and read-function
print("elapsed write-read time: %s ms" % _shared_array.read_time_ms)
print(my_array)


# second part
status, my_array_2 = _shared_array.read()

print("status: %s" % status)  # check if the read is valid with the status!
# check the read time: the elapsed time between write- and read-function
print("elapsed write-read time: %s ms" % _shared_array.read_time_ms)
print("current ndarray name: %s" % _shared_array.ndarray_mmap_name)
print(my_array_2)

# third part
print("supported numpy types:")
for _dtype in ndsharray.supported_types:
    print(_dtype)

