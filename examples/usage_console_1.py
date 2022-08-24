import numpy as np
import ndsharray

my_array = np.random.random((4, 2))  # array size can be changed during runtime

# array must not be specified at instantiation
shared_array = ndsharray.NdShArray("my_unique_tag123", array=my_array, r_w='w')

print("name of the ndsharray: %s" % shared_array.name)  # prints the unique identifier (uuid4)
print("access of the ndsharray: %s" % shared_array.access)  # prints access: 'r' or 'w'
print("current ndarray name: %s" % shared_array.ndarray_mmap_name)  # prints the current mmap name of the ndarray
print(my_array)


# second part
my_int_array = (255 * np.random.random((6, 3))).astype(np.uint8)
shared_array.write(my_int_array)

# the name of the ndarray mmap have been changed now - it is using a different uuid4 identifier:
print(shared_array.ndarray_mmap_name)
print(my_int_array)
print()

