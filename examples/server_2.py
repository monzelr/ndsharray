"""
Continuously extract random images and write them to a memory-mapped file.
"""
import time
import sys
import os
import subprocess
import uuid

from ndsharray import NdShArray
import numpy as np

# choose a shape
_shape_vid = (720, 1280, 3)  # normal video camera

_shape_bigdata = (10000, 5000)  # big data


def main():
    global _shape_bigdata, _shape_vid

    # create mapping NdShArray
    _tag = "My_NdShArray"
    print("using tag for sharing the numpy array: %s" % _tag)
    shared_array = NdShArray(_tag, r_w='w')  # Note: r_w='r' is must be specified

    # write array to the shared_array
    array = (255*np.random.random(_shape_vid).astype(np.float32)).astype(np.uint8)  # simulate a noisy image
    shared_array.write(array)

    _idx = 0

    try:
        while True:
            if _idx <= 5:
                array = (255*np.random.random(_shape_vid).astype(np.float32)).astype(np.uint8)  # simulate video image
            elif _idx <= 10:
                array = (255*np.random.random(_shape_bigdata).astype(np.float32)).astype(np.uint8)  # simulate big data
                print(array.shape)
            else:
                _idx = 0
                continue

            # write image
            start = time.perf_counter()
            shared_array.write(array)
            stop = time.perf_counter()
            print("Writing Duration:", (stop - start) * 1000, "ms")
            print("array shape: %s" % str(array.shape))
            print("array sum: %i" % np.sum(array))
            print()

            time.sleep(5)
            _idx += 1

    except KeyboardInterrupt:
        pass

    print("Please also close the client manually!")


if __name__ == "__main__":
    main()
