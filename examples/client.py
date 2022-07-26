"""
Continuously read images from a memory-mapped file and show them in console
"""
# python package
import time
import sys
import traceback

# external python packages
import numpy as np
from ndsharray import NdShArray


def main():
    try:
        _tag = sys.argv[1]
        print("connecting to tag %s" % _tag)
        shared_array = NdShArray(_tag, r_w='r')  # Note: r_w='r' must be specified

        while True:
            # read image
            _start_time = time.time()
            valid, array = shared_array.read()

            if valid:
                print("time for reading: %3.3f ms" % ((time.time()-_start_time) * 1000))
                print("elapsed time write/read: %3.3f ms" % shared_array.read_time_ms)
                print("array sum: %i" % np.sum(array))
                # print(array)
                print()

            time.sleep(0)
    except Exception:
        traceback.print_exc()
        input()



if __name__ == "__main__":

    main()

