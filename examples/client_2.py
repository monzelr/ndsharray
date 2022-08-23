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
        _tag = "My_NdShArray"
        print("connecting to tag %s" % _tag)
        try:
            shared_array = NdShArray(_tag, r_w='r')  # Note: r_w='r' is necessary for POSIX OS
        except ValueError as e:
            print("Important Note:")
            print("You will get a ValueError 'embedded null character' if the server did not start the NdShArray! "
                  "The server must start first, followed by the client.")
            raise e


        while True:
            # read image
            _start_time = time.time()
            valid, array = shared_array.read()

            if valid:
                print("time for reading: %3.3f ms" % ((time.time()-_start_time) * 1000))
                print("elapsed time write/read: %3.3f ms" % shared_array.read_time_ms)
                print("array shape: %s" % str(array.shape))
                print("array sum: %i" % np.sum(array))
                # print(array)
                print()

            time.sleep(0)
    except Exception:
        traceback.print_exc()



if __name__ == "__main__":

    main()

