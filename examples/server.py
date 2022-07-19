"""
Continuously extract random images and write them to a memory-mapped file.
"""
import time
import sys
import os
import subprocess
import uuid

from ndsharray import ndsharray
import numpy as np


def main():

    # create mapping ndsharray
    _tag = uuid.uuid4().hex
    print("using tag for sharing the numpy array: %s" % _tag)
    shared_array = ndsharray(_tag)

    _client = os.path.join(os.path.abspath(os.path.dirname(__file__)), "client.py")

    # we do not care about forking/spawning etc.
    if os.name == "nt":  # windows
        subprocess.Popen([sys.executable, _client, _tag], creationflags=subprocess.CREATE_NEW_CONSOLE)
    elif os.name == "posix":  # linux / macOS
        subprocess.Popen([sys.executable, _client, _tag])

    try:
        while True:
            array = (255 * np.random.random((720, 1280, 3))).astype(np.uint8)  # simulate a noisy image

            # write image
            start = time.perf_counter()
            shared_array.write(array)
            stop = time.perf_counter()
            print("Writing Duration:", (stop - start) * 1000, "ms")
            print("array sum: %i" % np.sum(array))
            print()

            time.sleep(5)

    except KeyboardInterrupt:
        pass

    print("Please also close the client manually!")


if __name__ == "__main__":
    main()
