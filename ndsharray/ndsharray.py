# python packages
import os
from mmap import mmap  # python >= 3.7: ACCESS_READ, ACCESS_DEFAULT, ACCESS_WRITE
if os.name == "posix":
    from mmap import MAP_SHARED, PROT_WRITE, PROT_READ
import time
import sys
from typing import Union, List, Tuple
import struct

# external python packages
import numpy as np

"""
:var supported_types: supported numpy dtype
"""
supported_types = []
for key, values in np.sctypes.items():
    for value in values:
        supported_types.append(value)

"""
:var nbytes_for_int: get the number of bytes for a python integer, this is python and system dependent
"""
nbytes_for_int = len(sys.maxsize.to_bytes((sys.maxsize.bit_length() + 7) // 8, 'big'))


def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    """
    converts integer to bytes

    :param i:
    :param signed:
    :return:
    """
    global nbytes_for_int
    return i.to_bytes(nbytes_for_int, byteorder='big', signed=signed)


def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    """
    converts bytes to an integer

    :param b:
    :param signed:
    :return:
    """
    return int.from_bytes(b, byteorder='big', signed=signed)


class ndsharray(object):
    """
    sharing numpy array between different processes

    """

    def __init__(self, tag_name: str, array: np.ndarray = np.ndarray((0, ), dtype=np.uint8),
                 r_w: Union[str, None] = None):
        """
        :param tag_name:
        :param array:
        :param r_w: 'r' or 'w' for 'read' or 'write' functionality, needed for linux/macOS operation systems only
        """
        object.__init__(self)

        # initialize last read time
        self._last_write_time = time.monotonic()
        self._read_time_ms = 0.0

        # save numpy array and its properties, this array is used for reading and writing
        self._array = array.copy()

        # convert array to bytes
        _bytes = self._array_to_bytes(array)
        self._buffer_size = len(_bytes)
        self._tag_name = tag_name

        # initialize mmap
        if os.name == "nt":
            self._mmap: mmap = mmap(-1, self._buffer_size, self._tag_name)
        elif os.name == "posix":
            if r_w == "w":
                self._fd = os.open("/tmp/%s" % self._tag_name, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
                os.truncate("/tmp/%s" % self._tag_name, self._buffer_size)  # resize file
                self._mmap: mmap = mmap(self._fd, self._buffer_size, MAP_SHARED)
            elif r_w == "r":
                self._fd = os.open("/tmp/%s" % self._tag_name, os.O_RDONLY)
                # os.truncate("/tmp/%s" % self._tag_name, self._buffer_size)  # resize file
                self._mmap: mmap = mmap(self._fd, self._buffer_size, MAP_SHARED, PROT_READ)
            else:
                raise ValueError("'r' or 'w' must be specified for posix operation system.")
        else:
            raise OSError("%s is not supported." % os.name)

    def __del__(self):
        # closing the file
        if os.name == "posix":
            os.close(self._fd)
            # _closed = False
            # while not _closed:
            #     try:
            #         _stat = os.fstat(self._fd)
            #         print("while True")
            #     except OSError as oe:
            #         print(oe)
            #         _closed = True

        # closing the mmap
        self._mmap.close()
        while not self._mmap.closed:
            time.sleep(0.001)

    @property
    def read_time_ms(self) -> float:
        """
        returns the write-read time of the two processes in milliseconds

        :return:
        """
        return self._read_time_ms

    @staticmethod
    def _array_to_bytes(array: np.ndarray) -> bytes:
        """
        encodes a numpy array to bytes using an own protocol

        protocol usage:
        - write-time (8 bytes)
        - numpy dtype index (integer, 8 bytes)
        - number of dimension (integer, 8 bytes)
        - length of axis (array dimension) 0 (integer, 8 bytes)
        - length of axis (array dimension) 1 (integer, 8 bytes)
        - length of axis (array dimension) 2 (integer, 8 bytes)
        - length of axis (array dimension) . (integer, 8 bytes)
        - length of axis (array dimension) . (integer, 8 bytes)
        - length of axis (array dimension) n (integer, 8 bytes)
        - bytes of numpy array
        - write-time (8 bytes)

        note: size of integer may defer because the maximum integer size sys.maxsize will be used (on python3, amd64
        it is 8 byte)

        :param bytes: byte-encoded numpy array using an own protocol
        :return:
        """
        global supported_types

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be from type np.ndarray.")

        if array.dtype not in supported_types:
            raise NotImplementedError("%s is a numpy.dtype which is not supported. "
                                      "The following numpy.dtypes are supported: %s"
                                      % (str(array.dtype), str([_t.__name__ for _t in supported_types])))

        _time = struct.pack("d", time.monotonic())

        _bytes = b''
        _bytes += _time
        _bytes += int_to_bytes(supported_types.index(array.dtype))
        _bytes += int_to_bytes(int(array.ndim))
        for s in range(array.ndim):
            _bytes += int_to_bytes(int(array.shape[s]))
        _bytes += array.tobytes()
        _bytes += _time

        return _bytes

    def _bytes_to_array(self, _bytes: bytes) -> Tuple[bool, bool, np.ndarray]:
        """

        :param _bytes:
        :return mmap_correct: boolean shows, if the mmap does fit to the size of the numpy ndarray, if it is not
                              correct, the mmap should be re-initialized and the buffer should be read out again
                              if mmap_correct is False, validity will be also False and the numpy array will be
                              empty
        :return validity: boolean displaying if the numpy array is corrupt or not (e.g. mixed numpy ndarray from
                          previous writing
        :return array: numpy.ndarray, mmap_correct and validity must be True, otherwise this array contains corrupt
                       data
        """
        global supported_types

        _mmap_correct = True
        _validity = False
        _array = np.ndarray((0, ))

        idx = 0
        _time_start = struct.unpack("d", _bytes[idx:8])[0]
        idx += 8
        _np_dtype = supported_types[bytes_to_int(_bytes[idx:idx+nbytes_for_int])]
        idx += nbytes_for_int
        if _np_dtype != self._array.dtype:
            return False, False, _array
        _np_dim = bytes_to_int(_bytes[idx:idx+nbytes_for_int])
        idx += nbytes_for_int
        if _np_dim != self._array.ndim:
            return False, False, _array
        _np_shape = []
        for s in range(_np_dim):
            _np_shape.append(bytes_to_int(_bytes[idx:idx + nbytes_for_int]))
            idx += nbytes_for_int
        _np_shape = tuple(_np_shape)
        if _np_shape != self._array.shape:
            return False, False, _array
        _byte_array = _bytes[idx:-8]
        idx = len(_bytes) - 8
        try:
            _time_end = struct.unpack("d", _bytes[idx:])[0]
        except ValueError:
            _time_end = 0

        _validity = _time_start == _time_end

        # check for mmap changes
        _array = np.frombuffer(_byte_array, dtype=_np_dtype).reshape(_np_shape)

        return _mmap_correct, _validity, _array

    def write(self, array: np.ndarray) -> None:
        """

        :param array: a numpy.ndarray which shall be saved in mmap
        :return:
        """
        _bytes = self._array_to_bytes(array)

        # check, if a new mmap has to be generated
        if self._array.dtype != array.dtype or self._array.ndim != array.ndim or self._array.shape != array.shape:
            self._array = array.copy()
            self._buffer_size = len(_bytes)

            if os.name == "posix":
                os.close(self._fd)
                _closed = False
                # while not _closed:
                #     try:
                #         _stat = os.fstat(self._fd)
                #         print("while True")
                #     except OSError as oe:
                #         print(oe)
                #         _closed = True

            self._mmap.close()
            while not self._mmap.closed:
                time.sleep(0.001)

            if os.name == "nt":
                self._mmap: mmap = mmap(-1, self._buffer_size, self._tag_name)  # , access=ACCESS_WRITE
            elif os.name == "posix":
                self._fd = os.open("/tmp/%s" % self._tag_name, os.O_TRUNC | os.O_RDWR)
                os.truncate("/tmp/%s" % self._tag_name, self._buffer_size)  # resize file
                self._mmap: mmap = mmap(self._fd, self._buffer_size, MAP_SHARED, PROT_WRITE)
            else:
                raise OSError("%s is not supported." % os.name)

        self._mmap.seek(0)
        self._mmap.write(_bytes)
        self._mmap.flush()

    def read(self) -> Tuple[bool, np.ndarray]:
        """

        :return validity: boolean displaying if the numpy array is ok or if it is either old or corrupt or not (e.g.
                          mixed numpy ndarray from previous writing). Note: validity is checked by checking if
                          buffer[0] and buffer[-1] have the same time stamp!
        :return array: numpy.ndarray, mmap_correct and validity must be True, otherwise this array contains corrupt data
        """
        global nbytes_for_int, supported_types

        _mmap_correct = True
        _validity = False
        _numpy_array = self._array

        # first stage of checking if new data have been arrived
        self._mmap.seek(0)
        _bytes = self._mmap.read(8)
        try:
            _write_time = struct.unpack("d", _bytes)[0]
        except ValueError:
            _write_time = 0
        if _write_time <= self._last_write_time:
            return False, _numpy_array

        # without checking, read the whole buffer
        _bytes += self._mmap.read()

        if len(_bytes) == self._buffer_size:
            # read time, dtype and ndim
            _mmap_correct, _validity, _numpy_array = self._bytes_to_array(_bytes)
        else:
            _mmap_correct = False

        if not _mmap_correct:
            # wrong mmap: do a 'save' read and re-build the mmap
            _validity, _numpy_array = self._read_and_rebuild_mmap()

        # for efficiency
        self._array = _numpy_array
        # for debug purpose
        self._read_time_ms = (time.monotonic()-_write_time) * 1000.0
        self._last_write_time = _write_time

        return _validity, _numpy_array

    def _read_and_rebuild_mmap(self) -> Tuple[bool, np.ndarray]:
        """

        :return validity: boolean displaying if the numpy array is corrupt or not (e.g. mixed numpy ndarray from
                          previous writing
        :return array: numpy.ndarray, mmap_correct and validity must be True, otherwise this array contains corrupt data
        """
        global nbytes_for_int, supported_types

        # read dtype and dimension of numpy array
        self._mmap.seek(0)
        _bytes = self._mmap.read(8+2*nbytes_for_int)  # skip the time: +8
        idx = 8
        _np_dtype = supported_types[bytes_to_int(_bytes[idx:idx+nbytes_for_int])]
        idx += nbytes_for_int
        _np_dim = bytes_to_int(_bytes[idx:idx+nbytes_for_int])
        idx += nbytes_for_int

        # read shape
        _bytes += self._mmap.read(_np_dim*nbytes_for_int)
        _np_shape = []
        for s in range(_np_dim):
            _np_shape.append(bytes_to_int(_bytes[idx:idx + nbytes_for_int]))
            idx += nbytes_for_int
        _np_shape = tuple(_np_shape)

        # rebuild _array and get the length
        self._array = np.ndarray(_np_shape, dtype=_np_dtype)
        _bytes = self._array_to_bytes(self._array)
        self._buffer_size = len(_bytes)

        if os.name == "posix":
            os.close(self._fd)
            # _closed = False
            # while not _closed:
            #     try:
            #         _stat = os.fstat(self._fd)
            #         print("while True")
            #     except OSError as oe:
            #         print(oe)
            #         _closed = True

        self._mmap.close()
        while not self._mmap.closed:
            time.sleep(0.001)

        # rebuild mmap
        if os.name == "nt":
            self._mmap: mmap = mmap(-1, self._buffer_size, self._tag_name)  # , access=ACCESS_WRITE
        elif os.name == "posix":
            self._fd = os.open("/tmp/%s" % self._tag_name, os.O_RDONLY)
            #os.truncate("/tmp/%s" % self._tag_name, self._buffer_size)  # resize file
            self._mmap: mmap = mmap(self._fd, self._buffer_size, MAP_SHARED, PROT_READ)
        else:
            raise OSError("%s is not supported." % os.name)

        # read the whole buffer at once
        self._mmap.seek(0)
        _bytes = self._mmap.read()

        # convert bytes to array
        _, _validity, _numpy_array = self._bytes_to_array(_bytes)

        return _validity, _numpy_array


if __name__ == "__main__":

    _mmap = ndsharray("test_array")

