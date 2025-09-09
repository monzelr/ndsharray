# python packages
import os
if os.name == "posix":
    from mmap import MAP_SHARED, PROT_READ
elif os.name == "nt":
    from mmap import ACCESS_READ, ACCESS_WRITE
import mmap
import uuid
import time
import sys
from typing import Union, Tuple
import struct
import warnings

# external python packages
import numpy as np

"""
:var supported_types: supported numpy dtype
"""
supported_types = [np.int8,
                   np.int16,
                   np.int32,
                   np.int64,
                   np.uint8,
                   np.uint16,
                   np.uint32,
                   np.uint64,
                   np.float16,
                   np.float32,
                   np.float64,
                   np.complex64,
                   np.complex128,
                   bool]

"""
:var n_bytes_for_int: get the number of bytes for a python integer, this is python version and system dependent
"""
n_bytes_for_int = len(sys.maxsize.to_bytes((sys.maxsize.bit_length() + 7) // 8, 'big'))


def int_to_bytes(i: int, *, signed: bool = False) -> bytes:
    """
    converts integer to bytes

    :param i:
    :param signed:
    :return:
    """
    global n_bytes_for_int
    return i.to_bytes(n_bytes_for_int, byteorder='big', signed=signed)


def bytes_to_int(b: bytes, *, signed: bool = False) -> int:
    """
    converts bytes to an integer

    :param b:
    :param signed:
    :return:
    """
    return int.from_bytes(b, byteorder='big', signed=signed)


def str_to_bytes(s: str) -> bytes:
    """
    converts string to bytes

    :param s:
    :return:
    """
    return bytearray(s.encode('utf-8'))


def bytes_to_str(b: bytes) -> str:
    """
    converts bytes to a string

    :param b:
    :return:
    """
    return b.decode('utf-8')


class NdShArray(object):
    """
    sharing numpy array between different processes

    """

    def __init__(self, name: str, array: np.ndarray = np.ndarray((0, ), dtype=np.uint8),
                 r_w: Union[str, None] = None):
        """
        :param name:
        :param array:
        :param r_w: 'r' or 'w' for 'read' or 'write' functionality, must be specified
        """
        object.__init__(self)

        # save numpy array and its properties, this array is used for reading and writing
        self._array: np.ndarray = array

        # save the name for the mmap
        self._name: str = name

        # save the read / write property
        if r_w == "w" or r_w == "r" or r_w == "W" or r_w == "R":
            self._access: str = r_w.lower()
        else:
            raise ValueError("'r' or 'w' must be specified for input argument 'r_w'.")

        # initialize last read time
        self._last_write_time: float = 0.0  # set to zero to get in every case the next numpy array
        self._read_time_ms: float = 0.0

        # write time will be saved to make sure it is increasing and unique
        self._write_time: float = 0.0

        # unique identifier for the ndarray mmap name
        self._uuid: str = uuid.uuid4().hex

        # holds just the tag-name of the mmap of the ndarray
        self._mmap: mmap.mmap
        self._fd: Union[None, int] = None

        # holds the numpy ndarray
        self._ndarray_mmap: Union[None, mmap.mmap] = None
        self._ndarray_fd: Union[None, int] = None

        self._is_valid = False

        # buffer size of the _mmap_ndarray
        _bytes = self._array_to_bytes(self._array)
        self._buffer_size: int = len(_bytes)

        # create the mmap which holds the name of the ndarray mmap
        self._mmap, self._fd = self._create_mmap(self._name, len(self.ndarray_mmap_name), r_w=self._access)

        # create ndarray mmap
        self._is_valid = self._create_ndarray_mmap()

        if self._access == "w":
            self.write(array)  # call write to force saving the array via mmap.flush!

    def __del__(self):

        # closing the mmap
        if hasattr(self, "_mmap"):
            self._close_mmap(self._mmap, self._fd)

        # closing the ndarray mmap
        if hasattr(self, "_ndarray_mmap"):
            self._close_mmap(self._ndarray_mmap, self._ndarray_fd)

    @property
    def name(self) -> str:
        """
        unique name of the mmap memory, serves as identifier for other processes
        the name must be declared at class instantiation and is read only after instantiation

        :return name:
        """
        return self._name

    @property
    def is_valid(self) -> bool:
        """
        checks if the header of the numpy array is valid or not

        :return:
        """
        return self._is_valid

    @property
    def ndarray_mmap_name(self) -> str:
        """
        returns the name of the mmap which holds the current ndarray

        ndarray_mmap_name consists of the name and an uuid which is generated for each new ndarray size (changes in
        dtype, shape or dimension does a change in size)

        :return ndarray_mmap_name:
        """
        return "%s_%s" % (self._name, self._uuid)

    @property
    def access(self) -> str:
        """
        access of the ndsharray; either 'w' for only writeable or 'r' for only readable

        :return access:
        """
        return self._access

    @property
    def read_time_ms(self) -> float:
        """
        returns the write-read time of the two processes in milliseconds

        :return:
        """
        return self._read_time_ms

    def _array_to_bytes(self, array: np.ndarray) -> bytes:
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

        note: size of integer may differ because the maximum integer size sys.maxsize will be used (on python3, amd64
        it is 8 byte)

        :param array: byte-encoded numpy array using an own protocol
        :return:
        """
        global supported_types

        if not isinstance(array, np.ndarray):
            raise TypeError("array must be from type np.ndarray.")

        if array.dtype not in supported_types:
            raise NotImplementedError("%s is a numpy.dtype which is not supported. "
                                      "The following numpy.dtypes are supported: %s"
                                      % (str(array.dtype), str([_t.__name__ for _t in supported_types])))

        _now = time.monotonic()
        if _now <= self._write_time:
            _now = float(np.nextafter(self._write_time, float('inf')))  # +1 ULP
        self._write_time = _now
        _time = struct.pack("d", self._write_time)

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
                          previous writing)
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
        _np_dtype = supported_types[bytes_to_int(_bytes[idx:idx+n_bytes_for_int])]
        idx += n_bytes_for_int
        if _np_dtype != self._array.dtype:
            return False, False, _array
        _np_dim = bytes_to_int(_bytes[idx:idx+n_bytes_for_int])
        idx += n_bytes_for_int
        if _np_dim != self._array.ndim:
            return False, False, _array
        _np_shape = []
        for s in range(_np_dim):
            _np_shape.append(bytes_to_int(_bytes[idx:idx + n_bytes_for_int]))
            idx += n_bytes_for_int
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
        write a numpy array into the mmap file, it might be from any type, shape or dimension

        Important Note:
            a mmap will be silently re-created if type, dimension or shape will be changed. the other process will read
            the first line of the mmap and will also re-create its mmap. Re-creating the mmap needs more time than a
            normal read.


        :param array: a numpy.ndarray which shall be saved in mmap
        :return None:
        """
        _bytes = self._array_to_bytes(array)

        # check, if a new mmap has to be generated
        if self._array.dtype != array.dtype or self._array.ndim != array.ndim or self._array.shape != array.shape:
            self._array = array
            self._buffer_size = len(_bytes)
            self._create_ndarray_mmap()

        self._ndarray_mmap.seek(0)
        self._ndarray_mmap.write(_bytes)
        self._ndarray_mmap.flush()

        # write name of ndarray mmap into mmap
        self._mmap.seek(0)
        self._mmap.write(str_to_bytes(self.ndarray_mmap_name))
        self._mmap.flush()

    def read(self) -> Tuple[bool, np.ndarray]:
        """
        reading the shared memory with mmap and numpy's frombuffer, which returns a view of the buffer and not a copy.

        Citing the documentation from numpy.frombuffer:
        "This function creates a view into the original object. This should be safe in general, but it may make sense
        to copy the result when the original object is mutable or untrusted."
        Source: https://numpy.org/doc/stable/reference/generated/numpy.frombuffer.html

        :return validity: boolean displaying if the numpy array is ok or if it is either old or corrupt or not (e.g.
                          mixed numpy ndarray from previous writing). Note: validity is checked by checking if
                          buffer[0] and buffer[-1] have the same time stamp!
        :return array: numpy.ndarray
        """
        global n_bytes_for_int, supported_types

        _recreated_map = False
        _mmap_correct = True
        _validity = False
        _numpy_array = self._array

        # get the ndarray mmap name
        self._mmap.seek(0)
        _ndarray_mmap_name = bytes_to_str(self._mmap.read(len(self._name)+33))
        if _ndarray_mmap_name != self.ndarray_mmap_name:
            self._create_ndarray_mmap()
            _recreated_map = True

        if self._is_valid:
            # first stage of checking if new data have been arrived
            self._ndarray_mmap.seek(0)
            _bytes = self._ndarray_mmap.read(8)
            try:
                _write_time = struct.unpack("d", _bytes)[0]
            except ValueError:
                _write_time = 0
            if _write_time <= self._last_write_time and not _recreated_map:
                return False, _numpy_array

            # without checking, read the whole buffer
            _bytes += self._ndarray_mmap.read()

            if len(_bytes) != self._buffer_size:
                self._create_ndarray_mmap()

            _mmap_correct, _validity, _numpy_array = self._bytes_to_array(_bytes)
            if not _mmap_correct:
                warnings.warn("The mmap of the ndarray seems to be corrupt and the used protocol does not fit.",
                              BytesWarning)

            # for efficiency
            self._array = _numpy_array
            # for debug purpose
            self._read_time_ms = (time.monotonic()-_write_time) * 1000.0
            self._last_write_time = _write_time

        return _validity, _numpy_array

    def _create_ndarray_mmap(self) -> bool:
        """
        creates two mmap:
            - the mmap with tag 'name' just holds the mmap-tag-name of ndarray
            - the mmap of the ndarray may change its name every time a new shape, dimension or dtype is detected

        :return:
        """
        global n_bytes_for_int

        self._close_mmap(self._ndarray_mmap, self._ndarray_fd)

        # now rebuild the mmap
        if self._access == "w":
            # create new uuid
            self._uuid = uuid.uuid4().hex

            self._ndarray_mmap, self._ndarray_fd = self._create_mmap(self.ndarray_mmap_name, self._buffer_size,
                                                                     r_w=self._access)

        elif self._access == "r":
            self._mmap.seek(0)
            _ndarray_mmap_name = bytes_to_str(self._mmap.read(len(self._name)+33))
            self._uuid = _ndarray_mmap_name[-32:]
            try:
                int(self._uuid, 16)
                self._is_valid = True
            except ValueError:
                self._is_valid = False

            try:
                if self._is_valid:
                    # create temporary mmap to get the dtype and dimension of the array
                    _tmp_mmap, _tmp_fd = self._create_mmap(self.ndarray_mmap_name, 8+2*n_bytes_for_int, r_w="r")
                    _tmp_mmap.seek(0)
                    _bytes = _tmp_mmap.read(8+2*n_bytes_for_int)  # skip the time: +8
                    idx = 8
                    _np_dtype = supported_types[bytes_to_int(_bytes[idx:idx+n_bytes_for_int])]
                    idx += n_bytes_for_int
                    _np_dim = bytes_to_int(_bytes[idx:idx+n_bytes_for_int])
                    self._close_mmap(_tmp_mmap, _tmp_fd)

                    # create temporary mmap to get the shape of the array
                    _tmp_2_mmap, _tmp_2_fd = self._create_mmap(self.ndarray_mmap_name,
                                                               8 + 2 * n_bytes_for_int + _np_dim * n_bytes_for_int,
                                                               r_w="r")
                    _tmp_2_mmap.seek(8+2*n_bytes_for_int)  # skip the time, dtype and dimension
                    # read shape
                    _bytes += _tmp_2_mmap.read(_np_dim * n_bytes_for_int)
                    idx = 8 + 2 * n_bytes_for_int
                    _np_shape = []
                    for s in range(_np_dim):
                        _np_shape.append(bytes_to_int(_bytes[idx:idx + n_bytes_for_int]))
                        idx += n_bytes_for_int
                    _np_shape = tuple(_np_shape)
                    self._close_mmap(_tmp_2_mmap, _tmp_2_fd)

                    # rebuild _array and get the length of the byte array -> super lazy and inefficient...
                    self._array = np.ndarray(_np_shape, dtype=_np_dtype)
                    self._buffer_size = len(self._array_to_bytes(self._array))

                    self._ndarray_mmap, self._ndarray_fd = self._create_mmap(self.ndarray_mmap_name, self._buffer_size,
                                                                             r_w=self._access)
            except:
                self._is_valid = False
        return self._is_valid

    @staticmethod
    def _create_mmap(name: str, buffer_size: int, r_w: str) -> Tuple[mmap.mmap, Union[None, int]]:
        """
        static helper function to create a mmap for a specific  operating system

        :param name:
        :param buffer_size:
        :param r_w:
        :return:
        """
        _mmap: Union[None, mmap.mmap] = None
        _fd: Union[None, int] = None

        if os.name == "nt":
            if r_w == "w":
                _mmap = mmap.mmap(-1, buffer_size, name, access=ACCESS_WRITE)
                _mmap.flush()
            elif r_w == "r":
                _mmap = mmap.mmap(-1, buffer_size, name, access=ACCESS_READ)
        elif os.name == "posix":
            if r_w == "w":
                _fd = os.open("/dev/shm/%s" % name, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
                os.truncate("/dev/shm/%s" % name, buffer_size)  # resize file
                _mmap = mmap.mmap(_fd, buffer_size, MAP_SHARED)
                _mmap.flush()
            elif r_w == "r":
                _fd = os.open("/dev/shm/%s" % name, os.O_RDONLY)
                _mmap = mmap.mmap(_fd, buffer_size, MAP_SHARED, PROT_READ)
        else:
            raise OSError("%s is not supported." % os.name)

        return _mmap, _fd

    @staticmethod
    def _close_mmap(_mmap: Union[mmap.mmap, None], _fd: Union[None, int]) -> None:
        """
        static helper function to close a mmap for a specific  operating system

        :param _mmap:
        :return:
        """

        # closing the mmap
        if _mmap is not None:
            # closing the mmap
            _mmap.close()
            while not _mmap.closed:
                time.sleep(0.001)

        # closing the ndarray file
        if os.name == "posix":
            if _fd is not None:
                os.close(_fd)
