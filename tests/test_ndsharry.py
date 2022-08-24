#!/usr/bin/env python

"""
Tests for `NdShArray` package.

The test covers the following tests:
- 4 different ndarray shapes
- 3 different ndarray dimensions
- all types in NdShArray.supported_types
- one NdShArray instance with different shapes/dimension/types, thus the mmap will be automatically re-created
"""

# internal python packages
import multiprocessing
from typing import Union
import logging
import logging.handlers
import unittest
import atexit
import time
import os

# external python packages
import numpy as np

# self written python pacakges
from ndsharray import NdShArray, supported_types

logger = logging.getLogger(__name__)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter and add it to the handlers
ch_formatter = logging.Formatter('%(asctime)s.%(msecs)03d [%(name)s %(levelname)s]: %(message)s',
                                 "%Y-%m-%d %H:%M:%S")
ch.setFormatter(ch_formatter)

logging.basicConfig(level=logging.DEBUG,
                    handlers=[ch])

_test_sizes = [(72, 128),
               (4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4),  # increase the dimension
               (24, 48, 3, 6),  # decrease the dimensionW
               (2400, 1800)]


class Testndsharray(unittest.TestCase):
    """Tests for `NdShArray` package."""

    # queues
    _queue_logger: multiprocessing.Queue = multiprocessing.Queue()

    # events
    _write_event: multiprocessing.Event = multiprocessing.Event()
    _read_write_event: multiprocessing.Event = multiprocessing.Event()
    _end_test_event: multiprocessing.Event = multiprocessing.Event()
    _is_initialized_event: multiprocessing.Event = multiprocessing.Event()

    _logger_queue_listener: Union[None, logging.handlers.QueueListener] = None
    _test_process: Union[None, multiprocessing.Process] = None

    # two NdShArray objects, one which will read and one which will write
    _ndsharray_read: Union[None, NdShArray] = None
    _ndsharray_write: Union[None, NdShArray] = None

    _name: str = "testing_ndsharray"

    @classmethod
    def setUpClass(cls):
        """
        setting up everything the second process
        :return:
        """
        global logger
        logger.info("Setting up the unittest.")

        cls._logger_queue_listener = logging.handlers.QueueListener(cls._queue_logger, *logger.handlers,
                                                                    respect_handler_level=True)
        cls._logger_queue_listener.start()

        # set up our write array
        cls._ndsharray_write = NdShArray(cls._name, r_w="w")

        cls._is_initialized_event.clear()
        cls._write_event.clear()
        cls._read_write_event.clear()
        cls._end_test_event.clear()
        cls._test_process = multiprocessing.Process(target=_testing_process,
                                                    args=(cls._name,
                                                          cls._is_initialized_event,
                                                          cls._write_event,
                                                          cls._read_write_event,
                                                          cls._end_test_event,
                                                          cls._queue_logger
                                                          ))
        cls._test_process.start()

        # wait for the initialization
        cls._is_initialized_event.wait(timeout=5)  # 5 seconds timeout

        # set up our read array
        cls._ndsharray_read = NdShArray("%s_copy" % cls._name, r_w="r")

    def setUp(self):
        """Set up test fixtures, if any."""

    @classmethod
    def tearDownClass(cls):
        """Tear down test fixtures, if any."""
        cls._end_test_event.set()

    def test_all(self):
        """

        :return:
        """
        global _test_sizes

        # our test arrays
        _write_array: np.ndarray
        _read_array: np.ndarray

        self._write_event.clear()
        self._read_write_event.clear()

        try:

            _break = False
            for _size in _test_sizes:
                for _type in supported_types:
                    logger.info("Testing size: %s, type: %s" % (str(_size), _type.__name__))
                    
                    _write_array = (np.random.random(_size) * 255).astype(_type)

                    _, _ = self._ndsharray_read.read()  # must be a first read before write

                    self._ndsharray_write.write(_write_array)
                    time.sleep(0.001)

                    self._write_event.set()

                    self._read_write_event.wait(timeout=5)
                    self._read_write_event.clear()

                    status = False
                    _read_array = np.ndarray((1, ), dtype=_type)
                    _timeout = 5
                    _time_start = time.time()
                    while not status and (time.time() - _time_start) <= _timeout:
                        status, _read_array = self._ndsharray_read.read()
                        time.sleep(0.001)
                    
                    if status:
                        _result = np.array_equal(_write_array, _read_array)
                        if not _result:
                            logger.error(("write_array: %s" % self._ndsharray_write.name).ljust(40) +
                                         ("read_array %s" % self._ndsharray_read.name).ljust(40))
                            logger.error(("\t shape: %s" % str(_write_array.shape)).ljust(40) +
                                         ("\t shape: %s" % str(_read_array.shape)).ljust(40))
                            logger.error(("\t type: %s" % str(_write_array.dtype)).ljust(40) +
                                         ("\t type: %s" % str(_read_array.dtype)).ljust(40))
                            logger.error(("\t dimension: %i" % _write_array.ndim).ljust(40) +
                                         ("\t dimension: %i" % _read_array.ndim).ljust(40))
                        self.assertTrue(_result)
                    else:
                        logger.error("The read NdShArray is invalid!")
                        if isinstance(_read_array, np.ndarray):
                            logger.error(("write_array: %s" % self._ndsharray_write.name).ljust(40) +
                                         ("read_array %s" % self._ndsharray_read.name).ljust(40))
                            logger.error(("\t shape: %s" % str(_write_array.shape)).ljust(40) +
                                         ("\t shape: %s" % str(_read_array.shape)).ljust(40))
                            logger.error(("\t type: %s" % str(_write_array.dtype)).ljust(40) +
                                         ("\t type: %s" % str(_read_array.dtype)).ljust(40))
                            logger.error(("\t dimension: %i" % _write_array.ndim).ljust(40) +
                                         ("\t dimension: %i" % _read_array.ndim).ljust(40))
                        raise Exception("The read NdShArray is invalid!")

                if _break:
                    break

            if _break:
                logger.info("Timeout occurred.")
        except Exception as e:
            logger.exception("Error occurred:")
            raise e

    @staticmethod
    def disconnect():
        """

        :return:
        """


def _testing_process(_name: str,
                     _is_initialized_event: multiprocessing.Event,
                     _write_event: multiprocessing.Event,
                     _read_write_event: multiprocessing.Event,
                     _end_test_event: multiprocessing.Event,
                     _queue_logger: multiprocessing.Queue) -> None:
    """

    :param _name:
    :param _is_initialized_event:
    :param _write_event:
    :param _read_write_event:
    :param _end_test_event:
    :param _queue_logger:
    :return:
    """
    logger.info("Initializing process '_testing_process with PID %i..." % os.getpid())

    # set up logger in _testing_process
    _logger = logging.getLogger(__name__)
    qh = logging.handlers.QueueHandler(_queue_logger)
    qh.setLevel(logging.DEBUG)
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(qh)

    # set up NdShArray
    _nds_read = NdShArray(_name, r_w="r")
    _nds_write = NdShArray("%s_copy" % _name, r_w="w")

    # signaling initialized
    _is_initialized_event.set()
    logger.info("... initialized.")

    try:
        while not _end_test_event.is_set():
            time.sleep(0.001)

            _write_event.wait(timeout=5)
            _write_event.clear()

            _status = False
            _result = np.ndarray((1, 1))
            _timeout = 5
            _time_start = time.time()
            while not _status and (time.time() - _time_start) <= _timeout:
                _status, _result = _nds_read.read()
                time.sleep(0.001)

            _read_write_event.set()

            _nds_write.write(_result)
            time.sleep(0.001)

    except:
        _logger.exception("Error occurred:\n")

    finally:
        pass  # NdShArray will close and clean itself by the python garbage collector - no need to anything

        logger.info("Process '_testing_process' with PID %i ended." % os.getpid())


if __name__ == "__main__":
    # do the test

    multiprocessing.freeze_support()

    unittest.main()
