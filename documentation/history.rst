=======
History
=======

Version 1.0.0
-------------
- initial release for PyPi

Version 1.0.1
-------------
- added only documentation

Version 1.1.0
-------------
- new property 'is_valid' to check if the mmap of the numpy array is valid

Version 1.1.1
-------------
- bug fixes for __init__ and __del__ functions

Version 1.1.2
-------------
- added support for numpy 2.x.x - no change in API

Version 1.1.4
-------------
- added the 'unit in the last place' (ULP) to the time.monotonic() in the ndsharray.write() function so multiple writes in a milli seconds are possible
- small fix for self.is_valid which has been written even it is not writeable
- no change in API