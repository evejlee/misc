# we over-ride the init to deal with verbose
# we over-ride point checking codes force inputs to be arrays
#
# note we do *not* over-ride the genrand* functions,
# as they perform conversions as needed
#
# we also grab the doc strings from the C code as needed, 
# only writing new ones in the over-ridden methods

import numpy
from numpy import array
from . import _mangle
__doc__=_mangle.__doc__

class Mangle(_mangle.Mangle):
    __doc__=_mangle.Mangle.__doc__
    def __init__(self, filename, verbose=False):
        if verbose:
            verb=1
        else:
            verb=0
        super(Mangle,self).__init__(filename,verb)

    def polyid_and_weight(self, ra, dec):
        """
        Check points against mask, returning (poly_id,weight).

        parameters
        ----------
        ra:  array
            Right ascension in degrees.  Can be an array.
        dec: array
            Declination in degrees.  Can be an array.
        """
        ra = array(ra, ndmin=1, dtype='f8', copy=False, order='C')
        dec = array(dec, ndmin=1, dtype='f8', copy=False, order='C')
        return super(Mangle,self).polyid_and_weight(ra,dec)

    def polyid(self, ra, dec):
        """
        Check points against mask, returning the polygon id or -1.

        parameters
        ----------
        ra:  array
            Right ascension in degrees.  Can be an array.
        dec: array
            Declination in degrees.  Can be an array.
        """
        ra = array(ra, ndmin=1, dtype='f8', copy=False, order='C')
        dec = array(dec, ndmin=1, dtype='f8', copy=False, order='C')
        return super(Mangle,self).polyid(ra,dec)

    def weight(self, ra, dec):
        """
        Check points against mask, returning the weight or 0.

        parameters
        ----------
        ra:  array
            Right ascension in degrees.  Can be an array.
        dec: array
            Declination in degrees.  Can be an array.
        """
        ra = array(ra, ndmin=1, dtype='f8', copy=False, order='C')
        dec = array(dec, ndmin=1, dtype='f8', copy=False, order='C')
        return super(Mangle,self).weight(ra,dec)

    def contains(self, ra, dec):
        """
        Check points against mask, returning 1 if contained 0 if not

        parameters
        ----------
        ra:  array
            Right ascension in degrees.  Can be an array.
        dec: array
            Declination in degrees.  Can be an array.
        """
        # we specify order to force contiguous
        ra = array(ra, ndmin=1, dtype='f8', copy=False, order='C')
        dec = array(dec, ndmin=1, dtype='f8', copy=False, order='C')
        return super(Mangle,self).contains(ra,dec)


    filename = property(_mangle.Mangle.get_filename,doc="The mask filename")
    area = property(_mangle.Mangle.get_area,doc="The area of the mask")
    npoly = property(_mangle.Mangle.get_npoly,doc="The number of polygons in the mask")
    is_pixelized = property(_mangle.Mangle.get_is_pixelized,doc="True if pixelized.")
    pixeltype = property(_mangle.Mangle.get_pixeltype,doc="The pixelization type, 'u' of unpixelized")
    pixelres = property(_mangle.Mangle.get_pixelres,doc="The pixel resolution, -1 if unpixelized")
    maxpix = property(_mangle.Mangle.get_pixelres,doc="The maximum pixel value")
    is_snapped = property(_mangle.Mangle.get_is_snapped,doc="True if snapped.")
    is_balkanized = property(_mangle.Mangle.get_is_balkanized,doc="True if balkanized.")
