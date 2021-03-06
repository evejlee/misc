# This file was automatically generated by SWIG (http://www.swig.org).
# Version 1.3.40
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.
# This file is compatible with both classic and new-style classes.

from sys import version_info
if version_info >= (2,6,0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_stomp', [dirname(__file__)])
        except ImportError:
            import _stomp
            return _stomp
        if fp is not None:
            try:
                _mod = imp.load_module('_stomp', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _stomp = swig_import_helper()
    del swig_import_helper
else:
    import _stomp
del version_info
try:
    _swig_property = property
except NameError:
    pass # Python < 2.2 doesn't have 'property'.
def _swig_setattr_nondynamic(self,class_type,name,value,static=1):
    if (name == "thisown"): return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name,None)
    if method: return method(self,value)
    if (not static) or hasattr(self,name):
        self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)

def _swig_setattr(self,class_type,name,value):
    return _swig_setattr_nondynamic(self,class_type,name,value,0)

def _swig_getattr(self,class_type,name):
    if (name == "thisown"): return self.this.own()
    method = class_type.__swig_getmethods__.get(name,None)
    if method: return method(self)
    raise AttributeError(name)

def _swig_repr(self):
    try: strthis = "proxy of " + self.this.__repr__()
    except: strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object : pass
    _newclass = 0


INSIDE_MAP = _stomp.INSIDE_MAP
FIRST_QUADRANT_OK = _stomp.FIRST_QUADRANT_OK
SECOND_QUADRANT_OK = _stomp.SECOND_QUADRANT_OK
THIRD_QUADRANT_OK = _stomp.THIRD_QUADRANT_OK
FOURTH_QUADRANT_OK = _stomp.FOURTH_QUADRANT_OK
class StompMap(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, StompMap, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, StompMap, name)
    __repr__ = _swig_repr
    def __init__(self, *args, **kwargs): 
        this = _stomp.new_StompMap(*args, **kwargs)
        try: self.this.append(this)
        except: self.this = this
    def set_system(self, *args, **kwargs): return _stomp.StompMap_set_system(self, *args, **kwargs)
    def area(self): return _stomp.StompMap_area(self)
    def system(self): return _stomp.StompMap_system(self)
    def genrand(self, *args, **kwargs): return _stomp.StompMap_genrand(self, *args, **kwargs)
    def contains(self, *args, **kwargs): return _stomp.StompMap_contains(self, *args, **kwargs)
    def TestNumpyVector(self, *args, **kwargs): return _stomp.StompMap_TestNumpyVector(self, *args, **kwargs)
    __swig_destroy__ = _stomp.delete_StompMap
    __del__ = lambda self : None;
StompMap_swigregister = _stomp.StompMap_swigregister
StompMap_swigregister(StompMap)



