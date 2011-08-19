# will want a different one for tutti
f=~astrodat/setup/setup-modules.sh
if [[ -e $f ]]; then

    source "$f"

    # those marked with * have platform dependent code, e.g. the are
    # stand-alone C or extensions for python, etc.

    # this are under $MODULESHOME/modulefiles
    # and installed under $MODULE_INSTALLS
    module load use.own

    module load parallel

    module load pyyaml
    # also loads pcre      # *
    module load swig       # *

    module load ccfits     # *


    # these are under my ~/privatemodules
    # and installed generally under ~/exports

    module load perllib
    module load shell_scripts
    module load espy

    module load desfiles

    module load local      # *

    # this is currently just the python extension
    module load stomp      # *

    module load esutil     # *
    module load recfile    # *

    module load cosmology  # *
    module load admom      # *
    module load fimage     # *

    module load numpydb    # *

    module load scikits_learn # *

    # python only
    module load sdsspy
    module load columns



    module load tmv        # *

    # prereq: ccfits, tmv, desfiles, esutil
    # also numpy if not using system
    module load wl/local   # *

fi

