#ifndef _SHEAR_HEADER
#define _SHEAR_HEADER

#include <stdio.h>
#include "defs.h"
#include "config.h"
#include "cosmo.h"
#include "healpix.h"
#include "lens.h"
#include "lensum.h"
#include "source.h"

struct shear {
    struct config*  config;
    struct cosmo*   cosmo;
    struct healpix* hpix;
    struct scat*    scat;
    struct lcat*    lcat;

    // hold pixels
    struct i64stack* pixstack;

    // this holds the info for a given lens
    struct lensum* lensum;

    // output file pointer
    FILE* fptr;
};

struct shear* shear_init(const char* config_filename);
struct shear* shear_delete(struct shear* shear);

void shear_calc_bylens(struct shear* shear);

void shear_proclens(struct shear* shear, size_t index);

#endif

