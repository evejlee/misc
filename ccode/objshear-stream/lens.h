#ifndef _LENS_HEADER
#define _LENS_HEADER

#include "cosmo.h"
#include "defs.h"

struct lens {

    double ra;
    double dec;
    double z;
    int64 zindex;

    double da;

    double sinra;
    double cosra;
    double sindec;
    double cosdec;

    int64 maskflags;

    // We will fill these with the healpix pixels within
    // the search radius for each lens
    struct i64stack* hpix;
    struct szvector* rev;

    double cos_search_angle;

#ifdef SDSSMASK
    double sinlam;
    double coslam;
    double sineta;
    double coseta;
#endif

};


struct lcat {
    size_t size;
    struct lens* data;
};

struct lcat* lcat_new(size_t n_lens);
struct lcat* lcat_read(const char* filename);
void lcat_add_da(struct lcat* lcat, struct cosmo* cosmo);
void lcat_add_search_angle(struct lcat* lcat, double rmax);

void lcat_print_one(struct lcat* lcat, size_t el);
void lcat_print_firstlast(struct lcat* lcat);

struct lcat* lcat_delete(struct lcat* lcat);


void lcat_disc_intersect(struct lcat* lcat, int64 nside, double rmax);

#endif
