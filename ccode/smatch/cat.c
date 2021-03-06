#include <stdlib.h>
#include <stdio.h>
#include "cat.h"
#include "point.h"
#include "healpix.h"
#include "point_hash.h"
#include "files.h"
#include "alloc.h"
#include "match.h"
#include "VEC.h"

struct cat* cat_new(size_t n, int64 nside) 
{
    struct cat* cat = alloc_or_die(sizeof(struct cat), "catalog struct");

    cat->pts = alloc_or_die(n*sizeof(struct point),"points");
    cat->size = n;
    cat->hpix = hpix_new(nside);

    return cat;
}
struct cat* read_cat(const char* fname, 
                     int64 nside, 
                     double radius_arcsec, 
                     int verbose) 
{

    FILE* fptr=open_file(fname);

    size_t nlines=countlines(fptr);
    if (verbose) wlog("    found %lu lines\n", nlines);
    rewind(fptr);

    struct cat* cat = _cat_read(fptr, 
                                nlines, 
                                nside,
                                radius_arcsec,
                                verbose);

    fclose(fptr);
    return cat;
}

void cat_match(struct cat* self, 
               double ra, 
               double dec, 
               VEC(Match) matches) // vector of struct match
{

    double x=0,y=0,z=0;

    int64 hpixid = hpix_eq2pix(self->hpix, ra, dec);

    Match match;

    VEC_CLEAR(matches);

    struct point_hash* pthash = point_hash_find(self->pthash, hpixid);
    if (pthash != NULL) {

        hpix_eq2xyz(ra,dec,&x,&y,&z);

        VEC_FOREACH(ptp, pthash->points) {
            double cos_angle = ptp->x*x + ptp->y*y + ptp->z*z;

            if (cos_angle > ptp->cos_radius) {
                match.index = ptp->index;
                match.cos_dist = cos_angle;

                // copies the entire structure
                VEC_PUSH(matches, match);
            }
        }

    }

    return;
}


struct cat* _cat_read(FILE* fptr, 
                      size_t nlines,
                      int64 nside,
                      double radius_arcsec,
                      int verbose)
{
    int radius_in_file=0;
    double ra=0, dec=0;
    double radius_radians=0;
    double cos_radius_global=0;
    VEC(int64) listpix = VEC_NEW(int64);

    size_t index=0;
    int barsize=70;

    struct cat* self = cat_new(nlines, nside);

    if (verbose) {
        wlog("    reading and building hash table\n");
        repeat_char('.', barsize); wlog("\n");
    }

    radius_in_file = _cat_get_radius(radius_arcsec,&radius_radians,
                                     &cos_radius_global);

    struct point_hash* pthash = NULL;
    struct point* pt = &self->pts[0];
    for (size_t i=0; i<self->size; i++) {
        pt->index=index;
        if (2 != fscanf(fptr, "%lf %lf", &ra, &dec)) {
            wlog("expected to read point at line %lu\n", i);
            exit(EXIT_FAILURE);
        }
        if (radius_in_file) {
            if (1 != fscanf(fptr, "%lf", &radius_arcsec)) {
                wlog("expected to read radius at line %lu\n", i);
                exit(EXIT_FAILURE);
            }
            radius_radians = radius_arcsec/3600.*D2R;
            pt->cos_radius = cos(radius_radians);
        } else {
            pt->cos_radius = cos_radius_global;
        }

        hpix_eq2xyz(ra,dec,&pt->x,&pt->y,&pt->z);
        hpix_disc_intersect(self->hpix, pt->x, pt->y, pt->z, radius_radians, 
                            listpix);

        // insert this point for each pixel it intersected
        VEC_FOREACH(pix, listpix) {
            pthash=point_hash_insert(pthash, *pix, pt);
        }

        pt++;
        index++;
        if (verbose) incr_bar(i+1, self->size, barsize, '=');
    }

    self->pthash=pthash;

    VEC_FREE(listpix);

    if (verbose) wlog("\n");

    if (index != self->size) {
        wlog("expected %lu lines but read %lu\n", self->size, index);
        exit(EXIT_FAILURE);
    }

    return self;
}

int _cat_get_radius(double radius_arcsec, 
                    double* radius_radians,
                    double* cos_radius_global)
{
    int radius_in_file=0;
    if (radius_arcsec <= 0) {
        radius_in_file=1;
    } else {
        radius_in_file=0;
        *radius_radians = radius_arcsec/3600.*D2R;
        *cos_radius_global = cos(*radius_radians);
    }
    return radius_in_file;
}
void repeat_char(char c, int n) {
    for (int i=0; i<n; i++) {
        fputc(c,stderr);
    }
}
void incr_bar(size_t this_index, 
              size_t ntot, 
              size_t ntoprint, 
              char c)
{
    if ( this_index % (ntot/ntoprint) != 0 ) 
        return;
    fputc(c,stderr);
}



