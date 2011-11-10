#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "math.h"
#include "source.h"
#include "Vector.h"
#include "log.h"

#ifdef SDSSMASK
#include "sdss-survey.h"
#endif

#ifndef WITH_TRUEZ
struct source* source_new(size_t n_zlens) {
    if (n_zlens == 0) {
        wlog("source_new: n_zlens must be > 0\n");
        exit(EXIT_FAILURE);
    }
#else
struct source* source_new(void) {
#endif

    struct source* src = calloc(1,sizeof(struct source));
    if (src == NULL) {
        wlog("Could not allocate struct source\n");
        exit(EXIT_FAILURE);
    }

#ifndef WITH_TRUEZ

        src->scinv = f64vector_new(n_zlens);

        if (src->scinv == NULL) {
            wlog("Could not allocate %ld scinv for source\n", n_zlens);
            exit(EXIT_FAILURE);
        }

#endif

    return src;
}

// source must already be allocated
int source_read(FILE* stream, struct source* src) {
    size_t i=0;
    double ra_rad,dec_rad;
    int nread=0;
    int nexpect=0;

    nread += fscanf(stream, "%lf %lf %lf %lf %lf", 
            &src->ra, &src->dec, &src->g1, &src->g2, &src->err);

#ifndef WITH_TRUEZ
    nexpect = 5+src->scinv->size;
    for (i=0; i<src->scinv->size; i++) {
        nread += fscanf(stream,"%lf", &src->scinv->data[i]);
    }
#else
    nexpect = 5+1;
    fscanf(stream,"%lf", &src->z);
#endif

    ra_rad = src->ra*D2R;
    dec_rad = src->dec*D2R;
    src->sinra = sin(ra_rad);
    src->cosra = cos(ra_rad);
    src->sindec = sin(dec_rad);
    src->cosdec = cos(dec_rad);

#ifdef SDSSMASK
    // add sin(lam),cos(lam),sin(eta),cos(eta)
    eq2sdss_sincos(src->ra,src->dec,
                   &src->sinlam, &src->coslam,
                   &src->sineta, &src->coseta);
#endif

    return (nread == nexpect);
}


/*
void scat_add_hpixid(struct scat* scat, struct healpix* hpix) {

    scat->minpix=INT64_MAX;
    scat->maxpix=-1;
    struct source* src = &scat->data[0];
    for (size_t i=0; i<scat->size; i++) {
        int64 id = hpix_eq2pix(hpix, src->ra, src->dec);
        src->hpixid = id;

        if (id > scat->maxpix) {
            scat->maxpix=id;
        }
        if (id < scat->minpix) {
            scat->minpix=id;
        }
        src++;
    }
}
*/

/*
#ifdef WITH_TRUEZ
void scat_add_dc(struct scat* scat, struct cosmo* cosmo) {
    struct source* src = &scat->data[0];
    for (size_t i=0; i<scat->size; i++) {
        src->dc = Dc(cosmo, 0.0, src->z);
        src++;
    }
}
#endif
*/

void source_print(struct source* src) {
    wlog("    ra:     %lf  dec: %lf\n", src->ra, src->dec);
    wlog("    g1:     %lf  g2: %lf\n", src->g1, src->g2);
    wlog("    err:    %lf\n", src->err);
    wlog("    hpixid: %ld\n", src->hpixid);

#ifdef SDSSMASK
    wlog("    sinlam: %lf\n", src->sinlam);
    wlog("    coslam: %lf\n", src->coslam);
    wlog("    sineta: %lf\n", src->sineta);
    wlog("    coseta: %lf\n", src->coseta);
#endif

#ifdef WITH_TRUEZ
    wlog("    z:      %lf\n", src->z);
    wlog("    dc:     %lf\n", src->dc);
#else
    size_t nzl = src->zlens->size;
    wlog("    zlens[0]: %lf  szinv[0]: %lf\n", 
           src->zlens->data[0], src->scinv->data[0]);
    wlog("    zlens[%ld]: %lf  szinv[%ld]: %lf\n", 
           nzl-1, src->zlens->data[nzl-1], nzl-1, src->scinv->data[nzl-1]);
#endif
}


// use like this:
//   source = source_delete(source);
struct source* source_delete(struct source* src) {

    if (src != NULL) {

#ifndef WITH_TRUEZ
        if (src->scinv != NULL) {
            src->scinv = f64vector_delete(src->scinv);
        }
#endif

        free(src);
    }
    return NULL;
}