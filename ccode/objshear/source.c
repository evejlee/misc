#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "math.h"
#include "source.h"
#include "Vector.h"

#ifndef WITH_TRUEZ
struct scat* scat_new(size_t n_source, size_t n_zlens) {
    if (n_zlens == 0) {
        printf("scat_new: n_zlens must be > 0\n");
        exit(EXIT_FAILURE);
    }
#else
struct scat* scat_new(size_t n_source) {
#endif

    if (n_source == 0) {
        printf("scat_new: size must be > 0\n");
        exit(EXIT_FAILURE);
    }

    struct scat* scat = malloc(sizeof(struct scat));
    if (scat == NULL) {
        printf("Could not allocate struct scat\n");
        exit(EXIT_FAILURE);
    }

    scat->size = n_source;

    scat->data = malloc(n_source*sizeof(struct source));
    if (scat->data == NULL) {
        printf("Could not allocate %ld sources in scat\n", n_source);
        exit(EXIT_FAILURE);
    }

#ifndef WITH_TRUEZ

    scat->zlens = f64vector_new(n_zlens);
    if (scat->zlens== NULL) {
        printf("Could not allocate %ld zlens in scat\n", n_zlens);
        exit(EXIT_FAILURE);
    }

    struct source* src = &scat->data[0];
    for (size_t i=0; i<scat->size; i++) {
        src->scinv = f64vector_new(n_zlens);

        if (src->scinv == NULL) {
            printf("Could not allocate %ld scinv in scat[%ld]\n", n_zlens, i);
            exit(EXIT_FAILURE);
        }

        // for convenience point to this zlens in each source structure
        // DONT FREE!
        src->zlens = scat->zlens;
        src++;
    }
#endif

    return scat;
}

struct scat* scat_read(const char* filename) {
    printf("Reading sources from %s\n", filename);
    FILE* fptr=fopen(filename,"r");
    if (fptr==NULL) {
        printf("Could not open file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    int rval;

    int64 sigmacrit_style;
    rval=fread(&sigmacrit_style, sizeof(int64), 1, fptr);

#ifndef WITH_TRUEZ
    if (sigmacrit_style != 2) {
        printf("Got sigmacrit_style = %ld but code is compiled for p(z), sigmacrit_style=2.\n", sigmacrit_style);
        exit(EXIT_FAILURE);
    }

    int64 n_zlens;
    rval=fread(&n_zlens, sizeof(int64), 1, fptr);
    printf("  Reading %ld zlens values: ", n_zlens);

    // using a temporary variable since scat is not yet allocated
    struct f64vector* zlens = f64vector_new(n_zlens);
    rval=fread(&zlens->data[0], sizeof(double), n_zlens, fptr);

    for (int64 i=0; i<n_zlens; i++) {
        printf("%lf ", zlens->data[i]);
    }
    printf("\n");

#else
    if (sigmacrit_style != 1) {
        printf("Got sigmacrit_style = %ld but code is compiled for true z, sigmacrit_style=1.\n", sigmacrit_style);
        exit(EXIT_FAILURE);
    }

#endif


    int64 nsource;
    rval=fread(&nsource, sizeof(int64), 1, fptr);
    printf("Reading %ld sources\n", nsource);
    printf("  creating scat...");

#ifndef WITH_TRUEZ
    struct scat* scat=scat_new(nsource, n_zlens);
    // scat->zlens is already allocated
    memcpy(scat->zlens->data, zlens->data, n_zlens*sizeof(double));
    f64vector_delete(zlens);
#else
    struct scat* scat=scat_new(nsource);
#endif

    printf("OK\n");

    printf("  reading data\n");
    struct source* src = &scat->data[0];
    double ra_rad,dec_rad;
    for (size_t i=0; i<scat->size; i++) {
        rval=fread(&src->ra, sizeof(double), 1, fptr);
        rval=fread(&src->dec, sizeof(double), 1, fptr);
        rval=fread(&src->g1, sizeof(double), 1, fptr);
        rval=fread(&src->g2, sizeof(double), 1, fptr);
        rval=fread(&src->err, sizeof(double), 1, fptr);

        // need to remove this from the file since we
        // always calculate it
        rval=fread(&src->hpixid, sizeof(int64), 1, fptr);

#ifndef WITH_TRUEZ

        // read the full inverse critical density for
        // interpolation.  Note these are already allocated
        rval=fread(&src->scinv->data[0], sizeof(double), n_zlens, fptr);

#else
        rval=fread(&src->z, sizeof(double), 1, fptr);

        // remove this from the file since we calculate it
        rval=fread(&src->dc, sizeof(double), 1, fptr);

#endif
        if (i == 0 || i == (scat->size-1)) {
            printf("  %ld: ra: %lf  dec: %lf\n", i, src->ra, src->dec);
        }

        ra_rad = src->ra*D2R;
        dec_rad = src->dec*D2R;
        src->sinra = sin(ra_rad);
        src->cosra = cos(ra_rad);
        src->sindec = sin(dec_rad);
        src->cosdec = cos(dec_rad);

        src++;
    }


    return scat;
}

void scat_print_one(struct scat* scat, size_t el) {
    struct source* src = &scat->data[el];
    printf("element %ld of scat:\n", el);
    printf("  ra: %lf  dec: %lf\n", R2D*asin(src->sinra), R2D*asin(src->sindec));
    printf("  g1: %lf  g2: %lf\n", src->g1, src->g2);
    printf("  err: %lf\n", src->err);
    printf("  hpixid: %ld\n", src->hpixid);
#ifdef WITH_TRUEZ
    printf("  z: %lf\n", src->z);
    printf("  dc: %lf\n", src->dc);
#else
    size_t nzl = src->zlens->size;
    printf("  zlens[0]: %lf  szinv[0]: %lf\n", 
           src->zlens->data[0], src->scinv->data[0]);
    printf("  zlens[%ld]: %lf  szinv[%ld]: %lf\n", 
           nzl-1, src->zlens->data[nzl-1], nzl-1, src->scinv->data[nzl-1]);
#endif
}
void scat_print_firstlast(struct scat* scat) {
    scat_print_one(scat, 0);
    scat_print_one(scat, scat->size-1);
}


// use like this:
//   scat = scat_delete(scat);
// This ensures that the scat pointer is set to NULL
struct scat* scat_delete(struct scat* scat) {

    if (scat != NULL) {

#ifndef WITH_TRUEZ
        f64vector_delete(scat->zlens);

        struct source* src = &scat->data[0];
        for (size_t i=0; i<scat->size; i++) {
            free(src->scinv);
            src++;
        }
#endif

        free(scat->data);
        free(scat);
    }
    return NULL;
}