#include <stdlib.h>
#include <stdio.h>
#include "lensum.h"
#include "defs.h"
#include "log.h"

struct lensums* lensums_new(size_t nlens, size_t nbin) {
    wlog("Creating lensums:\n");
    wlog("    nlens: %lu  nbin: %lu\n", nlens, nbin);

    struct lensums* lensums=calloc(1,sizeof(struct lensums));
    if (lensums == NULL) {
        wlog("failed to allocate lensums struct\n");
        exit(EXIT_FAILURE);
    }

    lensums->data = calloc(nlens, sizeof(struct lensum));
    if (lensums->data == NULL) {
        wlog("failed to allocate lensum array\n");
        exit(EXIT_FAILURE);
    }

    lensums->size = nlens;

    struct lensum* lensum = &lensums->data[0];

    for (size_t i=0; i<nlens; i++) {
        lensum->nbin = nbin;
        lensum->npair = calloc(nbin, sizeof(int64));
        lensum->wsum  = calloc(nbin, sizeof(double));
        lensum->dsum  = calloc(nbin, sizeof(double));
        lensum->osum  = calloc(nbin, sizeof(double));
        lensum->rsum  = calloc(nbin, sizeof(double));

        if (lensum->npair==NULL
                || lensum->wsum==NULL
                || lensum->dsum==NULL
                || lensum->osum==NULL
                || lensum->rsum==NULL) {

            wlog("failed to allocate lensum\n");
            exit(EXIT_FAILURE);
        }

        lensum++;
    }
    return lensums;

}

// this one we write all the data out in .rec binary format
/* 
SIZE =               206951
{'_DTYPE': [('zindex', '<i8'),
            ('weight', '<f8'),
            ('npair', '<i8', 21),
            ('rsum', '<f8', 21),
            ('wsum', '<f8', 21),
            ('dsum', '<f8', 21),
            ('osum', '<f8', 21)],
 '_VERSION': '1.0'}
END
*/


// this is for writing them all at once.  We actually usually
// write them one at a time
void lensums_write(struct lensums* lensums, FILE* stream) {

    struct lensum* lensum = &lensums->data[0];
    for (size_t i=0; i<lensums->size; i++) {
        lensum_write(lensum, stream);
        
        lensum++;
    }
}


struct lensum* lensums_sum(struct lensums* lensums) {
    struct lensum* tsum=lensum_new(lensums->data[0].nbin);

    struct lensum* lensum = &lensums->data[0];

    for (size_t i=0; i<lensums->size; i++) {
        tsum->weight   += lensum->weight;
        tsum->totpairs += lensum->totpairs;
        tsum->sshsum   += lensum->sshsum;
        for (size_t j=0; j<lensum->nbin; j++) {
            tsum->npair[j] += lensum->npair[j];
            tsum->rsum[j] += lensum->rsum[j];
            tsum->wsum[j] += lensum->wsum[j];
            tsum->dsum[j] += lensum->dsum[j];
            tsum->osum[j] += lensum->osum[j];
        }
        lensum++;
    }
    return tsum;
}



void lensums_print_sum(struct lensums* lensums) {
    struct lensum* lensum = lensums_sum(lensums);
    lensum_print(lensum);
    lensum=lensum_delete(lensum);
}

// these write the stdout
void lensums_print_one(struct lensums* lensums, size_t index) {
    wlog("element %ld of lensums:\n",index);
    struct lensum* lensum = &lensums->data[index];
    lensum_print(lensum);
}

void lensums_print_firstlast(struct lensums* lensums) {
    lensums_print_one(lensums, 0);
    lensums_print_one(lensums, lensums->size-1);
}

struct lensums* lensums_delete(struct lensums* lensums) {
    if (lensums != NULL) {
        struct lensum* lensum = &lensums->data[0];

        for (size_t i=0; i<lensums->size; i++) {
            free(lensum->npair);
            free(lensum->wsum);
            free(lensum->dsum);
            free(lensum->osum);
            free(lensum->rsum);
            lensum++;
        }
    }
    free(lensums);
    return NULL;
}

struct lensum* lensum_new(size_t nbin) {
    struct lensum* lensum=calloc(1,sizeof(struct lensum));
    if (lensum == NULL) {
        wlog("failed to allocate lensum\n");
        exit(EXIT_FAILURE);
    }

    lensum->nbin = nbin;

    lensum->npair = calloc(nbin, sizeof(int64));
    lensum->wsum  = calloc(nbin, sizeof(double));
    lensum->dsum  = calloc(nbin, sizeof(double));
    lensum->osum  = calloc(nbin, sizeof(double));
    lensum->rsum  = calloc(nbin, sizeof(double));

    if (lensum->npair==NULL
            || lensum->wsum==NULL
            || lensum->dsum==NULL
            || lensum->osum==NULL
            || lensum->rsum==NULL) {

        wlog("failed to allocate lensum\n");
        exit(EXIT_FAILURE);
    }

    return lensum;
}


// add the second lensum into the first
void lensum_add(struct lensum* dest, struct lensum* src) {

    dest->weight   += src->weight;
    dest->totpairs += src->totpairs;
    dest->sshsum   += src->sshsum;
    for (size_t i=0; i<src->nbin; i++) {
        dest->npair[i] += src->npair[i];
        dest->rsum[i] += src->rsum[i];
        dest->wsum[i] += src->wsum[i];
        dest->dsum[i] += src->dsum[i];
        dest->osum[i] += src->osum[i];
    }

}

void lensum_write(struct lensum* lensum, FILE* stream) {
    int nbin = lensum->nbin;
    int i=0;

    fprintf(stream,"%ld %.15e %ld %.15e ", 
            lensum->zindex, lensum->weight, lensum->totpairs, lensum->sshsum);

    for (i=0; i<nbin; i++) 
        fprintf(stream,"%ld ", lensum->npair[i]);

    for (i=0; i<nbin; i++) 
        fprintf(stream,"%.15e ", lensum->rsum[i]);
    for (i=0; i<nbin; i++) 
        fprintf(stream,"%.15e ", lensum->wsum[i]);
    for (i=0; i<nbin; i++) 
        fprintf(stream,"%.15e ", lensum->dsum[i]);
    for (i=0; i<nbin-1; i++) 
        fprintf(stream,"%.15e ", lensum->osum[i]);
    fprintf(stream,"%.15e\n", lensum->osum[nbin-1]);

}

// these write the stdout
void lensum_print(struct lensum* lensum) {
    wlog("  zindex:   %ld\n", lensum->zindex);
    wlog("  weight:   %lf\n", lensum->weight);
    wlog("  sshsum:   %lf\n", lensum->sshsum);
    wlog("  ssh:      %lf\n", lensum->sshsum/lensum->weight);
    wlog("  totpairs: %ld\n", lensum->totpairs);
    wlog("  nbin:     %ld\n", lensum->nbin);
    wlog("  bin       npair            wsum            dsum            osum           rsum\n");

    for (size_t i=0; i<lensum->nbin; i++) {
        wlog("  %3lu %11ld %15.6lf %15.6lf %15.6lf   %e\n", 
             i,
             lensum->npair[i],
             lensum->wsum[i],
             lensum->dsum[i],
             lensum->osum[i],
             lensum->rsum[i]);
    }
}



void lensum_clear(struct lensum* lensum) {

    lensum->zindex=-1;
    lensum->weight=0;
    lensum->totpairs=0;
    lensum->sshsum=0;
    for (size_t i=0; i<lensum->nbin; i++) {
        lensum->npair[i] = 0;
        lensum->wsum[i] = 0;
        lensum->dsum[i] = 0;
        lensum->osum[i] = 0;
        lensum->rsum[i] = 0;
    }
}

struct lensum* lensum_delete(struct lensum* lensum) {
    if (lensum != NULL) {
        free(lensum->npair);
        free(lensum->rsum);
        free(lensum->wsum);
        free(lensum->dsum);
        free(lensum->osum);
    }
    free(lensum);
    return NULL;
}
