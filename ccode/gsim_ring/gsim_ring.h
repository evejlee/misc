#ifndef _GSIM_RING_HEADER_GUARD
#define _GSIM_RING_HEADER_GUARD

#include <stdio.h>
#include "gmix.h"
#include "shape.h"
#include "dist.h"
#include "gsim_ring_config.h"

#define GAUSS_PADDING 5
#define RING_IMAGE_NSUB 16
#define RING_PSF_S2N 1.0e8

// for now models are always simple or bulge+disk, so pars are, e.g.
// 
//    [eta,eta2,Tbulge,Tdisk,Fbulge,Fdisk]
//
// it is OK to have either flux zero.  
//
// The PSF is for now a simple model with only a scale length and shape.  Can
// be GMIX_COELLIP (but only one gauss) or GMIX_TURB.
//
//    [eta1,eta2,T]
//

struct gsim_ring {
    struct gsim_ring_config conf;

    // can generalize this
    struct dist_gauss cen1_dist;
    struct dist_gauss cen2_dist;

    struct dist_gmix3_eta shape_prior;

    struct dist_lognorm T_dist;
    struct dist_lognorm counts_dist;

};

struct ring_pair {
    double s2n;

    double cen1_start;
    double cen2_start;
    double cen1_offset;
    double cen2_offset;

    struct gmix *gmix1;
    struct gmix *gmix2;
    struct gmix *psf_gmix;
};

struct ring_image_pair {
    struct image *im1;
    struct image *wt1;
    struct image *im2;
    struct image *wt2;
    double cen1;
    double cen2;

    double skysig1;
    double skysig2;


    struct image *psf_image;

    double psf_cen1;
    double psf_cen2;

    double psf_skysig;

    double cen1_offset;
    double cen2_offset;
};

// the shortened pars
long ring_get_npars_short(enum gmix_model model, long *flags);

// creates a new ring pair, convolved wih the PSF and sheared

struct ring_pair *ring_pair_new(enum gmix_model model,
                                const double *pars, long npars,
                                enum gmix_model psf_model,
                                const double *psf_pars,
                                long psf_npars,
                                const struct shape *shear,
                                double s2n,
                                double cen1_offset,
                                double cen2_offset,
                                long *flags);

struct ring_pair *ring_pair_free(struct ring_pair *self);

void ring_pair_print(const struct ring_pair *self, FILE* stream);


struct ring_image_pair *ring_image_pair_new(const struct ring_pair *self, long *flags);

struct ring_image_pair *ring_image_pair_free(struct ring_image_pair *self);

#endif
