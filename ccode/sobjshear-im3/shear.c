#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#include "defs.h"
#include "shear.h"
#include "sconfig.h"
#include "cosmo.h"
#include "healpix.h"
#include "lens.h"
#include "lensum.h"
#include "source.h"
#include "interp.h"
#include "tree.h"
#include "sdss-survey.h"
#include "quad.h"
#include "log.h"


struct shear* shear_init(const char* config_url, const char* lens_url) {

    struct shear* shear = calloc(1, sizeof(struct shear));
    if (shear == NULL) {
        wlog("Failed to alloc shear struct\n");
        exit(EXIT_FAILURE);
    }

    shear->config=sconfig_read(config_url);

    struct sconfig* config=shear->config;
    wlog("config structure:\n");
    sconfig_print(config);

    // now initialize the structures we need
    wlog("Initalizing cosmo in flat universe\n");
    int flat=1;
    double omega_k=0;
    shear->cosmo = cosmo_new(config->H0, flat, 
                             config->omega_m, 
                             1-config->omega_m, 
                             omega_k);

    wlog("cosmo structure:\n");
    cosmo_print(shear->cosmo);

    wlog("Initalizing healpix at nside: %ld\n", config->nside);
    shear->hpix = hpix_new(config->nside);

    shear->lcat = lcat_read(lens_url);

    // order is important here
    wlog("Adding Da to lenses\n");
    lcat_add_da(shear->lcat, shear->cosmo);
    wlog("Adding cos(search_angle) to lenses\n");
    lcat_add_search_angle(shear->lcat, config->rmax);
    wlog("Intersecting all lenses with healpix at rmax: %lf\n", config->rmax);
    lcat_disc_intersect(shear->lcat, shear->hpix, config->rmax);
    wlog("Building hpix tree for lenses\n");
    lcat_build_hpix_tree(shear->hpix, shear->lcat);

    lcat_print_firstlast(shear->lcat);

    // this holds the sums for each lens
    shear->lensums = lensums_new(shear->lcat->size, config->nbin);
    for (size_t i=0; i<shear->lensums->size; i++) {
        shear->lensums->data[i].zindex = shear->lcat->data[i].zindex;
    }

    if (config->scstyle == SCSTYLE_INTERP) {
        // interpolation region
        double* zl=config->zl->data;
        int64 nzl=config->nzl;
        shear->max_zlens = zl[nzl-1];
        shear->min_zlens = fmax(zl[0], shear->config->min_zlens_interp);
    } else {
        shear->min_zlens = 0;
        shear->max_zlens = 9999;
    }

    wlog("min_zlens: %lf  max_zlens: %lf\n", shear->min_zlens, shear->max_zlens);


    return shear;

}

void shear_process_source(struct shear* self, struct source* src) {
    src->hpixid = hpix_eq2pix(self->hpix, src->ra, src->dec);

    int64 hpix_mod = src->hpixid - self->hpix->half_npix;
    struct tree_node* node = tree_find(self->lcat->hpix_tree, hpix_mod);

    if (node == NULL) {
        return;
    }

    struct lens* lens=NULL;
    struct lensum* lensum=NULL;
    size_t ind=0;
    for (size_t i=0; i<node->indices->size-1; i++) {
        ind = node->indices->data[i];

        lens = &self->lcat->data[ind];
        lensum = &self->lensums->data[ind];

        if (lens->z >= self->min_zlens && lens->z <= self->max_zlens) {
            shear_procpair(self, src, lens, lensum);
        }
    }

}

static void reset_longitude_bounds(double *angle,
                                   double min,
                                   double max)
{
    while (*angle<min) {
        *angle += 360.0;
    }
    while (*angle>=max) {
        *angle -= 360.0;
    }
    return;
}

/*
   r, posangle in radians

   returns 0 if the radius is beyond the max allowed radius as defined in
   lens->cos_search_angle. In that case fewer calculations are performed

*/
static int get_quad_info(const struct lens* lens,
                         const struct source* src,
                         int *quadrant,
                         double *r_radians,
                         double *posangle_radians)
{
    double cosradiff, sinradiff, arg, cos_r, pa;

    cosradiff = src->cosra*lens->cosra + src->sinra*lens->sinra;
    sinradiff = src->sinra*lens->cosra - src->cosra*lens->sinra;

    // cos of angle separation
    cos_r = lens->sindec*src->sindec + lens->cosdec*src->cosdec*cosradiff;

    if (cos_r < lens->cos_search_angle) {
        // outside search radius
        return 0;
    }

    if (cos_r > 1.0)
        cos_r=1.0;
    else if (cos_r < -1.0)
        cos_r=-1.0;

    // angle of separation in radians
    (*r_radians) = acos(cos_r);

    // position angle
    arg = lens->sindec*cosradiff - lens->cosdec*src->sindec/src->cosdec;

    // -pi,pi
    (*posangle_radians) = atan2(sinradiff, arg) - M_PI_2;

    // degrees -180,180
    pa = (*posangle_radians)*R2D;
    reset_longitude_bounds(&pa, -180.0, 180.0);

    (*quadrant) = quadeq_get_quadrant(pa);

    return 1;
}

/*

  SDSS specific, make sure the source is in an acceptable quadrant for this
  lens

 */
static int shear_test_quad_sdss(struct lens* l, struct source* s) {
    return test_quad_sincos_sdss(l->maskflags,
                                 l->sinlam, l->coslam,
                                 l->sineta, l->coseta,
                                 s->sinlam, s->coslam,
                                 s->sineta, s->coseta);
}




void shear_procpair(struct shear* self, 
                    struct source* src, 
                    struct lens* lens, 
                    struct lensum* lensum) {

    struct sconfig* config=self->config;

    // make sure object is in a pair of unmasked adjacent
    // quadrants.  Using short-circuiting in if statement


    // checks against max radius
    int quadrant;
    double r_radians, posangle_radians;
    int keep=get_quad_info(lens, src,
                           &quadrant,
                           &r_radians,
                           &posangle_radians);

    if (!keep) {
        goto _procpair_bail;
    }

    if (config->mask_style == MASK_STYLE_EQ) {
        if (!quadeq_check_quadrant(lens->maskflags, quadrant)) {
            goto _procpair_bail;
        }
    } else if (config->mask_style == MASK_STYLE_SDSS) {
        if (!shear_test_quad_sdss(lens, src)) {
            goto _procpair_bail;
        }
    }

    double cos2pa = cos(2*posangle_radians);
    double sin2pa = sin(2*posangle_radians);

    // note we already checked if lens z was in our interpolation range
    double scinv;
    if (src->scstyle == SCSTYLE_INTERP) {
        scinv = f64interplin(src->zlens, src->scinv, lens->z);
    } else {
        double dcl = lens->da*(1.+lens->z);
        scinv = scinv_pre(lens->z, dcl, src->dc);
    }

    if (scinv <= 0) {
        goto _procpair_bail;
    }

    double r;
    if (config->r_units==UNITS_MPC) {
        // Mpc
        r = r_radians*lens->da;
    } else {
        // arcmin
        r = r_radians*R2D*60.;
    }

    double logr = log10(r);
    int rbin = (int)( (logr-config->log_rmin)/config->log_binsize );

    if (rbin < 0 || rbin >= config->nbin) {
        goto _procpair_bail;
    }


    double scinv2 = scinv*scinv;

    double gt = -(src->g1*cos2pa + src->g2*sin2pa);
    double gx =  (src->g1*sin2pa - src->g2*cos2pa);

    double eweight = src->weight;
    double weight = scinv2*eweight;

    double x = r*cos(posangle_radians);
    double y = r*sin(posangle_radians);

    lensum->weight += weight;
    lensum->totpairs += 1;

    lensum->x2sum += weight*x*x;
    lensum->y2sum += weight*y*y;
    lensum->xysum += weight*x*y;

    lensum->npair[rbin] += 1;

    lensum->wsum[rbin] += weight;
    lensum->dsum[rbin] += weight*gt/scinv;
    lensum->osum[rbin] += weight*gx/scinv;

    lensum->rsum[rbin] += r;


_procpair_bail:

    return;

}




void shear_print_sum(struct shear* self) {
    wlog("Total sums:\n\n");

    lensums_print_sum(self->lensums);

}

// this is for when we haven't written the file line by line[:w
void shear_write(struct shear* self, FILE* stream) {
    lensums_write(self->lensums, stream);
}

struct shear* shear_delete(struct shear* self) {

    if (self != NULL) {

        self->config   = sconfig_delete(self->config);
        self->lcat     = lcat_delete(self->lcat);
        self->hpix     = hpix_delete(self->hpix);
        self->cosmo    = cosmo_delete(self->cosmo);
        self->lensums  = lensums_delete(self->lensums);

    }
    free(self);
    return NULL;
}


