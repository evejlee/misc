/*

   This is a quick app to run on some high s/n galsim 
   images.

   Only coellip for now.

*/
#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "gmix.h"
#include "gmix_image.h"
#include "gmix_image_fits.h"
#include "gmix_mcmc.h"
#include "mca.h"

#include "fitsio.h"

struct object {
    // location on object grid
    size_t orow;
    size_t ocol;

    // guess at center, just middle of region for each object
    double rowguess;
    double colguess;

    // boundary in main image
    struct image_mask mask;
};

struct fit_data {

    const struct image *image; // the input image
    double ivar;               // ivar for the image

    enum gmix_par_type par_type; // parameter array type

    struct gmix *obj;            // to be filled from pars array

    // optional
    const struct gmix *psf;      // pre-determined psf gmix
    struct gmix *conv;           // convolved gmix to be filled 
                                 // from psf and obj
};


struct fitter {
    double a;

    int burn_per_walker;
    int steps_per_walker;

    // we can keep these around
    struct mca_chain *burnin_chain;
    struct mca_chain *chain;

    // replace each time?
    struct mca_stats *stats;

    // replace each time?
    struct fit_data *psf_fit_data;
    struct fit_data *fit_data;
};

// read the bound info into the object
int object_bound_read(struct object *self, FILE* fptr)
{
    int nread=
        fscanf(stdin,"%lu %lu %lf %lf %lu %lu %lu %lu",
                &self->orow,&self->ocol,
                &self->rowguess, &self->colguess,
                &self->mask.rowmin,&self->mask.rowmax,
                &self->mask.colmin,&self->mask.colmax);
    return (nread==8);
}

struct fit_data 
*fit_data_new(const struct image *image,
              double ivar,
              enum gmix_par_type par_type,
              int ngauss, 
              const struct gmix *psf) // can be NULL
{
    struct fit_data *self=NULL;
    self=calloc(1,sizeof(struct fit_data));
    if (!self) {
        wlog("error: could not allocate struct "
             "fit_data.  %s: %d\n",__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }
    self->image=image;
    self->ivar=ivar;
    self->par_type=par_type;
    self->psf=psf;

    self->obj=gmix_new(ngauss);
    if (!self->obj) {
        wlog("error: could not allocate %d gmix "
             "%s: %d\n",ngauss,__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }

    if (self->psf) {

        int ngauss_psf=self->psf->size;
        int ntot=ngauss*ngauss_psf;
        self->conv=gmix_new(ntot);

        if (!self->conv) {
            wlog("error: could not allocate %d gmix "
                 "%s: %d\n",ntot,__FILE__,__LINE__);
            exit(EXIT_FAILURE);
        }
    }
    return self;
}

struct fit_data 
*fit_data_free(struct fit_data *self)
{
    if (self) {
        self->obj=gmix_free(self->obj);
        self->conv=gmix_free(self->conv);
        free(self);
    }
    return NULL;
}


double lnprob(const double *pars, 
              size_t npars, 
              const void *void_data)
{
    double lnp=0;
    int flags=0;

    struct fit_data *data=(struct fit_data*) void_data;

    const struct gmix *gmix=NULL;

    if (data->par_type == GMIX_COELLIP) {
        if (!gmix_fill_coellip(data->obj, pars, npars) ) {
            exit(EXIT_FAILURE);
        }
    } else if (data->par_type == GMIX_APPROX_DEV) {
        if (!gmix_fill_dev(data->obj, pars, npars) ) {
            exit(EXIT_FAILURE);
        }
    } else if (data->par_type == GMIX_APPROX_EXP) {
        if (!gmix_fill_dev(data->obj, pars, npars) ) {
            exit(EXIT_FAILURE);
        }
    } else {
        wlog("error: expected par_type of approx dev,exp or coellip "
                "but got %d. %s: %d\n",
                data->par_type,__FILE__,__LINE__);
        exit(EXIT_FAILURE);
    }


    if (data->psf) {
        if (!gmix_fill_convolve(data->conv, data->obj, data->psf)) {
            exit(EXIT_FAILURE);
        }
        gmix=data->conv;
    } else {
        gmix=data->obj;
    }

    // flags are only set for conditions we want to
    // propagate back through the likelihood
    lnp=gmix_image_loglike(data->image,
                           gmix,
                           data->ivar,
                           &flags);

    return lnp;
}


void process_object(struct object *obj)
{

    obj->flags=0;
}



int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr,"gmix-galsim image psf < objlist > output \n");
        exit(EXIT_FAILURE);
    }

    const char *image_file=argv[1];
    const char *psf_file=argv[2];

    struct image *image=image_read_fits(image_file,0);
    struct image *psf=image_read_fits(psf_file,0);

    struct object object={0};
    while (object_bound_read(&object,stdin)) {

        struct mca_stats *stats=object_process(&object,image,psf);

        fprintf(stdout,"%lu %lu ",object->orow,object->ocol);
        mca_stats_write_flat(self->stats,stdout);
        printf("\n");
    }
                     
    psf=image_free(psf);
    image=image_free(image);

}
