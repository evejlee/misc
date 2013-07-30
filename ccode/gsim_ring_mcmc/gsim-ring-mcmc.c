#include <stdlib.h>
#include <stdio.h>
#include "gmix.h"
#include "gmix_em.h"
#include "gsim_ring.h"
#include "shape.h"
#include "image.h"
#include "obs.h"
#include "time.h"
#include "fileio.h"

#include "mca.h"

#include "config.h"
#include "result.h"
#include "gmix_mcmc_config.h"
#include "gmix_mcmc.h"
#include "object.h"
#include "result.h"

#include "mca.h"

#include "randn.h"

// make an object list
struct obs_list *make_obs_list(const struct image *image,
                               const struct image *weight,
                               const struct image *psf_image,
                               long psf_ngauss,
                               double coord_row, // center of coord system
                               double coord_col,
                               long *flags)

{
    struct obs_list *self=obs_list_new(1);
    struct jacobian jacob = {0};

    jacobian_set_identity(&jacob);
    jacobian_set_cen(&jacob, coord_row, coord_col);

    obs_fill(&self->data[0],
             image,
             weight,
             psf_image,
             &jacob,
             psf_ngauss,
             flags);
    //jacobian_print(&self->data[0].jacob,stderr);

    if (*flags != 0) {
        self=obs_list_free(self);
    }
    return self;
}

// note row,col returned are the "center start", before offsetting,
// so we can properly use it in the prior
static
struct ring_image_pair *get_image_pair(struct gsim_ring *ring,
                                       double s2n,
                                       double *T, double *counts)
{
    struct ring_pair *rpair=NULL;
    struct ring_image_pair *impair=NULL;
    long flags=0;

    rpair = ring_pair_new(ring, s2n, &flags);
    if (flags != 0) {
        goto _get_image_pair_bail;
    }

    impair = ring_image_pair_new(rpair, &flags);

    if (flags != 0) {
        goto _get_image_pair_bail;
    }

    *T=gmix_get_T(rpair->gmix1);
    *counts=gmix_get_psum(rpair->gmix1);

_get_image_pair_bail:
    rpair = ring_pair_free(rpair);
    if (flags != 0) {
        impair = ring_image_pair_free(impair);
        fprintf(stderr,"failed to make image pair, aborting: %s: %d\n",
                __FILE__,__LINE__);
        exit(1);
    }

    return impair;
}

void print_one(const struct gmix_mcmc *self)
{
    //mca_stats_write_brief(self->chain_data.stats, stderr);
    mca_stats_write_flat(self->chain_data.stats, stdout);

    printf("%ld %.16g %.16g %.16g %.16g %.16g %.16g %.16g",
           self->nuse,
            self->P,
            self->Q[0],
            self->Q[1],
            self->R[0][0],
            self->R[0][1],
            self->R[1][0],
            self->R[1][1]);

    printf("\n");
}

//
// process all the PSFs in the set of observations
// need to move this code into gmix_em
//

static void fill_psf_gmix_guess(struct gmix *self, double row, double col, double counts)
{
    long flags=0;
    double frac = counts/self->size;
    for (long i=0; i<self->size; i++) {
        struct gauss2 *g=&self->data[i];
        gauss2_set(g,
                   frac + 0.1*srandu(),
                   row + srandu(),
                   col + srandu(),
                   2.0 + srandu(), 
                   0.0 + 0.1*srandu(), 
                   2.0 + srandu(),
                   &flags);
    }
}

static double get_em_sky_to_add(double im_min)
{
    return 0.001 - im_min;
}
static void process_psfs(struct gmix_mcmc *self)
{
    long n_retry = 100, try=0;
    struct gmix_em gmix_em={0};
    double min=0, max=0;

    gmix_em.maxiter=self->conf.em_maxiter;
    gmix_em.tol=self->conf.em_tol;

    const struct obs_list *obs_list=self->obs_list;
    for (long i=0; i<obs_list->size; i++) {
        const struct obs *obs=&obs_list->data[i];
        const struct image *psf_image = obs->psf_image;
        struct gmix *psf_gmix = obs->psf_gmix;

        // counts should be pre-sky addition
        double counts = image_get_counts(psf_image);

        struct image *psf_image_sky = image_new_copy(psf_image);
        image_get_minmax(psf_image, &min, &max);

        double sky_to_add = get_em_sky_to_add(min);
        image_add_scalar(psf_image_sky, sky_to_add);
        IM_SET_SKY(psf_image_sky, sky_to_add);

        double row=(IM_NROWS(psf_image)-1.0)/2.0;
        double col=(IM_NCOLS(psf_image)-1.0)/2.0;

        for (try=0; try<n_retry; try++) {
            fill_psf_gmix_guess(psf_gmix, row, col, counts);
            gmix_em_run(&gmix_em, psf_image_sky, psf_gmix);
            if (gmix_em.flags == 0) {
                break;
            }
        }

        if (try==n_retry) {
            // can later use or not use psf
            fprintf(stderr, "error processing psf failed %ld times, aborting: %s: %d\n", 
                    n_retry, __FILE__,__LINE__);
            exit(1);
        }

        psf_image_sky=image_free(psf_image_sky);
    }

}

void process_one(struct gmix_mcmc *self,
                 struct ring_image_pair *impair,
                 long pairnum,
                 double T,
                 double counts,
                 long *flags)
{
    double row_guess=0, col_guess=0;
    struct obs_list *obs_list=NULL;
    const struct image *im=NULL;
    const struct image *wt=NULL;

    if (pairnum==1) {
        im=impair->im1;
        wt=impair->wt1;
    } else {
        im=impair->im2;
        wt=impair->wt2;
    }
    obs_list = make_obs_list(im, wt, impair->psf_image,
                             self->conf.psf_ngauss,
                             impair->coord_cen1, // center of coord system
                             impair->coord_cen2, // center of coord system
                             flags);
    if (*flags != 0) {
        fprintf(stderr,"error making obs list\n");
        goto _process_one_bail;
    }

    gmix_mcmc_set_obs_list(self, obs_list);
    process_psfs(self);

    while (1) {
        *flags=0;
        gmix_mcmc_run(self, row_guess, col_guess, T, counts, flags);
        if (*flags != 0) {
            // this is currently fatal
            fprintf(stderr,"error running mcmc\n");
            goto _process_one_bail;
        }

        mca_chain_stats_fill(self->chain_data.stats, self->chain_data.chain);
        *flags |= gmix_mcmc_calc_pqr(self);
        if (*flags == 0) {
            double dnuse=(double)self->nuse;
            double frac = dnuse/MCA_CHAIN_NSTEPS(self->chain_data.chain);
            if (frac > GMIX_MCMC_MINFRAC_USE) {
                break;
            } else {
                fprintf(stderr,
                 "only %g used in pqr, re-trying with different guesses\n",frac);
            }
        } else if (*flags == GMIX_MCMC_NOPOSITIVE) {
            fprintf(stderr,
              "problem calculating pqr, re-trying with different guesses\n");
        } else {
            // should not happen
            fprintf(stderr,"fatal error in calculating pqr\n");
            goto _process_one_bail;
        }
    }


#if 0
    mca_chain_plot(self->chain_data.burnin_chain,"");
    mca_chain_plot(self->chain_data.chain,"");
#endif

_process_one_bail:
    obs_list = obs_list_free(obs_list);
    return;
}


// process and print results to stdout
// some log info goes to stderr
void process_pair(struct gsim_ring *ring,
                  struct gmix_mcmc *gmix_mcmc,
                  double s2n)
{

    long flags=0;
    double T=0, counts=0;

    struct ring_image_pair *impair = get_image_pair(ring, s2n, &T, &counts);

    process_one(gmix_mcmc,
                impair,
                1,
                T,
                counts,
                &flags);

    if (flags != 0) {
        goto _process_pair_bail;
    }
    print_one(gmix_mcmc);

    process_one(gmix_mcmc,
                impair,
                2,
                T,
                counts,
                &flags);

    if (flags != 0) {
        goto _process_pair_bail;
    }
    print_one(gmix_mcmc);

_process_pair_bail:
    impair = ring_image_pair_free(impair);
    if (flags != 0) {
        fprintf(stderr, "error processing pair, aborting: %s: %d\n", __FILE__,__LINE__);
        exit(1);
    }

    return;
}

static void print_header(long nlines, long npars)
{
    printf("SIZE =                  %16ld\n", nlines);
    printf("{'_DTYPE': [('npars', 'i2'),\n");
    printf("            ('arate', 'f8'),\n");
    printf("            ('pars', 'f8', %ld),\n", npars);
    printf("            ('pcov', 'f8', (%ld,%ld)),\n", npars, npars);
    printf("            ('nuse', 'i4'),\n");
    printf("            ('P', 'f8'),\n");
    printf("            ('Q', 'f8', 2),\n");
    printf("            ('R', '<f8', (2, 2))],\n");
    printf(" '_DELIM':' ',\n");
    printf(" '_VERSION': '1.0'}\n");
    printf("END\n");
    printf("\n");
}


static void run_sim(struct gsim_ring *ring,
                    struct gmix_mcmc *gmix_mcmc,
                    double s2n,
                    long npairs)
{

    print_header(npairs*2, gmix_mcmc->conf.npars);
    for (long i=0; i<npairs; i++) {
        fprintf(stderr,"%ld/%ld\n", i+1, npairs);
        process_pair(ring, gmix_mcmc, s2n);
    }
}

static struct gsim_ring *load_ring(const char *name)
{
    long flags=0;
    struct gsim_ring *self=gsim_ring_new_from_file(name, &flags);
    if (flags != 0) {
        fprintf(stderr,"fatal error reading ring config, exiting\n");
        exit(1);
    }
    gsim_ring_config_print(&ring->conf, stderr);

    return self;
}


static struct gmix_mcmc *load_gmix_mcmc(const char *name)
{
    long flags=0;
    struct gmix_mcmc *self=gmix_mcmc_new_from_config(name,&flags);
    if (flags != 0) {
        fprintf(stderr,"fatal error reading mcmc conf, exiting\n");
        exit(1);
    }
    gmix_mcmc_config_print(&self->conf, stderr);

    return self;
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        printf("usage: %s sim-conf gmix-mcmc-config s2n npairs\n", argv[0]);
        exit(1);
    }
    randn_seed();

    struct gsim_ring *ring=load_ring(argv[1]);
    struct gmix_mcmc *gmix_mcmc = load_gmix_mcmc(argv[2]);

    double s2n = atof(argv[3]);
    long npairs = atol(argv[4]);


    fprintf(stderr,"running sim with s2n: %g and npairs: %ld\n", s2n, npairs);
    run_sim(&ring, gmix_mcmc, s2n, npairs);
    fprintf(stderr,"finished running sim\n");

    // cleanup
    gsim_ring=gsim_ring_free(ring);
    gmix_mcmc = gmix_mcmc_free(gmix_mcmc);

    return 0;
}
