#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

#include "../image.h"
#include "../gmix.h"
#include "../gmix_image.h"
#include "../gmix_mcmc.h"
#include "../admom.h"

// these will hold pointers to data used in the fits
const struct image *_global_image_tmp;
struct gmix *_global_gmix_tmp;
double _global_ivar_tmp;

double lnprob(const double *pars, size_t npars, const void *void_data)
{
    double lnprob=0;
    int flags=0;

    gmix_fill_coellip(_global_gmix_tmp, pars, npars);

    // flags are only set for conditions we want to
    // propagate back through the likelihood
    lnprob=gmix_image_loglike(_global_image_tmp,
                              _global_gmix_tmp, 
                              _global_ivar_tmp,
                              &flags);

    return lnprob;
}
size_t get_pars_and_guess(int nrow, int ncol,
                          double **pars_true, double **guess, double **widths)
{
    size_t ngauss=1;
    size_t npars=2*ngauss+4;

    (*pars_true)=calloc(npars, sizeof(double));
    (*guess)    =calloc(npars, sizeof(double));
    (*widths)   =calloc(npars, sizeof(double));

    (*pars_true)[0] = 0.5*nrow;
    (*pars_true)[1] = 0.5*ncol;
    (*pars_true)[2] = 0.2;
    (*pars_true)[3] = 0.1;
    (*pars_true)[4] = 12.;
    (*pars_true)[5] = 1.;

    (*widths)[0] = 1.0;
    (*widths)[1] = 1.0;
    (*widths)[2] = 0.1;
    (*widths)[3] = 0.1;
    (*widths)[4] = 0.1*(*pars_true)[4];
    (*widths)[5] = 0.1*(*pars_true)[5];

    (*guess)[0] = (*pars_true)[0] + (*widths)[0]*(drand48()-0.5);
    (*guess)[1] = (*pars_true)[1] + (*widths)[1]*(drand48()-0.5);
    (*guess)[2] = (*pars_true)[2] + (*widths)[2]*(drand48()-0.5);
    (*guess)[3] = (*pars_true)[3] + (*widths)[3]*(drand48()-0.5);
    (*guess)[4] = (*pars_true)[4] + (*widths)[4]*(drand48()-0.5);
    (*guess)[5] = (*pars_true)[5] + (*widths)[5]*(drand48()-0.5);

    return npars;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr,"test-gauss s2n\n");
        exit(1);
    }
    double s2n=atof(argv[1]);

    char image_fname[] = "tmp/test-image.dat";
    char noisy_image_fname[] = "tmp/test-image-noisy.dat";
    char burn_fname[] = "tmp/chain-burnin.dat";
    char chain_fname[] = "tmp/chain.dat";
    //char fit_fname[] = "test-image-fit.dat";
    size_t nrow=25, ncol=25;
    int nsub=1;
    size_t nwalkers=20;
    size_t burn_per_walker=200;
    size_t steps_per_walker=200;
    double a=2;
    double skysig=0;
    double s2n_meas=0;

    time_t tm;
    (void) time(&tm);
    srand48((long) tm);

    double *pars_true=NULL, *guess=NULL, *widths=NULL;
    size_t npars = get_pars_and_guess(nrow,ncol,&pars_true, &guess, &widths);

    if (0 != system("mkdir -p tmp")) {
        fprintf(stderr,"could not make ./tmp");
        exit(1);
    }
    struct gmix *gmix_true=gmix_make_coellip(pars_true, npars);


    // make the image and noisy image
    struct image* image = gmix_image_new(gmix_true, nrow, ncol, nsub);
    wlog("storing image in '%s'\n", image_fname);
    image_write_file(image, image_fname);

    struct image *noisy_im = image_newcopy(image);
    // need to fix this program
    admom_add_noise(noisy_im, s2n, &gmix_true->data[0], &skysig, &s2n_meas);
    double ivar = 1./(skysig*skysig);
    wlog("storing noisy image in '%s'\n", noisy_image_fname);
    image_write_file(noisy_im, noisy_image_fname);

    int ngauss=1;
    struct gmix *gmix_tmp=gmix_new(ngauss);

    // global variables hold pointers to data
    _global_image_tmp = (const struct image*) noisy_im;
    _global_gmix_tmp = gmix_tmp;
    _global_ivar_tmp = ivar;

    wlog("building guesses for %lu walkers\n", nwalkers);
    struct mca_chain *start_chain = gmix_mcmc_make_guess_coellip(guess, widths, npars, nwalkers);

    wlog("creating burn-in chain for %lu steps per walker\n", burn_per_walker);
    struct mca_chain *burnin_chain=mca_chain_new(nwalkers, burn_per_walker, npars);

    wlog("    running burn-in\n");
    mca_run(burnin_chain, a, start_chain, &lnprob, NULL);

    wlog("    writing burn chain to %s\n", burn_fname);
    mca_chain_write_file(burnin_chain, burn_fname);

    wlog("creating chain for %lu steps per walker\n", steps_per_walker);
    struct mca_chain *chain=mca_chain_new(nwalkers, steps_per_walker, npars);
    wlog("    running chain\n");
    mca_run(chain, a, burnin_chain, &lnprob, NULL);

    wlog("    writing chain to %s\n", chain_fname);
    mca_chain_write_file(chain, chain_fname);

    wlog("brief stats\n");
    struct mca_stats *stats = mca_chain_stats(chain);
    mca_stats_write_brief(stats, stderr);


    start_chain=mca_chain_free(start_chain);
    burnin_chain=mca_chain_free(burnin_chain);
    chain=mca_chain_free(chain);
    stats=mca_stats_free(stats);

    return 0;
}
