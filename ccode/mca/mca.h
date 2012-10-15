/*
   mca - a library implementing affine invariant MCMC as
   outlined in Goodman & Weare 2010.

   Some implementation inspiration from the Emcee python version.
*/
#ifndef _MCA_HEADER_GUARD
#define _MCA_HEADER_GUARD


#include <stdlib.h>
#include <stdio.h>
#include <float.h>

#define MCA_LOW_VAL (-DBL_MAX)

struct mca_chain {
    size_t nwalkers;
    size_t steps_per_walker;
    size_t npars;

    // indexed by (iwalker,istep,ipar)
    //    npars*steps_per_walker*iwalker + npars*istep + ipar
    // where istep is in [0,steps_per_walker)
    // or ignoring walkers by (istep,ipar)
    //    npars*istep + ipar
    // where istep is in [0,nwalkers*nsteps_per_walker)

    double *pars;

    // index by (iwalker,istep)
    //   steps_per_walker*iwalker + istep
    // or ignoring walkers by (istep)
    double *lnprob;

    int *accept;
};

struct mca_stats {
    size_t npars;

    double *mean;

    /* index by npar*i + j */
    double *cov;
};

/*
struct mca {
    // holds the pars and lnprob
    struct mca_chain *chain;

    // This is the function to calculate the log(prob)
    double (*lnprob_func)(double *pars, size_t npars, void *userdata);

    // The user can store data here for the log(prob) function
    // This field will not be set or freed by mca_run()
    const void *userdata;
};
*/

#define MCA_CHAIN(mca) (mca)->chain

#define MCA_CHAIN_NPARS(chain)  (chain)->npars
#define MCA_CHAIN_NWALKERS(chain)  (chain)->nwalkers
#define MCA_CHAIN_NSTEPS(chain)  \
    (chain)->steps_per_walker*(chain)->nwalkers
#define MCA_CHAIN_NSTEPS_BYWALKER(chain)  (chain)->steps_per_walker


#define MCA_CHAIN_PAR_BYWALKER(chain, iwalker, istep, ipar)    \
    (chain)->pars[                                             \
        (chain)->npars*(chain)->steps_per_walker*(iwalker)       \
      + (chain)->npars*(istep)                                   \
      + (ipar)                                                   \
    ]
#define MCA_CHAIN_PAR(mca, istep, ipar)    \
    (chain)->pars[ (chain)->npars*(istep)  + (ipar) ]


#define MCA_CHAIN_LNPROB_BYWALKER(chain, iwalker, istep)        \
    (chain)->lnprob[                               \
        (chain)->steps_per_walker*(iwalker)          \
      + (istep)                                           \
    ]
#define MCA_CHAIN_LNPROB(chain, istep)    \
    (chain)->lnprob[(istep)]

#define MCA_CHAIN_ACCEPT_BYWALKER(chain, iwalker, istep)        \
    (chain)->accept[                               \
        (chain)->steps_per_walker*(iwalker)          \
      + (istep)                                           \
    ]
#define MCA_CHAIN_ACCEPT(chain, istep)    \
    (chain)->accept[(istep)]


#define MCA_STATS_NPARS(stats) (stats)->npars

#define MCA_STATS_MEAN(stats, i) ({                                           \
    double _mn=-MCA_LOW_VAL;                                                  \
    if ((i) >= (stats)->npars) {                                              \
        fprintf(stderr,                                                       \
            "stats error: mean index %lu out of bounds [0,%lu)\n",            \
                        (i), (stats)->npars);                                 \
        fprintf(stderr,"returning %.16g\n", _mn);                             \
    } else {                                                                  \
        _mn = (stats)->mean[(i)];                                             \
    }                                                                         \
    _mn;                                                                      \
})

#define MCA_STATS_COV(stats, i, j) ({                                         \
    double _cov=-MCA_LOW_VAL;                                                 \
    if ((i) >= (stats)->npars || (j) >= (stats)->npars) {                     \
        fprintf(stderr,                                                       \
            "stats error: cov index (%lu,%lu) out of bounds [0,%lu)\n",       \
                        (i), (j),(stats)->npars);                             \
        fprintf(stderr,"returning %.16g\n", _cov);                            \
    } else {                                                                  \
        _cov= (stats)->cov[(i)*(stats)->npars + (j)];                         \
    }                                                                         \
    _cov;                                                                     \
})


// here istep is within the walker's sub-chain
/*
#define MCA_PAR_BYWALKER(mca, iwalker, istep, ipar)                 \
    MCA_CHAIN_PAR_BYWALKER((mca)->chain, iwalker, istep, ipar)      \

// here istep is the overall step, [0,nwalkers*steps_per_walker)
#define MCA_PAR(mca, istep, ipar)                 \
    MCA_CHAIN_PAR((mca)->chain, istep, ipar)      \

#define MCA_LNPROB_BYWALKER(mca, iwalker, istep)        \
    MCA_CHAIN_LNPROB_BYWALKER((mca)->chain, iwalker, istep)
#define MCA_LNPROB(mca, istep)    \
    MCA_CHAIN_LNPROB((mca)->chain, istep)


#define MCA_NSTEPS_BYWALKER(mca) \
    MCA_CHAIN_NSTEPS_BYWALKER((mca)->chain)
#define MCA_NSTEP(mca) \
    MCA_CHAIN_NSTEPS((mca)->chain)
*/





/*
*/
/*
struct mca *mca_new(size_t nwalkers,
                    size_t steps_per_walker,
                    size_t npars,
                    double (*lnprob_func)(double *pars, size_t npars, void *userdata),
                    const void *userdata);
struct mca *mca_del(struct mca *self);
*/

struct mca_chain *mca_chain_new(size_t nwalkers,
                                size_t steps_per_walker,
                                size_t npars);
struct mca_chain *mca_chain_del(struct mca_chain *self);

void mca_chain_print(struct mca_chain *chain, FILE *stream);

struct mca_chain *mca_make_guess(double *centers, 
                                 double *ballsizes,
                                 size_t npars, 
                                 size_t nwalkers);

struct mca_stats *mca_stats_new(size_t npar);
struct mca_stats *mca_stats_del(struct mca_stats *self);
struct mca_stats *mca_chain_stats(struct mca_chain *chain);
void mca_stats_print(struct mca_stats *self, FILE *stream);
void mca_stats_print_full(struct mca_stats *self, FILE *stream);

/*

   Fill the chain with MCMC steps.

   The *last* set of walkers in the "start" chain will be the starting point
   for the chain.  This way you can feed a start as a single chain or as the
   last step of a previous burn-in run.

   take input pars and add random errors in a ball
   for each walker

   loop over steps
     loop over walkers
       choose a random walker from the complement
       make a stretch move
       if accept
           copy new pars
       else 
           copy old pars
*/

void mca_run(double a,
             const struct mca_chain *start,
             struct mca_chain *chain,
             double (*lnprob_func)(double *pars, size_t npars, void *userdata),
             void *userdata);

/* copy the last step in the start chain to the first step
   in the chain */
void mca_set_start(const struct mca_chain *start,
                   struct mca_chain *chain);

/*
   make a stretch move

   comp_pars are from a randomly drawn walker from the set complement of the
   current walker.

   newpar = comp_par + z*(par-comp_par)

   where z is drawn from the mca_rand_gofz, which is
        1/sqrt(z) in the range [1/a,a]

   Note this is a *vectoral* move; z is a scalar in that vectoral equation.

   On return the value of z is set and the newpars are filled in.

*/
void mca_stretch_move(double a,
                      const double *pars, 
                      const double *comp_pars, 
                      size_t ndim,
                      double *newpars,
                      double *z);

/*
   Determine of a stretch move should be accepted

   Returns 1 if yes 0 if no
*/
int mca_accept(int ndim,
               double lnprob_old,
               double lnprob_new,
               double z);

/*
   copy the parameters
*/
void mca_copy_pars(const double *pars_src, double *pars_dst, size_t npars);

/* 
   generate random integers in [0,n)

   Don't forget to seed srand48!
 */
long mca_rand_long(long n);

/*

   get a random long index in [0,n) from the *complement* of the input
   current value, i.e. such that index!=current

*/
long mca_rand_complement(long current, long n);

/*
   generate random numbers 
       {
       { 1/sqrt(z) if z in (1/a,a)
       {     0     otherwise
       {

   When used in the affine invariant mcmc, a value
   of a=2 gives about 50% acceptance rate
 */
double mca_rand_gofz(double a);


/*
   Generate normal random numbers.

   Note we get two per run but I'm only using one.
*/
double mca_randn();


#endif
