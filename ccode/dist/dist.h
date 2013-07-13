/*
   A set of distributions
*/
#ifndef _PRIORS_HEADER_GUARD
#define _PRIORS_HEADER_GUARD

#define DIST_LOG_LOWVAL -9.999e9
#define DIST_LOG_MINARG 1.0e-10

#define DIST_BAD_DIST 0x1
#define DIST_WRONG_NPARS 0x2

//#include "VEC.h"

enum dist {
    // 1d
    DIST_GAUSS,
    DIST_LOGNORMAL,
    // 2d
    DIST_G_BA,
    DIST_GMIX3_ETA 
};

// generic distributions
/*
struct dist1d {
    enum dist dist_type;
    void *data;
};
struct dist2d {
    enum dist dist_type;
    void *data;
};

struct dist1d *dist1d_new(enum dist dist_type, VEC(double) pars, long *flags);
struct dist2d *dist2d_new(enum dist dist_type, VEC(double) pars, long *flags);

struct dist1d *dist1d_free(struct dist1d *self);
struct dist2d *dist2d_free(struct dist2d *self);

double dist1d_lnprob(struct dist1d *self, double x);
double dist1d_prob(struct dist1d *self, double x);
double dist2d_lnprob(struct dist2d *self, double x, double y);
double dist2d_prob(struct dist2d *self, double x, double y);
*/

// these should always be value types, so they can be copied
// that means static sized fields
struct dist_gauss {
    enum dist dist_type;
    double mean;
    double sigma;
    double ivar;
};

struct dist_lognorm {
    enum dist dist_type;
    double mean;
    double sigma;

    double logmean;
    double logivar;
};

struct dist_g_ba {
    enum dist dist_type;
    double sigma;
    double ivar;
};

// 3 round gaussians centered at zero
// evaluation is easy,
//    prob = sum_i( N_i*exp( -0.5*ivar*( eta_1^2 + eta_2^2 ) )
// where N_i is amp_i/(2*PI)*ivar

struct dist_gmix3_eta {
    enum dist dist_type;
    double gauss1_ivar;
    double gauss1_pnorm; // amp*norm = amp*ivar/(2*PI)

    double gauss2_ivar;
    double gauss2_pnorm;

    double gauss3_ivar;
    double gauss3_pnorm;
};


enum dist dist_string2dist(const char *dist_name, long *flags);
long dist_get_npars(enum dist dist_type, long *flags);


struct dist_gauss *dist_gauss_new(double mean, double sigma);
void dist_gauss_fill(struct dist_gauss *self, double mean, double sigma);
double dist_gauss_lnprob(const struct dist_gauss *self, double x);
void dist_gauss_print(const struct dist_gauss *self, FILE *stream);

struct dist_lognorm *dist_lognorm_new(double mean, double sigma);
void dist_lognorm_fill(struct dist_lognorm *self, double mean, double sigma);
double dist_lognorm_lnprob(const struct dist_lognorm *self, double x);
void dist_lognorm_print(const struct dist_lognorm *self, FILE *stream);

struct dist_g_ba *dist_g_ba_new(double sigma);
void dist_g_ba_fill(struct dist_g_ba *self, double sigma);
double dist_g_ba_lnprob(const struct dist_g_ba *self, double g1, double g2);
double dist_g_ba_prob(const struct dist_g_ba *self, double g1, double g2);
void dist_g_ba_print(const struct dist_g_ba *self, FILE *stream);

void dist_gmix3_eta_fill(struct dist_gmix3_eta *self,
                         double sigma1, double sigma2, double sigma3,
                         double p1, double p2, double p3);



struct dist_gmix3_eta *dist_gmix3_eta_new(double sigma1,
                                          double sigma2,
                                          double sigma3,
                                          double p1,
                                          double p2,
                                          double p3);
                                          
double dist_gmix3_eta_lnprob(const struct dist_gmix3_eta *self, double eta1, double eta2);
double dist_gmix3_eta_prob(const struct dist_gmix3_eta *self, double eta1, double eta2);
void dist_gmix3_eta_print(const struct dist_gmix3_eta *self, FILE *stream);

#endif

