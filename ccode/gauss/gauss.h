#ifndef _GAUSS_HEADER_GUARD
#define _GAUSS_HEADER_GUARD

#ifndef M_TWO_PI
#define M_TWO_PI   6.28318530717958647693
#endif

struct gauss {
    double p;
    double row;
    double col;
    double irr;
    double irc;
    double icc;

    // derived quantities
    double det;

    double e1;
    double e2;

    double drr; // irr/det
    double drc;
    double dcc;

    double norm; // 1/( 2*pi*sqrt(det) )
};

// use this to keep the structure internally consistent
int gauss_set(struct gauss *self,
              double p,
              double row,
              double col,
              double irr,
              double irc,
              double icc);


#endif
