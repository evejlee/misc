/*
 
    Algorithm is simple:

    Start with a guess for the N gaussians.

    Render the gaussians on the pixel grid.  Then the new estimate
    for the gaussian weight "p" are for gaussian i is

        pnew[gi] = sum_pix( gi[pix]/gtot[pix]*imnorm[pix] )

    where imnorm is image/sum(image) and gtot[pix] is
        
        gtot[pix] = sum_gi(gi[pix]) + nsky

    and nsky is the sky/sum(image)
    
    These new p[gi] can then be used to update the mean and covariance
    as well.  To update the mean in coordinate x

        mx[gi] = sum_pix( gi[pix]/gtot[pix]*imnorm[pix]*x )/pnew[gi]
 
    where x is the pixel value in either row or column.

    Similarly for the covariance for coord x and coord y.

        cxy = sum_pix(  gi[pix]/gtot[pix]*imnorm[pix]*(x-xc)*(y-yc) )/pcen[gi]

    setting x==y gives the diagonal terms.

    Then repeat until some tolerance in the moments is achieved.

    These calculations can be done very efficiently within a single loop.
    
 */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "gmix_image.h"
#include "image.h"
#include "gvec.h"
#include "defs.h"

/*
 * This function is way too long, but we care about speed and there are a lot
 * of variables.  Better to get them on the stack once and be done with it
 */

int gmix_image(struct gmix* self,
               struct image *image, 
               struct gvec *gvec,
               size_t *iter)
{
    int flags=0;
    size_t i=0;
    size_t ngauss = gvec->size;
    size_t nbytes = gvec->size*sizeof(double);
    double det=0, chi2=0, b=0;
    double u=0, v=0, uv=0, u2=0, v2=0, igrat=0;
    double gtot=0, imnorm=0, skysum=0.0, tau=0;
    double wmomlast=0, wmom=0, wmomdiff=0, psum=0;

    struct gauss* gauss=NULL;

    double sky=image_sky(image);
    double counts=image_counts(image);
    size_t npoints = IMSIZE(image);

    double nsky = sky/counts;
    double psky = sky/(counts/npoints);

    // these are all stack allocated

    double *gi = alloca(ngauss);
    double *trowsum = alloca(ngauss);
    double *tcolsum = alloca(ngauss);
    double *tu2sum = alloca(ngauss);
    double *tuvsum = alloca(ngauss);
    double *tv2sum = alloca(ngauss);

    // these need to be zeroed on each iteration
    double *pnew = alloca(ngauss);
    double *rowsum = alloca(ngauss);
    double *colsum = alloca(ngauss);
    double *u2sum = alloca(ngauss);
    double *uvsum = alloca(ngauss);
    double *v2sum = alloca(ngauss);

    wmomlast=-9999;
    *iter=0;
    while (*iter < self->maxiter) {
        if (self->verbose)
            gvec_print(stderr,gvec);

        skysum=0;
        psum=0;
        memset(pnew,0,nbytes);
        memset(rowsum,0,nbytes);
        memset(colsum,0,nbytes);
        memset(u2sum,0,nbytes);
        memset(uvsum,0,nbytes);
        memset(v2sum,0,nbytes);

        for (size_t col=0; col<image->ncols; col++) {
            for (size_t row=0; row<image->nrows; row++) {

                imnorm=IMGET(image,row,col);
                imnorm /= counts;

                gtot=0;
                gauss = &gvec->data[0];
                for (i=0; i<ngauss; i++) {
                    det = gauss->irr*gauss->icc - gauss->irc*gauss->irc;
                    det = fabs(det);
                    if (det == 0) {
                        flags+=GMIX_ERROR_NEGATIVE_DET;
                        goto _gmix_image_bail;
                    }
                    u = (row-gauss->row);
                    v = (col-gauss->col);

                    u2 = u*u;
                    v2 = v*v;
                    uv = u*v;

                    chi2=gauss->icc*u2 + gauss->irr*v2 - 2.0*gauss->irc*uv;
                    chi2 /= det;
                    b = M_TWO_PI*sqrt(det);

                    gi[i] = gauss->p*exp( -0.5*chi2 )/b;
                    gtot += gi[i];

                    trowsum[i] = row*gi[i];
                    tcolsum[i] = col*gi[i];
                    tu2sum[i]  = u2*gi[i];
                    tuvsum[i]  = uv*gi[i];
                    tv2sum[i]  = v2*gi[i];

                    gauss++;
                }
                gtot += nsky;
                igrat = imnorm/gtot;
                for (i=0; i<ngauss; i++) {
                    tau = gi[i]*igrat;  // Dave's tau*imnorm
                    pnew[i] += tau;
                    psum += tau;

                    rowsum[i] += trowsum[i]*igrat;
                    colsum[i] += tcolsum[i]*igrat;
                    u2sum[i]  += tu2sum[i]*igrat;
                    uvsum[i]  += tuvsum[i]*igrat;
                    v2sum[i]  += tv2sum[i]*igrat;
                }
                skysum += nsky*imnorm/gtot;

            } // rows
        } // cols

        wmom=0;
        gauss=gvec->data;
        for (i=0; i<ngauss; i++) {
            gauss->p   = pnew[i];
            gauss->row = rowsum[i]/pnew[i];
            gauss->col = colsum[i]/pnew[i];
            gauss->irr = u2sum[i]/pnew[i];
            gauss->irc = uvsum[i]/pnew[i];
            gauss->icc = v2sum[i]/pnew[i];
            wmom += gauss->p*gauss->irr + gauss->p*gauss->icc;

            gauss++;
        }
        psky = skysum;

        wmom /= psum;
        wmomdiff = fabs(wmom-wmomlast);
        if (wmomdiff/wmom < self->tol) {
            break;
        }
        wmomlast = wmom;
        (*iter)++;
    }

_gmix_image_bail:
    if (self->maxiter == (*iter)) {
        flags += GMIX_ERROR_MAXIT;
    }
    return flags;
}
