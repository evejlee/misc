/*
   Tools for adding noise to images using a gmix as the weight and
   a target s/n.
*/
#ifndef _GMIX_IMAGE_RAND_HEADER_GUARD
#define _GMIX_IMAGE_RAND_HEADER_GUARD

#include "image.h"
#include "gmix.h"

/*
   Add normal random deviates with sigma "skysig"
*/

int gmix_image_add_randn(struct image *image, 
                         double s2n,
                         const struct gmix *gmix,
                         double *skysig);

#endif
