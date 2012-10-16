#ifndef _GMIX_IMAGE_HEADER_GUARD
#define _GMIX_IMAGE_HEADER_GUARD

#define GMIX_IMAGE_NEGATIVE_DET 0x1

#include <stdlib.h>
#include "gmix.h"

/*

   Render the input gaussian mixture model into an image.
   A new image is created and returend.

*/
struct image *gmix_image_new(struct gmix *gmix, 
                             size_t nrows, 
                             size_t ncols, 
                             int nsub);

/*

   Render the input gaussian mixture model into the
   input image.

*/

int gmix_image_fill(struct image *image, 
                    struct gmix *gmix, 
                    int nsub);

/*

   calculate the ln(like) between the image and the input gaussian mixture

*/
double gmix_image_loglike(struct image *image, 
                          struct gmix *gmix, 
                          double ivar,
                          int *flags);

#endif
