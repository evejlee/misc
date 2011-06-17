#ifndef _HISTOGRAM_H
#define _HISTOGRAM_H

#include <stdint.h>
#include "Vector.h"

// special simplified binsize=1, integer histogrammer
// The simplifications from binsize=1 and integer are significant
void i64hist1(const struct i64vector* vec,
              const struct szvector* sort_index,
              struct i64vector* h,
              struct i64vector* rev);



#endif
