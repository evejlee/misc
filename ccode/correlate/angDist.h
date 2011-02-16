#include <algorithm>
#include <cmath>
#include "types.h"

// Overloaded: this one calculates the andgist between zero and input redshift
float
angDist(float H0, float omega_m, float zmin, float zmax);

// Overloaded: This one takes in a minimum redshift
float
angDist(float H0, float omega_m, float z);

// Conformal time
float
aeta(float z, float omega_m);

