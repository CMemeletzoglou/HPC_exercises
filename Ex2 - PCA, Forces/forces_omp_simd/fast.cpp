#include <cmath>
#include <x86intrin.h>
#include <omp.h>

#include "particles.h"
#include "utils.h"

/**
 * Compute the gravitational forces in the system of particles
 * Use AVX and OpenMP to speed up computation
**/
void computeGravitationalForcesFast(Particles& particles)
{
}

int main()
{
	testAll(computeGravitationalForcesFast);
	return 0;
}
