#include <cmath>
#include <x86intrin.h> // x86 Intrinsic functions, contains the headers needed for each microarchitecture
#include <omp.h>
#include <cstring> // memcpy

#include "particles.h"
#include "utils.h"

// Helper function to print the contents of a 256-bit AVX register
static void print_avx(const char * label, __m256d v)
{
	std::cout << std::endl;
	__attribute__((aligned(32))) double a[4];
	std::cout << label << ":\t" << "\n";

	memcpy(a, &v, sizeof(a));

	for (int i = 0; i < 4; i++)
		std::cout << a[i] << '\t';

	std::cout << '\n';
}

/*
 * Compute the gravitational forces in the system of particles
 * Use AVX and OpenMP to speed up computation
 */
void computeGravitationalForcesFast(Particles& particles)
{
	int ndiv4 = particles.n / 4;

	const double G = 6.67408e-11;
	__m256d _g_const = _mm256_set1_pd(G);

	#pragma omp parallel for
	for (int i=0; i<particles.n; i++)
	{
		// load current particle's xyz coordinates and mass
		__m256d _xi = _mm256_set1_pd(particles.x[i]); 
		__m256d _yi = _mm256_set1_pd(particles.y[i]);
		__m256d _zi = _mm256_set1_pd(particles.z[i]);
		__m256d _mi = _mm256_set1_pd(particles.m[i]);

		// initialize the AVX registers that will store the sum of forces for each dimension
		__m256d _fxi = _mm256_setzero_pd();
		__m256d _fyi = _mm256_setzero_pd();
		__m256d _fzi = _mm256_setzero_pd();

		// Create a helper register that contains the index i
		__m256d _i = _mm256_set1_pd(i);

		for (int j = 0; j < 4*ndiv4 ; j+=4)
		{
			/* With the outer (i-loop) we choose a current particle and with the inner
			 * (j-loop) we choose a group of 4 particles, that will be used to compute
			 * the force exerted by them onto the i-th particle. 
			 * 
			 * In the original version (vanilla.cpp), an if statement is used in order to
			 * avoid computing the force exerted by particle i onto itself. 
			 
			 * However, when performing SIMD operations, we use consecutive elements
			 * and thus, we cannot "skip" some elements. 
			 * Therefore, we cannot avoid calculating the force exerted by
			 * particle i, onto itself.
			 * 
			 * Thus, we don't need that value, we are going to "mask" it out, using the 
			 * 256-bit value returned by _mm256_cmp_pd, since there is no other way
			 * to find if i is in [j,j+3].
			 * In order to have an equivalent of the "if(i !=j)" of the vanilla code,
			 * we will compare the values of two 256-bit registers, one loaded the value
			 * of i (reg. _i) and one loaded with the next *4* values of j (reg. _j).
			 * We, then, check if two 64-bit (double) elements, of the two registers, are different.
			 * In the register positions where this is true, the output mask contains 
			 * 0xFFFF...FF and 0, otherwise. 
			 * 
			 * The 256-bit return value of _mm256_cmp_pd, will only be zero in **one** position,
			 * the position where i == j. This position indicates the calculated force that must
			 * be ignored, as it corresponds to the force exerted by particle i onto itself.
			 * We mask out this force, by performing a bitwise AND operation between the contents
			 * of the register containing the calculated forces and the mask returned by _mm256_cmp_pd.
			 * 
			 * 		For example:
			 * 	
			 *    i    |    i    |    i    |    i 
			 *    j    |   j+1   |   j+2   |   j+3 
			 *
			 * 		compare not equal
			 * 	
			 *    0    |    1    |    1    |    1    -> AND mask
			 *
			 *    0    | Fi(j+1) | Fi(j+2) | Fi(j+3)
			 */

			// Helper register that contains the different j indexes
			__m256d _j = _mm256_set_pd(j+3, j+2, j+1, j); 
			__m256d _mask = _mm256_cmp_pd(_i, _j, _CMP_NEQ_OQ); // mask register

			// load the next 4 particles
			__m256d _x = _mm256_load_pd(&particles.x[j]); 			
			__m256d _y = _mm256_load_pd(&particles.y[j]);
			__m256d _z = _mm256_load_pd(&particles.z[j]);
			__m256d _mj = _mm256_load_pd(&particles.m[j]);

			// Calculate ri - rj
			__m256d _xdiff = _mm256_sub_pd(_xi, _x);
			__m256d _ydiff = _mm256_sub_pd(_yi, _y);
			__m256d _zdiff = _mm256_sub_pd(_zi, _z);
			
			// Calculate (ri - rj).^2
			__m256d _xpow2 = _mm256_mul_pd(_xdiff, _xdiff);
			__m256d _ypow2 = _mm256_mul_pd(_ydiff, _ydiff);
			__m256d _zpow2 = _mm256_mul_pd(_zdiff, _zdiff);

			/* Calculate ||ri - rj||^3 as:
			 * first compute : sqrt( (xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2 )^3, using :
			 * sqrt(_xpow2 + _ypow2 + _zpow2)^3, using :
			 * sqrt(_xpow2 + _ypow2 + _zpow2) * (_xpow2 + _ypow2 + _zpow2)
			 */
			__m256d _tmp = _mm256_add_pd(_xpow2, _ypow2);
			_tmp = _mm256_add_pd(_tmp, _zpow2);
			__m256d _tmp_sqrt = _mm256_sqrt_pd(_tmp);
			_tmp = _mm256_mul_pd(_tmp, _tmp_sqrt);

			// Calculate force's magnitude (Fij)
			__m256d _magnitude = _mm256_mul_pd(_mi, _mj);
			_magnitude = _mm256_mul_pd(_magnitude, _g_const);
			_magnitude = _mm256_div_pd(_magnitude, _tmp); 

			/* Perform the bitwise AND between the 4 computed forces and the mask
			 * returned by the earlier execution of the _mm256_cmp_pd() function.
			 * This will zero-out the magnitude of the force of particle i onto
			 * itself.
			 */
			_magnitude = _mm256_and_pd(_magnitude, _mask);

			/* Accumulate the forces (for each coordinate) to the AVX
			 * registers that contain the forces calculated by previous iterations
			 * of the j-indexed for loop 
			 */ 
			_xdiff = _mm256_mul_pd(_xdiff, _magnitude);
			_ydiff = _mm256_mul_pd(_ydiff, _magnitude);
			_zdiff = _mm256_mul_pd(_zdiff, _magnitude);

			_fxi = _mm256_add_pd(_fxi, _xdiff);
			_fyi = _mm256_add_pd(_fyi, _ydiff);
			_fzi = _mm256_add_pd(_fzi, _zdiff);
		}

		// Manual reduction of each AVX vector (one for every coordinate) to a single double value
		particles.fx[i] += ((double *)&_fxi)[0] + ((double *)&_fxi)[1] + ((double *)&_fxi)[2] + ((double *)&_fxi)[3];
		particles.fy[i] += ((double *)&_fyi)[0] + ((double *)&_fyi)[1] + ((double *)&_fyi)[2] + ((double *)&_fyi)[3];
		particles.fz[i] += ((double *)&_fzi)[0] + ((double *)&_fzi)[1] + ((double *)&_fzi)[2] + ((double *)&_fzi)[3];

		// Handle remaining entries
		for (int j = ndiv4 * 4; j < particles.n; j++)
		{
			if (i!=j) // "vanilla" code
			{
				double tmp = pow(particles.x[i]-particles.x[j], 2.0) +
							pow(particles.y[i]-particles.y[j], 2.0) +
							pow(particles.z[i]-particles.z[j], 2.0);

				double magnitude = G * particles.m[i] * particles.m[j] / pow(tmp, 1.5);

				particles.fx[i] += (particles.x[i]-particles.x[j]) * magnitude;
				particles.fy[i] += (particles.y[i]-particles.y[j]) * magnitude;
				particles.fz[i] += (particles.z[i]-particles.z[j]) * magnitude;
			}
		}
	}
}

int main()
{
	testAll(computeGravitationalForcesFast);
	return 0;
}
