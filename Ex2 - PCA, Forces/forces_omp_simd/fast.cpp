#include <cmath>
#include <immintrin.h>
#include <omp.h>

#include "particles.h"
#include "utils.h"

/**
 * Compute the gravitational forces in the system of particles
 * Use AVX and OpenMP to speed up computation
**/
void computeGravitationalForcesFast(Particles& particles)
{
	const double G = 6.67408e-11;

	__m256d _g_const = _mm256_set1_pd(G);

	for (int i=0; i<particles.n; i++)
	{
		// load current particle
		__m256d _xi = _mm256_set1_pd(particles.x[i]); // pi_x|pi_x|pi_x|pi_x
		__m256d _yi = _mm256_set1_pd(particles.y[i]);
		__m256d _zi = _mm256_set1_pd(particles.z[i]);
		__m256d _mi = _mm256_set1_pd(particles.m[i]);

		// initialize the AVX registers to store the sum of forces foreach dimension
		__m256d _fxi = _mm256_setzero_pd();
		__m256d _fyi = _mm256_setzero_pd();
		__m256d _fzi = _mm256_setzero_pd();

		// Create a helper register that contains the index i
		__m256d _i = _mm256_set1_pd(i);

		for (int j=0; j<particles.n; j+=4)
		{
			/** 
			 * The idea of how to use a mask to zero out the forces that are calculated when i == j.
			 * There is no way to avoid calculating them, since vectorization loads the values of
			 * consecutive memory addresses to the registers.
			 * There is also no other way to find if i is in [j,j+3], since we also want to know the
			 * which double inside the AVX register has i == j.
			 *    i    |    i    |    i    |    i 
			 *    j    |   j+1   |   j+2   |   j+3 
			 * compare not equal
			 *    1    |    0    |    1    |    1    -> mask
			 * multiply(mask, force)
			 *   Fij   |    0    | Fi(j+2) | Fi(j+3)
			 * alternatively: Instead of multiply do bitwise and
			**/

			// Helper register that contains the different j indexes
			__m256d _j = _mm256_set_pd(j, j+1, j+2, j+3);
			__m256d mask = _mm256_cmp_pd(_i, _j, _CMP_NEQ_OQ);

			// load 4 particles
			__m256d _x = _mm256_load_pd(&particles.x[j]);
			__m256d _y = _mm256_load_pd(&particles.y[j]);
			__m256d _z = _mm256_load_pd(&particles.z[j]);
			__m256d _mj = _mm256_set_pd(particles.m[j], particles.m[j+1], particles.m[j+2], particles.m[j+3]);

			// Calculate ri - rj
			__m256d _xdiff = _mm256_sub_pd(_xi, _x);
			__m256d _ydiff = _mm256_sub_pd(_yi, _y);
			__m256d _zdiff = _mm256_sub_pd(_zi, _z);
			
			// Calculate (ri - rj).^2
			__m256d _xpow2 = _mm256_mul_pd(_xdiff, _xdiff);
			__m256d _ypow2 = _mm256_mul_pd(_ydiff, _ydiff);
			__m256d _zpow2 = _mm256_mul_pd(_zdiff, _zdiff);

			// Calculate ||ri - rj||^3
			// = sqrt((xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2)^3
			// = sqrt(_xpow2 + _ypow2 + _zpow2)^3
			// = sqrt(_xpow2 + _ypow2 + _zpow2) * (_xpow2 + _ypow2 + _zpow2)
			__m256d _tmp = _mm256_add_pd(_xpow2, _ypow2);
			_tmp = _mm256_add_pd(_tmp, _zpow2);
			__m256d _tmp_sqrt = _mm256_sqrt_pd(_tmp);
			_tmp = _mm256_mul_pd(_tmp, _tmp_sqrt);

			// Calculate force's magnitude (Fij)
			__m256d _magnitude = _mm256_mul_pd(_g_const, _mi);
			_magnitude = _mm256_mul_pd(_magnitude, _mj);
			_magnitude = _mm256_div_pd(_magnitude, _tmp);

			// Accumulate the forces (for each coordinate) to the AVX
			// registers that contain the forces calculated by previous iterations of th j-indexed for loop 
			_xdiff = _mm256_mul_pd(_xdiff, _magnitude);
			_ydiff = _mm256_mul_pd(_ydiff, _magnitude);
			_zdiff = _mm256_mul_pd(_zdiff, _magnitude);

			// Multiply the results with the mask so you make 0 the doubles
			// inside the AVX register that have i == j
			// _xdiff = _mm256_mul_pd(_xdiff, mask);
			// _ydiff = _mm256_mul_pd(_ydiff, mask);
			// _zdiff = _mm256_mul_pd(_zdiff, mask);

			// Bitwise and the results with the mask so you make 0 the doubles
			// inside the AVX register that have i == j
			_xdiff = _mm256_and_pd(_xdiff, mask);
			_ydiff = _mm256_and_pd(_ydiff, mask);
			_zdiff = _mm256_and_pd(_zdiff, mask);

			_fxi = _mm256_add_pd(_fxi, _xdiff);
			_fyi = _mm256_add_pd(_fyi, _ydiff);
			_fzi = _mm256_add_pd(_fzi, _zdiff);
		}

		// __m256d _tmp_fi = _mm256_hadd_pd(_fxi, _fxi); // fxi3+fxi2 fxi3+fxi2 fxi1+fxi0 fxi1+fxi0
		// particles.fx[i] += ((double *)&_tmp_fi)[0] + ((double *)&_tmp_fi)[2];
		// _tmp_fi = _mm256_hadd_pd(_fyi, _fyi); // fxi3+fxi2 fxi3+fxi2 fxi1+fxi0 fxi1+fxi0
		// particles.fy[i] += ((double *)&_tmp_fi)[0] + ((double *)&_tmp_fi)[2];
		// _tmp_fi = _mm256_hadd_pd(_fzi, _fzi); // fxi3+fxi2 fxi3+fxi2 fxi1+fxi0 fxi1+fxi0
		// particles.fz[i] += ((double *)&_tmp_fi)[0] + ((double *)&_tmp_fi)[2];
		
		// Manual reduction of each AVX vector (one for every coordinate) to a single double value
		particles.fx[i] += ((double *)&_fxi)[0] + ((double *)&_fxi)[1] + ((double *)&_fxi)[2] + ((double *)&_fxi)[3];
		particles.fy[i] += ((double *)&_fyi)[0] + ((double *)&_fyi)[1] + ((double *)&_fyi)[2] + ((double *)&_fyi)[3];
		particles.fz[i] += ((double *)&_fzi)[0] + ((double *)&_fzi)[1] + ((double *)&_fzi)[2] + ((double *)&_fzi)[3];
	}
	
	//TODO: Handle remaining entries
}

int main()
{
	testAll(computeGravitationalForcesFast);
	return 0;
}
