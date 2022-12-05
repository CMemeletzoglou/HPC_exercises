#include <cmath>
#include <immintrin.h>
#include <omp.h>

#include "particles.h"
#include "utils.h"

// C
#include <stdio.h>
#include <cstring>

// static void print_avx_hex(const char * label, __m256d v)
// {
// 	std::cout << std::endl;
//     double a[4];
//     _mm256_storeu_pd((double *)a, v);
// 	for (int i = 0; i < 4; i++)
// 		std::cout << label << "[" << i << "] = " << std::hex << a[i] << std::endl;
// }

//  uint8_t a[16] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

//     __m128i v = _mm_loadu_si128((__m128i *)a);

//     printf("v = %#vx\n", v);
//     printf("v = %#vhx\n", v);
//     printf("v = %#vlx\n", v);

static void print_avx(const char * label, __m256d v)
{
	std::cout << std::endl;
	double a[4];
	// _mm256_storeu_pd((double *)a, v); // MEM[mem_addr+255:mem_addr] := b[255:0] 
		// a[0] = 3 a[1] = 2 a[2] = 1 a[3] = 0
	// std::cout << label << ":\t" << "\n";
	// for (int i = 3; i >= 0; i--)
	// 	std::cout << a[i] << '\t';

	memcpy(a, &v, sizeof(a));

	for(int i=3; i >= 0; i--)
		// printf("element %d = %f\n", abs((i - 3) % 4), a[i]);
		std::cout << label << "[" << abs((i - 3) % 4) << "] = " << a[i] << std::endl;


	std::cout << '\n';
}

/**
 * Compute the gravitational forces in the system of particles
 * Use AVX and OpenMP to speed up computation
**/
void computeGravitationalForcesFast(Particles& particles)
{
	const double G = 6.67408e-11;

	__m256d _g_const = _mm256_set1_pd(G);

	int ndiv4 = particles.n / 4;

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

		for (int j = 0; j < 4*ndiv4 ; j+=4)
		{
			/* 
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
			 */

			// Helper register that contains the different j indexes
			__m256d _j = _mm256_set_pd(j, j+1, j+2, j+3); // 0 1 2 3
			// print_avx("_j", _j);
			__m256d mask = _mm256_cmp_pd(_i, _j, _CMP_NEQ_OQ); // 0 -nan -nan -nan
			// print_avx("mask", mask);

			// load 4 particles dst[255:0] := MEM[mem_addr+255:mem_addr]
			// memory: x0 x1 x2 x3    x3 x2 x1 x0
			__m256d _x = _mm256_load_pd(&particles.x[j]); // x3 x2 x1 x0

			std::cout << "TETESDFAS " << ((double*)&_x)[3] << std::endl;
			printf("particles[%d] = %f\n", j, (double)particles.x[j]);
			printf("particles[%d] = %f\n", j+1, (double)particles.x[j+1]);
			printf("particles[%d] = %f\n", j+2, (double)particles.x[j+2]);
			printf("particles[%d] = %f\n", j+3, (double)particles.x[j+3]);

			print_avx("_x", _x);
			return;

			
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

			// Calculate ||ri - rj||^3
			// = sqrt((xi-xj)^2 + (yi-yj)^2 + (zi-zj)^2)^3
			// = sqrt(_xpow2 + _ypow2 + _zpow2)^3
			// = sqrt(_xpow2 + _ypow2 + _zpow2) * (_xpow2 + _ypow2 + _zpow2)
			__m256d _tmp = _mm256_add_pd(_xpow2, _ypow2);
			_tmp = _mm256_add_pd(_tmp, _zpow2);
			__m256d _tmp_sqrt = _mm256_sqrt_pd(_tmp);
			_tmp = _mm256_mul_pd(_tmp, _tmp_sqrt);

			// print_avx("_tmp", _tmp);

			// Calculate force's magnitude (Fij)
			__m256d _magnitude = _mm256_mul_pd(_mi, _mj);
			_magnitude = _mm256_mul_pd(_magnitude, _g_const);
			_magnitude = _mm256_div_pd(_magnitude, _tmp); // zero

			// print_avx("_magnitude", _magnitude);
			
			// Bitwise and the results with the mask so you make 0 the doubles
			// inside the AVX register that have i == j
			_magnitude = _mm256_and_pd(_magnitude, mask);

			

			// Accumulate the forces (for each coordinate) to the AVX
			// registers that contain the forces calculated by previous iterations of th j-indexed for loop 
			_xdiff = _mm256_mul_pd(_xdiff, _magnitude);
			_ydiff = _mm256_mul_pd(_ydiff, _magnitude);
			_zdiff = _mm256_mul_pd(_zdiff, _magnitude);

			_fxi = _mm256_add_pd(_fxi, _xdiff);
			_fyi = _mm256_add_pd(_fyi, _ydiff);
			_fzi = _mm256_add_pd(_fzi, _zdiff);
		}

		for (int i = ndiv4 * 4; i < particles.n; i++)
		{
			// TODO
		}

		// TODO: Handle remaining entries

		// Manual reduction of each AVX vector (one for every coordinate) to a single double value
		particles.fx[i] += ((double *)&_fxi)[0] + ((double *)&_fxi)[1] + ((double *)&_fxi)[2] + ((double *)&_fxi)[3];
		particles.fy[i] += ((double *)&_fyi)[0] + ((double *)&_fyi)[1] + ((double *)&_fyi)[2] + ((double *)&_fyi)[3];
		particles.fz[i] += ((double *)&_fzi)[0] + ((double *)&_fzi)[1] + ((double *)&_fzi)[2] + ((double *)&_fzi)[3];
	}

	
	
}

int main()
{
	testAll(computeGravitationalForcesFast);
	return 0;
}
