#include <emmintrin.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>
// #include <mpi.h>

/* I/O routines */
void store_binary_data(char *filename, double *data, int n)
{
	FILE *fp;
	fp = fopen(filename, "wb");
	if (fp == NULL)
	{
		printf("fopen(%s, \"wb\") FAILED!\n", filename);
		exit(1);
	}
	size_t nelems = fwrite(data, sizeof(double), n, fp);
	assert(nelems == n); // check that all elements were actually written
	fclose(fp);
}

void load_binary_data(const char *filename, double *data, const int n)
{
	FILE *fp;
	fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		printf("fopen(%s, \"wb\") FAILED!\n", filename);
		exit(1);
	}
	size_t nelems = fread(data, sizeof(double), n, fp);
	assert(nelems == n); // check that all elements were actually read
	fclose(fp);
}

void copy_to_aligned(double *mem, double **aligned_data, const int mem_row_size, const int aligned_data_row_size, const int num_of_rows)
{
	for (int i = 0; i < num_of_rows; i++)
		for (int j = 0; j < aligned_data_row_size; j++)
			aligned_data[i][j] = mem[i*mem_row_size + j];
}

#if defined(MPI)

void load_binary_data_mpi(const char *filename, double *data, const int qelems, const int probdim, const int N)
{
	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// Open the file (collective call)
	MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

        // Calculate the offset for each rank
        MPI_Offset base;
        MPI_File_get_position(f, &base);

	MPI_Offset data_len = N * sizeof(double);
	MPI_Offset offset = rank * data_len; // rank offset

	int num_elems = N;
	if (rank == nprocs - 1)
		num_elems += (qelems % nprocs) * (probdim + 1);

	// Read the data
        MPI_Status status;
        MPI_File_read_at_all(f, base + offset, data, num_elems, MPI_DOUBLE, &status); // blocking collective call

        // Close the file
        MPI_File_close(&f);        
}

#endif

double read_nextnum(FILE *fp)
{
	double val;

	int c = fscanf(fp, "%lf", &val);
	if (c <= 0) {
		fprintf(stderr, "fscanf returned %d\n", c);
		exit(1);
	}
	return val;
}

/* Timer */
double gettime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double) (tv.tv_sec+tv.tv_usec/1000000.0);
}

/* Function to approximate */
double fitfun(double *x, int n)
{
	double f = 0.0;
	int i;

#if 1
	for(i=0; i<n; i++)	/* circle */
		f += x[i]*x[i];
#endif
#if 0
	for(i=0; i<n-1; i++) {	/*  himmelblau */
		f = f + pow((x[i]*x[i]+x[i+1]-11.0),2) + pow((x[i]+x[i+1]*x[i+1]-7.0),2);
	}
#endif
#if 0
	for (i=0; i<n-1; i++)   /* rosenbrock */
		f = f + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
#endif
#if 0
	for (i=0; i<n; i++)     /* rastrigin */
		f = f + pow(x[i],2) + 10.0 - 10.0*cos(2*M_PI*x[i]);
#endif

	return f;
}


/* random number generator  */
#define SEED_RAND()     srand48(1)
#define URAND()         drand48()

#ifndef LB
#define LB -1.0
#endif
#ifndef UB
#define UB 1.0
#endif

double get_rand(int k)
{
	return (UB-LB)*URAND()+LB;
}


/* utils */
double compute_min(double *v, int n)
{
	int i;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) vmin = v[i];

	return vmin;
}

double compute_max(double *v, int n)
{
	int i;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) vmax = v[i];

	return vmax;
}

double compute_sum(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s;
}

double compute_sum_pow(double *v, int n, int p)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i], p);

	return s;
}

double compute_mean(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

double compute_std(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return sqrt(s/(n-1));
}

double compute_var(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

double compute_dist(double *v, double *w, int n)
{
#if defined (SIMD)

	__builtin_assume_aligned(v, 32);
	__builtin_assume_aligned(w, 32);

#if defined(DEBUG)
	// check for 32-Byte alignment
	assert( (((size_t)v & 0x1F) == 0) && (((size_t)w & 0x1F) == 0) );
#endif
	double sum_values;

	int ndiv4 = n / 4;
	__m256d _sum_v = _mm256_setzero_pd();
	__m256d _v, _w, _diff;
	for (int i = 0; i < ndiv4; i++)
	{
		_v = _mm256_load_pd(&v[i*4]);
		_w = _mm256_load_pd(&w[i*4]);
		_diff = _mm256_sub_pd(_v, _w);
		_diff = _mm256_mul_pd(_diff, _diff); // diff squared
		_sum_v = _mm256_add_pd(_sum_v, _diff); // add to sum vector reg
	}

	// reduce the vector's 4 elements into one, using two SSE horizontal additions
	// __m128d _total_sum = _mm_hadd_pd(_mm256_extractf128_pd(_sum_v, 0), _mm256_extractf128_pd(_sum_v, 1));
	// _total_sum = _mm_hadd_pd(_total_sum, _total_sum);
	// _mm_store_pd(sum_values, _total_sum);
	// _mm_storeh_pd(&(sum_values), _total_sum);

	__m256d _hsum = _mm256_hadd_pd(_sum_v, _sum_v);
	__m128d _total_sum = _mm_add_pd(_mm256_extractf128_pd(_hsum, 1), _mm256_castpd256_pd128(_hsum));
	_mm_storeh_pd(&(sum_values), _total_sum);

	// handle the remaining entries
	double sum = 0.0f;
	for (int i = ndiv4 * 4; i < n; i++)
		sum += pow(v[i] - w[i], 2);

	sum_values += sum;

	return sqrt(sum_values);
#else
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s+= pow(v[i]-w[i],2);
	}

	return sqrt(s);
#endif
}

double compute_max_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}

double compute_min_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) {
			vmin = v[i];
			p = i;
		}

	*pos = p;
	return vmin;
}

double compute_root(double dist, int norm)
{
	if (dist == 0) return 0;

	switch (norm) {
	case 2:
		return sqrt(dist);
	case 1:
	case 0:
		return dist;
	default:
		return pow(dist, 1 / (double) norm);
	}
}

double compute_distance(double *pat1, double *pat2, int lpat, int norm)
{
	register int i;
	double dist = 0.0;

	for (i = 0; i < lpat; i++) {
		double diff = 0.0;

		diff = pat1[i] - pat2[i];

		switch (norm) {
		double   adiff;

		case 2:
			dist += diff * diff;
			break;
		case 1:
			dist += fabs(diff);
			break;
		case 0:
			if ((adiff = fabs(diff)) > dist)
			dist = adiff;
			break;
		default:
			dist += pow(fabs(diff), (double) norm);
			break;
		}
	}

	return dist;	// compute_root(dist);
}

// npat -> TRAINELEMS
// lpat -> PROBDIM
void compute_knn_brute_force(double **xdata, double *q, int npat, int lpat, int knn, int *nn_x, double *nn_d)
{
	int i, max_i;
	double max_d, new_d;

	/* initialize pairs of index and distance */
	for (i = 0; i < knn; i++)
	{
		nn_x[i] = -1;
		nn_d[i] = 1e99 - i; // TODO: is "-i" needed (???), prob not
	}

	max_d = compute_max_pos(nn_d, knn, &max_i);
	for (i = 0; i < npat; i++)
	{
		new_d = compute_dist(q, xdata[i], lpat);	// euclidean
		if (new_d < max_d) // add point to the list of knns, replace element max_i
		{	
			nn_x[max_i] = i;
			nn_d[max_i] = new_d;
		}
		max_d = compute_max_pos(nn_d, knn, &max_i);
	}

	/* sort the knn list */ // bubble sort
	int temp_x, j;
	double temp_d;
	for (i = (knn - 1); i > 0; i--)
	{
		for (j = 1; j <= i; j++)
		{
			if (nn_d[j-1] > nn_d[j])
			{
				temp_d = nn_d[j-1];
				nn_d[j-1] = nn_d[j];
				nn_d[j] = temp_d;
				
				temp_x = nn_x[j-1];
				nn_x[j-1] = nn_x[j];
				nn_x[j] = temp_x;
			}
		}
	}
}


/* compute an approximation based on the values of the neighbors */
double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
{
#if defined (SIMD)
	// plain mean (other possible options: inverse distance weight, closest value inheritance)

	// assume 32-byte alignment of ydata vector
	__builtin_assume_aligned(ydata, 32);

#if defined(DEBUG)
	// check for 32-Byte alignment
	assert( (((size_t)ydata & 0x1F) == 0) );
#endif

	double sum_values;

	__m256d _sum_v = _mm256_setzero_pd(); // zero-out the sum vector reg
	__m256d _ydata;
	int knn_div4 = knn / 4;

	for (int i = 0; i < knn_div4; i++)
	{
		_ydata = _mm256_load_pd(ydata + 4 * i); // load groups of 4 elements of the vector 
		_sum_v = _mm256_add_pd(_sum_v, _ydata); // vertically add them into the sum vector reg
	}
	/* After the for 4-stride for loop finished, the _sum_v vector register contains:
	 * 			s0 || s1 || s2 || s3
	 * These values need to be added together (reduced) to produce the final vector sum value.
	 * Use _mm256_extractf128_pd(_hsum, 0) to extract the lower 128-bit part s0 || s1 and
	 * _mm256_extractf128_pd(_hsum, 1) to extract the upper 128-bit part s2 || s3.
	 * Then use two SSE horizontal additions to execute the final reduction
	 */

	// sum the 4 elements of the sum vector to get the final sum value

	// ***************************************
	// 	AVX instructions-only way
	// ***************************************
	__m256d _hsum = _mm256_hadd_pd(_sum_v, _sum_v);
	__m128d _total_sum = _mm_add_pd(_mm256_extractf128_pd(_hsum, 1), _mm256_castpd256_pd128(_hsum));
	_mm_storeh_pd(&(sum_values), _total_sum);

	// __m128d _total_sum = _mm_hadd_pd(_mm256_extractf128_pd(_sum_v, 0), _mm256_extractf128_pd(_sum_v, 1));
	// _total_sum = _mm_hadd_pd(_total_sum, _total_sum);
	// _mm_storeh_pd(&(sum_values), _total_sum);

	// handle the remaining entries
	double sum = 0.0f;
	for (int i = knn_div4 * 4; i < knn; i++)
		sum += ydata[i];

	sum_values+= sum;

	return sum_values / knn;
#else
	int i;
	double sum_v = 0.0;
	for (i = 0; i < knn; i++)
		sum_v += ydata[i];

	return sum_v / knn;
#endif
}
