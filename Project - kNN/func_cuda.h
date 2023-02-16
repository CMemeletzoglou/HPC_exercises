#pragma once

#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

// struct that will preserve the k nearest neighbors for each query.
// Will be used in order to load training data in a blocking fashion,
// iterating over all query points. Thus for each query point we will
// need to preserve the k nearest neighbors that have been found so far
// in the preceding blocks.
typedef struct query_s
{
	double x[PROBDIM];
	int nn_idx[NNBS]; // The index (< TRAINELEMS) of the k nearest neighbors
	double nn_dist[NNBS]; // The distance between the query point and each one of the k nearest neighbors
	// double nn_val[NNBS];
} query_t;

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

void load_binary_data(const char *filename, double *data, query_t *queries, const int n)
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

	// If queries are loaded, initialize the queries structs
	if (queries != NULL)
	{
		for (int i = 0; i < QUERYELEMS; i++)
		{
			for (int k = 0; k < PROBDIM; k++)
				queries[i].x[k] = data[i * (PROBDIM + 1) + k];

			for (int j = 0; j < NNBS; j++)
				queries[i].nn_idx[j] = -1;

			for (int j = 0; j < NNBS; j++)
				queries[i].nn_dist[j] = 1e99 - j;

			// for (int j = 0; j < NNBS; j++)
			// 	queries[i].nn_val[j] = -1;
		}
	}
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
double compute_mean(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

double compute_var(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

__device__ double compute_dist(double *v, double *w, int n)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s+= pow(v[i]-w[i],2);
	}

	return sqrt(s);
}

__device__ double compute_max_pos(double *v, int n, int *pos)
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

void extract_vectors(double *input_arr, double *output_arr, int nrows, int input_dim, int output_dim)
{
	for(int i = 0; i < nrows; i++)
		for(int j = 0; j < output_dim; j++)
			output_arr[i * output_dim + j] = input_arr[i * input_dim + j];
}

// __device__ void compute_knn_brute_force_cuda(double *xdata, double *ydata, query_t *q, int dim, int k, int train_block_idx, int train_block_size)
__device__ void compute_knn_brute_force_cuda(double *xdata, double *ydata, double *q, int dim, int k, int train_block_idx, int train_block_size)
{
	int i, gi, xdata_idx, max_i;
	double max_d, new_d;
	
	// thread block shared memory arrays, for the k neighbor indexes and distances, that each thread computes
	__shared__ int nn_idx[k * blockDim.y];
	__shared__ double nn_dist[k * blockDim.y];
	
	int train_block_start = train_block_idx * train_block_size;

	// find K neighbors
	// max_d = compute_max_pos(q->nn_dist, k, &max_i);
	max_d = 1e99;
	for (i = 0; i < train_block_size; i++) // i runs inside each training block's boundaries
	{
		xdata_idx = train_block_start + i;
		
		new_d = compute_dist(q, &(xdata[xdata_idx * dim]), dim); // euclidean
		if (new_d < max_d) // add point to the list of knns, replace element max_i
		{	
			nn_idx[threadIdx.y * k + 0]
			// q->nn_idx[max_i] = xdata_idx;
			// q->nn_dist[max_i] = new_d;
			// q->nn_val[max_i] = ydata[xdata_idx];
		}
		max_d = compute_max_pos(q->nn_dist, k, &max_i);
	}

	// column-wise reduction, i.e. for each query point
	
	
}

/* compute an approximation based on the values of the neighbors */
// double predict_value(int dim, int knn, double *xdata, double *ydata, double *point, double *dist)
__device__ double predict_value(query_t* q, double *ydata, int dim, int knn)
{
	int i;
	double sum_v = 0.0;
	for (i = 0; i < knn; i++)
		sum_v += ydata[q->nn_idx[i]];

	return sum_v / knn;
}
