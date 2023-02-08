#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

#define DEBUG 1

// My L1d = 192 KiB, and I want to cache the maximum amount of training points.
// Each training point has a size of: PROBDIM * sizeof(double) = 16 * 8 = 128 bytes.
// Thus, in L1d I may preserve in L1d cache 192,000 / 128 = 1,500 training points simultaneously.
// I also need to be able to store in cache the query point aswell (!!)
// and have a block size that will evenly devide TRAINELEMS.
// The easy solution is to get the max power of 2 that is less that 1,500,
// since TRAINELEMS is also a power of 2.
#define TRAIN_BLOCK_SIZE 4096
// #define TRAIN_BLOCK_SIZE 128

static double **xdata;
static double ydata[TRAINELEMS];

// #define MAX_NNB	256

double find_knn_value(query_t *q, int knn)
{
	double xd[knn * PROBDIM];     // the knn neighboring points/vectors of size PROBDIM
	double fd[knn];	      	      // function values for the knn neighbors

	for (int i = 0; i < knn; i++)
		fd[i] = ydata[q->nn_idx[i]];

	for (int i = 0; i < knn; i++) 
		for (int j = 0; j < PROBDIM; j++)
			xd[i * PROBDIM + j] = xdata[q->nn_idx[i]][j];

	return predict_value(PROBDIM, knn, xd, fd);
}

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *mem = (double *)malloc(TRAINELEMS * (PROBDIM + 1) * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * (PROBDIM + 1) * sizeof(double));
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));
	
	xdata = (double **)malloc(TRAINELEMS * sizeof(double *));
	
	load_binary_data(trainfile, mem, NULL, TRAINELEMS*(PROBDIM+1));

	for (int i = 0; i < TRAINELEMS; i++)
		xdata[i] = mem + i * (PROBDIM + 1); //&mem[i*PROBDIM];

	for (int i = 0; i < TRAINELEMS; i++)
	{
#if defined(SURROGATES)
		ydata[i] = mem[i * (PROBDIM + 1) + PROBDIM];
#else
		ydata[i] = 0;
#endif
	}

	load_binary_data(queryfile, query_mem, queries, QUERYELEMS * (PROBDIM + 1));

#if DEBUG
	FILE *fpout = fopen("output.knn.txt","w");
#endif
	double *y = malloc(QUERYELEMS * sizeof(double));

	double t0, t1, t2, t3, t_sum;
	double sse = 0.0;
	double err, err_sum = 0.0;
	
	assert(TRAINELEMS % TRAIN_BLOCK_SIZE == 0);

	// initialize the query element surrogate values array
	for (int i = 0; i < QUERYELEMS; i++)
	{	
#if defined(SURROGATES)
		y[i] = query_mem[i * (PROBDIM + 1) + PROBDIM];
#else
		y[i] = 0.0;
#endif
	}

	/* For each training elements block, we calculate each query point's k neighbors,
	 * using the training elements, that belong to the current training element block.
	 * The calculation of each query point's neighbors, occurs inside compute_knn_brute_force.
	 */
	t0 = gettime();
	for (int train_offset = 0; train_offset < TRAINELEMS; train_offset += TRAIN_BLOCK_SIZE)
		for (int i = 0; i < QUERYELEMS; i++)
			compute_knn_brute_force(xdata, &(queries[i]), PROBDIM, NNBS, train_offset, TRAIN_BLOCK_SIZE);

	t1 = gettime();
	t_sum = t1 - t0;

	for (int i = 0; i < QUERYELEMS; i++)
	{
		t2 = gettime();
		double yp = find_knn_value(&(queries[i]), NNBS);
		t3 = gettime();
		t_sum += t3 - t2;
		
		sse += (y[i] - yp) * (y[i] - yp);
		err = 100.0 * fabs((yp - y[i]) / y[i]);

#if DEBUG
		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
#endif
		err_sum += err;
	}

	double mse = sse / QUERYELEMS;
	double ymean = compute_mean(y, QUERYELEMS);
	double var = compute_var(y, QUERYELEMS, ymean);
	double r2 = 1 - (mse / var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	printf("Total time = %lf sec\n", t_sum);
	printf("Average time/query = %lf sec\n", t_sum / QUERYELEMS);

	free(mem);
	free(xdata);
	free(query_mem);
	free(queries);
	free(y);

	return 0;
}
