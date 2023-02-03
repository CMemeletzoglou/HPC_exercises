#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

static double **xdata;
static double ydata[TRAINELEMS];

double find_knn_value(double *p, int n, int knn)
{
	int nn_x[knn];
	double nn_d[knn];

	compute_knn_brute_force(xdata, p, TRAINELEMS, PROBDIM, knn, nn_x, nn_d); // brute-force / linear search

	double xd[knn * PROBDIM];     // the knn neighboring points/vectors of size PROBDIM
	__attribute__((aligned(32))) double fd[knn];	      	      // function values for the knn neighbors

	for (int i = 0; i < knn; i++)
		fd[i] = ydata[nn_x[i]];

	for (int i = 0; i < knn; i++) 
		for (int j = 0; j < PROBDIM; j++)
			xd[i * PROBDIM + j] = xdata[nn_x[i]][j];

	return predict_value(PROBDIM, knn, xd, fd, p, nn_d);
}

int main(int argc, char *argv[])
{
	/* Load all data from files in memory */
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *mem = (double *)malloc(TRAINELEMS * (PROBDIM + 1) * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * (PROBDIM + 1) * sizeof(double));	
	
	load_binary_data(trainfile, mem, TRAINELEMS*(PROBDIM+1));
	load_binary_data(queryfile, query_mem, QUERYELEMS * (PROBDIM + 1));

#if defined(DEBUG)
	/* Create/Open an output file */
	FILE *fpout = fopen("output.knn.txt","w");
#endif

	// Create handler arrays that will separate xdata and surrogate
	// since we are never going to be using both to perform a computation
	// We either going to use the xdata of two points (ex. when calculating distance from one another)
	// or use ydata (surrogates) (ex. when predicting the value of a query point)
	xdata = (double **)malloc(TRAINELEMS * sizeof(double *));
	double **query_xdata = (double **)malloc(QUERYELEMS * sizeof(double *));

#if defined(SIMD)
	int posix_res;
	// Allocate new memory for the handler arrays, so it is aligned and copy the data there
	// Align each xdata[i] to a 32 byte boundary so you may later use SIMD
	for (int i = 0; i < TRAINELEMS; i++)
	{
		posix_res = posix_memalign((void **)(&(xdata[i])), 32, PROBDIM*sizeof(double));
		assert(posix_res == 0);
	}
	align_data(mem, xdata, (PROBDIM+1), PROBDIM, TRAINELEMS);

	// Align each query_xdata[i] to a 32 byte boundary so you may later use SIMD
	for (int i = 0; i < QUERYELEMS; i++)
	{
		posix_res = posix_memalign((void **)(&(query_xdata[i])), 32, PROBDIM*sizeof(double));
		assert(posix_res == 0);
	}
	align_data(query_mem, query_xdata, (PROBDIM+1), PROBDIM, QUERYELEMS);
#else
	// Assign to the handler arrays, pointers to the already allocated mem
	for (int i = 0; i < TRAINELEMS; i++)
		xdata[i] = &mem[i*(PROBDIM + 1)];

	for (int i = 0; i < QUERYELEMS; i++)
		query_xdata[i] = &query_mem[i*(PROBDIM + 1)];
#endif

	/* Configure and Initialize the ydata handler arrays */
	// TODO: (keep the code uniform) Either make query_ydata a global static array or dynamically allocate ydata array here aswell
	double *query_ydata = malloc(QUERYELEMS * sizeof(double));
	
	for (int i = 0; i < TRAINELEMS; i++)
	{
#if defined(SURROGATES)
		ydata[i] = mem[i * (PROBDIM + 1) + PROBDIM];
#else
		ydata[i] = 0;
#endif
	}

	for (int i = 0; i < QUERYELEMS; i++)
	{
#if defined(SURROGATES)
		query_ydata[i] = query_mem[i * (PROBDIM + 1) + PROBDIM];
#else
		query_ydata[i] = 0;
#endif
	}

	/* COMPUTATION PART */

	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;
	
	for (int i = 0; i < QUERYELEMS; i++)
	{	/* requests */
		t0 = gettime();
		double yp = find_knn_value(query_xdata[i], PROBDIM, NNBS);
		t1 = gettime();
		t_sum += (t1-t0);
		if (i == 0)
			t_first = (t1-t0);

		sse += (query_ydata[i] - yp) * (query_ydata[i] - yp);

#if defined(DEBUG)
		for (int k = 0; k < PROBDIM; k++)
			fprintf(fpout, "%.5f ", query_mem[i * (PROBDIM + 1) + k]);
#endif

		err = 100.0 * fabs((yp - query_ydata[i]) / query_ydata[i]);

#if defined(DEBUG)
		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
#endif
		err_sum += err;
	}

	/* CALCULATE AND DISPLAY RESULTS */

	double mse = sse / QUERYELEMS;
	double ymean = compute_mean(query_ydata, QUERYELEMS);
	double var = compute_var(query_ydata, QUERYELEMS, ymean);
	double r2 = 1 - (mse / var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	printf("Total time = %lf secs\n", t_sum);
	printf("Time for 1st query = %lf secs\n", t_first);
	printf("Time for 2..N queries = %lf secs\n", t_sum - t_first);
	printf("Average time/query = %lf secs\n", (t_sum - t_first) / (QUERYELEMS - 1));

	/* CLEANUP */

#if defined(DEBUG)
	/* Close the output file */
	fclose(fpout);
#endif

#if defined(SIMD)
	for (int i = 0; i < TRAINELEMS; i++)
		free(xdata[i]);
#endif
	free(xdata);
	free(mem);
#if defined(SIMD)
	for (int i = 0; i < QUERYELEMS; i++)
		free(query_xdata[i]);
#endif
	free(query_ydata);
	free(query_mem);

	return 0;
}
