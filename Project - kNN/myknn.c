#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

static double **xdata;
static double *ydata;

int main(int argc, char *argv[])
{
	/* Load all data from files in memory */
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	int L1d_size, train_block_size = 1;
	get_L1d_size(&L1d_size); // get L1d cache size
	// calculate the appropriate train block size as the previous power of 2
	if(L1d_size > 0)
		train_block_size = pow(2, floor(log2((L1d_size * 1000) / (PROBDIM * sizeof(double)))));

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *mem = (double *)malloc(TRAINELEMS * (PROBDIM + 1) * sizeof(double));
	ydata = (double *)malloc(TRAINELEMS * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * (PROBDIM + 1) * sizeof(double));
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));

#if defined(SIMD)
	int posix_res;
        // Malloc aligned space for query.x data 
        for (int i = 0; i < QUERYELEMS; i++)
        {
                posix_res = posix_memalign((void **)(&(queries[i].x)), 32, PROBDIM * sizeof(double));
                assert(posix_res == 0);
        }
#endif

	load_binary_data(trainfile, mem, NULL, TRAINELEMS*(PROBDIM+1));
	load_binary_data(queryfile, query_mem, queries, QUERYELEMS * (PROBDIM + 1));

#if defined(DEBUG)
	/* Create/Open an output file */
	FILE *fpout = fopen("output.knn.txt","w");
#endif

	/* Create handler arrays that will be used to separate xdata's PROBDIM vectors
	 * and the corresponding surrogate values, since we never need both
	 * in order to perform a computation.
	 * We either going to use the xdata of two points (ex. when calculating distance from one another)
	 * or use ydata (surrogates) (ex. when predicting the value of a query point)
	 */
	xdata = (double **)malloc(TRAINELEMS * sizeof(double *));

#if defined(SIMD)
	// Allocate new memory for the handler arrays, so that it is aligned and copy the data there
	// Align each xdata[i] to a 32 byte boundary so you may later use SIMD
	for (int i = 0; i < TRAINELEMS; i++)
	{
		posix_res = posix_memalign((void **)(&(xdata[i])), 32, PROBDIM * sizeof(double));
		assert(posix_res == 0);
	}
	copy_to_aligned(mem, xdata, (PROBDIM+1), PROBDIM, TRAINELEMS);
#else
	// Assign to the handler arrays, pointers to the already allocated mem
	for (int i = 0; i < TRAINELEMS; i++)
		xdata[i] = &mem[i*(PROBDIM + 1)];
#endif

	/* Configure and Initialize the ydata handler arrays */
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

	assert(TRAINELEMS % train_block_size == 0);

	/* COMPUTATION PART */

	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	/* For each training elements block, we calculate each query point's k neighbors,
	 * using the training elements, that belong to the current training element block.
	 * The calculation of each query point's neighbors, occurs inside compute_knn_brute_force.
	 */
	for (int train_offset = 0; train_offset < TRAINELEMS; train_offset += train_block_size)
	{
		t0 = gettime();
		for (int i = 0; i < QUERYELEMS; i++)
		{
			compute_knn_brute_force(xdata, ydata, &(queries[i]), PROBDIM, NNBS, train_offset, 0, train_block_size);
			if (i == 0)
				t_first += gettime() - t0;
		}
		t1 = gettime();
		t_sum += t1 - t0;
	}


	for (int i = 0; i < QUERYELEMS; i++)
	{
		t0 = gettime();
		double yp = predict_value(queries[i].nn_val, NNBS);
		t1 = gettime();
		t_sum += t1 - t0;
		if (i == 0)
			t_first += t1 - t0;
		
		sse += (query_ydata[i] - yp) * (query_ydata[i] - yp);
		err = 100.0 * fabs((yp - query_ydata[i]) / query_ydata[i]);
#if defined(DEBUG)
		fprintf(fpout,"%.5f %.5f %.2f\n", query_ydata[i], yp, err);
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
	for (int i = 0; i < QUERYELEMS; i++)
		free(queries[i].x);
#endif
	free(queries);
	free(query_ydata);
	free(query_mem);

#if defined(SIMD)
	for (int i = 0; i < TRAINELEMS; i++)
		free(xdata[i]);
#endif
	free(xdata);
	free(ydata);
	free(mem);

	return 0;
}
