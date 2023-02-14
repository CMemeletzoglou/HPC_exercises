#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

__device__ double find_knn_value(double *dev_mem, double *dev_ydata, query_t *q, int *knn)
{
	// double fd[knn];	      	      // function values for the knn neighbors

	// compute_knn_brute_force(xdata, p, TRAINELEMS, PROBDIM, knn, nn_x, nn_d); // brute-force / linear search

	// for (int i = 0; i < knn; i++)
	// 	fd[i] = ydata[nn_x[i]];

	double sum_v = 0.0;
	for (int i = 0; i < *knn; i++)
		sum_v += dev_ydata[i];

	return sum_v / *knn;
}

__global__ void knn_kernel(double *dev_queries)
{
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

        for (int i = 0; i < QUERYELEMS; i++)
	{	/* requests */
		// t0 = gettime();
		// double yp = find_knn_value(&dev_queries[i], NNBS);
		// t1 = gettime();
		// t_sum += (t1-t0); // don't care will count at host side
		// if (i == 0)
		// 	t_first = (t1-t0);

		// sse += (y[i] - yp) * (y[i] - yp);

#if DEBUG
		for (int k = 0; k < PROBDIM; k++)
			fprintf(fpout, "%.5f ", query_mem[i * (PROBDIM + 1) + k]);
#endif

		// err = 100.0 * fabs((yp - y[i]) / y[i]);

#if DEBUG
		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
#endif
		// err_sum += err;
	}

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
        
#if defined(DEBUG)
        /* Create/Open an output file */
        FILE *fpout = fopen("output.knn.txt","w");
#endif
        int vector_size = PROBDIM + 1;
        double *dev_mem, *dev_query_mem, *dev_ydata, *dev_query_ydata, *dev_fd;
        query_t *dev_queries;
	
	// ******************************************************************
	// ************************** Host mallocs **************************
	// ******************************************************************
	double *mem = (double *)malloc(TRAINELEMS * vector_size * sizeof(double));
	double *ydata = (double *)malloc(TRAINELEMS * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double));
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));
	double *query_ydata = (double*)malloc(QUERYELEMS * sizeof(double));

	// ******************************************************************
	// ************************** Load Data *****************************
	// ******************************************************************
	load_binary_data(trainfile, mem, NULL, TRAINELEMS * (PROBDIM + 1));
	load_binary_data(queryfile, query_mem, queries, QUERYELEMS * vector_size);

	// ******************************************************************
	// ************************** Device mallocs ************************
	// ******************************************************************
	/* Create handler arrays that will be used to separate xdata's PROBDIM vectors
	 * and the corresponding surrogate values, since we never need both
	 * in order to perform a computation.
	 * We either going to use the xdata of two points (ex. when calculating distance from one another)
	 * or use ydata (surrogates) (ex. when predicting the value of a query point)
	 */
	// allocate global memory on the device for the training element and query vectors
        cudaMalloc((void**)&dev_mem, TRAINELEMS * vector_size * sizeof(double));
        cudaMalloc((void**)&dev_ydata, TRAINELEMS * sizeof(double));
        cudaMalloc((void**)&dev_query_mem, QUERYELEMS * vector_size * sizeof(double));
        cudaMalloc((void**)&dev_queries, QUERYELEMS * sizeof(query_t));
        cudaMalloc((void**)&dev_query_ydata, QUERYELEMS * sizeof(double));

	// ******************************************************************
	// ************************** Host data init ************************
	// ******************************************************************

        // init all data on CPU **Then** send them to GPU
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

	// ******************************************************************
	// ************************** Copyout data to device ****************
	// ******************************************************************	
	cudaMemcpy(mem, dev_mem, TRAINELEMS * vector_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(query_mem, dev_query_mem, QUERYELEMS * vector_size * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(queries, dev_queries, QUERYELEMS * sizeof(query_t), cudaMemcpyHostToDevice);
	
	printf("here1\n");
	// what to do here with the query_t.x internal pointers?
	for (int i = 0; i < QUERYELEMS; i++)
	{
		cudaMalloc((void **)&(dev_queries[i].x), PROBDIM * sizeof(double));
		// cudaMemcpy(dev_queries[i].x, queries[i].x, PROBDIM * sizeof(double), cudaMemcpyHostToDevice);
	}
	printf("here2\n");

	cudaMemcpy(dev_ydata, ydata, TRAINELEMS * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_query_ydata, query_ydata, QUERYELEMS * sizeof(double), cudaMemcpyHostToDevice);


	/* COMPUTATION PART */
        double t_start = gettime();

        // cuda kernel launch here
        //sync before getting final time
        double t_sum = gettime() - t_start;
	
	/* CALCULATE AND DISPLAY RESULTS */

        // these will be calculated on the CPU

	// double mse = sse / QUERYELEMS;
	// double ymean = compute_mean(query_ydata, QUERYELEMS);
	// double var = compute_var(query_ydata, QUERYELEMS, ymean);
	// double r2 = 1 - (mse / var);

	// printf("Results for %d query points\n", QUERYELEMS);
	// printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	// printf("MSE = %.6f\n", mse);
	// printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	printf("Total time = %lf secs\n", t_sum);
	// printf("Time for 1st query = %lf secs\n", t_first);
	// printf("Time for 2..N queries = %lf secs\n", t_sum - t_first);
	// printf("Average time/query = %lf secs\n", (t_sum - t_first) / (QUERYELEMS - 1));

	/* CLEANUP */

#if defined(DEBUG)
	/* Close the output file */
	fclose(fpout);
#endif

	free(queries);
	free(query_mem);
	free(query_ydata);

	free(ydata);
	free(mem);

        cudaFree(dev_mem);
        cudaFree(dev_query_mem);
        cudaFree(dev_queries);
	cudaFree(dev_ydata);
        cudaFree(dev_query_ydata);

	return 0;
}
