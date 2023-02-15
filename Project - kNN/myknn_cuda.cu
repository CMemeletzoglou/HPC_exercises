#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

#define TRAIN_BLOCK_SIZE 1024 //initial testing

__device__ double find_knn_value(double *dev_mem, double *dev_ydata, query_t *q, int k, int block_start, int block_size)
{
	// double fd[knn];	      	      // function values for the knn neighbors

	compute_knn_brute_force_cuda(dev_mem, dev_ydata, q, PROBDIM, k, block_start, block_size); // brute-force / linear search

	// for (int i = 0; i < knn; i++)
	// 	fd[i] = ydata[nn_x[i]];

	
	// predict value inlined -> change this later to a function call
	double sum_v = 0.0;
	for (int i = 0; i < k; i++)
		sum_v += dev_ydata[i];

	return sum_v / k;
}

__global__ void knn_kernel(double *dev_mem, double *dev_ydata, double *dev_query_ydata, query_t *dev_queries, int k, double *sse, double *err_sum)
{
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	// double sse = 0.0;
	double err; //, err_sum = 0.0;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	for (int block_start = 0; block_start < TRAINELEMS; block_start += TRAIN_BLOCK_SIZE)
	{
		int query_idx = block_start + tid;
		// t0 = gettime();
		double yp = find_knn_value(dev_mem, dev_ydata, &(dev_queries[query_idx]), k, block_start, TRAIN_BLOCK_SIZE);
		printf("For query %d and block start %d, yp = %.5f\n", tid, block_start, yp);

		// double yp = find_knn_value(&dev_queries[i], NNBS);
		// t1 = gettime();
		// t_sum += (t1-t0); // don't care will count at host side
		// if (i == 0)
		// 	t_first = (t1-t0);

		// sse += (y[i] - yp) * (y[i] - yp);

		sse[tid] += (dev_query_ydata[tid] - yp) * (dev_query_ydata[tid] - yp);
		err = 100.0 * fabs((yp - dev_query_ydata[tid]) / dev_query_ydata[tid]);
		err_sum[tid] += err;
	}

#if DEBUG
		for (int k = 0; k < PROBDIM; k++)
			fprintf(fpout, "%.5f ", query_mem[i * (PROBDIM + 1) + k]);
#endif


#if DEBUG
		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
#endif
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

	// int dev;
	// cudaGetDevice(&dev); // get GPU device number
	// cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, dev); // get cuda Device Properties
	// size_t shared_mem_size = prop.sharedMemPerBlock; // shared memory size

#if defined(DEBUG)
        /* Create/Open an output file */
        FILE *fpout = fopen("output.knn.txt","w");
#endif
        int vector_size = PROBDIM + 1;
	double *dev_mem, *dev_ydata, *dev_query_ydata, *dev_sse, *dev_err; //*dev_query_mem
	query_t *dev_queries;
	
	// ******************************************************************
	// ************************** Host mallocs **************************
	// ******************************************************************
	double *mem = (double *)malloc(TRAINELEMS * vector_size * sizeof(double));	 // Training Element vectors
	double *ydata = (double *)malloc(TRAINELEMS * sizeof(double));	  	  	 // Training Element Surrogate values
	double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double)); // Query Element vectors
	double *query_ydata = (double*)malloc(QUERYELEMS * sizeof(double));		 // Query Element Surrogate values
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));		 // Query Element helper structs

	// ******************************************************************
	// ************************** Load Data *****************************
	// ******************************************************************
	load_binary_data(trainfile, mem, NULL, TRAINELEMS * (PROBDIM + 1));
	load_binary_data(queryfile, query_mem, queries, QUERYELEMS * vector_size);

	// ******************************************************************
	// ************************** Device mallocs ************************
	// ******************************************************************
	// allocate global memory on the device for the training element and query vectors

        cudaMalloc((void**)&dev_mem, TRAINELEMS * vector_size * sizeof(double));	 // Device Training Element vectors
        cudaMalloc((void**)&dev_ydata, TRAINELEMS * sizeof(double));			 // Device Training Element Surrogate values
        
	// ---- maybe un-needed ------------------------------
	// cudaMalloc((void**)&dev_query_mem, QUERYELEMS * vector_size * sizeof(double));	 // Device Query Element vectors
        // ---------------------------------------------------

        cudaMalloc((void**)&dev_query_ydata, QUERYELEMS * sizeof(double));		 // Device Query Element Surrogate values
        cudaMalloc((void**)&dev_queries, QUERYELEMS * sizeof(query_t));			 // Device Query Element helper structs

	cudaMalloc((void **)&dev_sse, QUERYELEMS * sizeof(double));
	cudaMalloc((void **)&dev_err, QUERYELEMS * sizeof(double));

	cudaMemset(dev_sse, 0, QUERYELEMS * sizeof(double));
	cudaMemset(dev_err, 0, QUERYELEMS * sizeof(double));

	// ******************************************************************
	// ************************** Host data init ************************
	// ******************************************************************
        // init all data on CPU **Then** send them to GPU

	for (int i = 0; i < TRAINELEMS; i++) // init training elements' surrogate values
	{
#if defined(SURROGATES)
		ydata[i] = mem[i * vector_size + PROBDIM];
#else
		ydata[i] = 0;
#endif
	}

	for (int i = 0; i < QUERYELEMS; i++) // init query elements' surrogate values
	{
#if defined(SURROGATES)
		query_ydata[i] = query_mem[i * vector_size + PROBDIM];		
#else
		query_ydata[i] = 0;
#endif
	}

	// ******************************************************************
	// ************************** Copyout data to device ****************
	// ******************************************************************	
	cudaMemcpy(dev_mem, mem, TRAINELEMS * vector_size * sizeof(double), cudaMemcpyHostToDevice); 		 // copy train elems
	cudaMemcpy(dev_ydata, ydata, TRAINELEMS * sizeof(double), cudaMemcpyHostToDevice); 			 // copy train elems surrogate values
       
        // ---- maybe un-needed ------------------------------
        // cudaMemcpy(query_mem, dev_query_mem, QUERYELEMS * vector_size * sizeof(double), cudaMemcpyHostToDevice); // copy query elems
        // ---------------------------------------------------

	cudaMemcpy(dev_query_ydata, query_ydata, QUERYELEMS * sizeof(double), cudaMemcpyHostToDevice); 		 // copy query elems surrogate values
        cudaMemcpy(dev_queries, queries, QUERYELEMS * sizeof(query_t), cudaMemcpyHostToDevice);			 // copy query elems structs

	/* COMPUTATION PART */
        double t_start = gettime();
        // cuda kernel launch here.. initial test.. 1 thread block with 1024 threads
	knn_kernel<<<1, 1024>>>(dev_mem, dev_ydata, dev_query_ydata, dev_queries, NNBS, dev_sse, dev_err);

	//sync before getting final time
	cudaDeviceSynchronize(); // wait for GPU to finish executin the kNN kernel
	double t_sum = gettime() - t_start;

	double sse = 0.0f, err_sum = 0.0f;
	double *buf = (double *)malloc(QUERYELEMS * sizeof(double));
	cudaMemcpy(buf, dev_sse, QUERYELEMS * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < QUERYELEMS; i++)
		sse += buf[i];

	cudaMemcpy(buf, dev_err, QUERYELEMS * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < QUERYELEMS; i++)
		err_sum += buf[i];

	/* CALCULATE AND DISPLAY RESULTS */

	// these will be calculated on the CPU

	double mse = sse / QUERYELEMS;
	double ymean = compute_mean(query_ydata, QUERYELEMS);
	double var = compute_var(query_ydata, QUERYELEMS, ymean);
	double r2 = 1 - (mse / var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	printf("Total time = %lf secs\n", t_sum);
	// printf("Time for 1st query = %lf secs\n", t_first);
	// printf("Time for 2..N queries = %lf secs\n", t_sum - t_first);
	// printf("Average time/query = %lf secs\n", (t_sum - t_first) / (QUERYELEMS - 1));

	/* CLEANUP */

#if defined(DEBUG)
	/* Close the output file */
	fclose(fpout);
#endif

	free(mem);
	free(ydata);
	free(query_mem);
	free(query_ydata);
	free(queries);

        cudaFree(dev_mem);
	cudaFree(dev_ydata);
        // cudaFree(dev_query_mem);
        cudaFree(dev_query_ydata);
        cudaFree(dev_queries);

	cudaFree(dev_sse);
	cudaFree(dev_err);

	return 0;
}
