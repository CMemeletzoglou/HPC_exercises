#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_cuda.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

// #define TRAIN_BLOCK_SIZE 1024 //initial testing

static int ntrain_blocks, train_block_size;


__global__ void knn_kernel(double *mem, double *ydata, double *query_mem, query_t *query_ydata, int k, double *sse, double *err, int *nblocks, int *block_size)
{
	double t0, t1, t_first = 0.0, t_sum = 0.0;
        double yp;
	
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	// threads of the same column, will handle the same query
	// threads of the same row will compute the neighbors of the query assigned to their column, 
	// for the training element block assigned to their row

	// so tx indicates the query element index, while ty indicates the training element block index 
	
	if(tx < QUERYELEMS && ty < *nblocks) // stay inside problem boundaries
		compute_knn_brute_force_cuda(mem, ydata, &(query_mem[tx]), PROBDIM, k, ty, *block_size);


		// compute_knn_brute_force_cuda(dev_mem, dev_ydata, &(dev_queries[tx]), PROBDIM, k, ty, *block_size);






	// for (int block_start = 0; block_start < TRAINELEMS; block_start += TRAIN_BLOCK_SIZE)
	// 	compute_knn_brute_force_cuda(dev_mem, dev_ydata, &(dev_queries[tid]), PROBDIM, k, block_start, TRAIN_BLOCK_SIZE); // brute-force / linear search

	// Predict query point value
        // yp = predict_value(&(dev_queries[tid]), dev_ydata, PROBDIM, NNBS);
        // sse[tid] = (dev_query_ydata[tid] - yp) * (dev_query_ydata[tid] - yp);
        // err[tid] = 100.0 * fabs((yp - dev_query_ydata[tid]) / dev_query_ydata[tid]);
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
	double *dev_mem, *dev_ydata, *dev_query_ydata, *dev_sse, *dev_err;
	// query_t *dev_queries;

	double *dev_query_mem;
	
	// ******************************************************************
	// ************************** Host mallocs **************************
	// ******************************************************************
	double *mem = (double *)malloc(TRAINELEMS * vector_size * sizeof(double));	 // Training Element vectors
	double *ydata = (double *)malloc(TRAINELEMS * sizeof(double));	  	  	 // Training Element Surrogate values
	double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double)); // Query Element vectors
	double *query_ydata = (double*)malloc(QUERYELEMS * sizeof(double));		 // Query Element Surrogate values
	// query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));		 // Query Element helper structs

	double *train_buf = (double*)malloc(TRAINELEMS * PROBDIM * sizeof(double));
	double *query_buf = (double*)malloc(QUERYELEMS * PROBDIM * sizeof(double));

	// ******************************************************************
	// ************************** Load Data *****************************
	// ******************************************************************
	load_binary_data(trainfile, mem, NULL, TRAINELEMS * (PROBDIM + 1));
	// load_binary_data(queryfile, query_mem, queries, QUERYELEMS * vector_size);
	load_binary_data(queryfile, query_mem, NULL, QUERYELEMS * vector_size);

	extract_vectors(mem, train_buf, TRAINELEMS, PROBDIM + 1, PROBDIM);

	// construct a "pure" query elements array to pass to the device
	extract_vectors(query_mem, query_buf, QUERYELEMS, PROBDIM + 1, PROBDIM);

	// ******************************************************************
	// ************************** Device mallocs ************************
	// ******************************************************************
	// allocate global memory on the device for the training element and query vectors

        cudaMalloc((void**)&dev_mem, TRAINELEMS * PROBDIM * sizeof(double));	 	 // Device Training Element vectors
        cudaMalloc((void**)&dev_ydata, TRAINELEMS * sizeof(double));			 // Device Training Element Surrogate values

        cudaMalloc((void**)&dev_query_ydata, QUERYELEMS * sizeof(double));		 // Device Query Element Surrogate values
        // cudaMalloc((void**)&dev_queries, QUERYELEMS * sizeof(query_t));			 // Device Query Element helper structs

	cudaMalloc((void**)&dev_query_mem, QUERYELEMS * PROBDIM * sizeof(double)); // array to host "pure" query element vectors

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
	cudaMemcpy(dev_mem, train_buf, TRAINELEMS * PROBDIM * sizeof(double), cudaMemcpyHostToDevice); 		 // copy train elems
	cudaMemcpy(dev_ydata, ydata, TRAINELEMS * sizeof(double), cudaMemcpyHostToDevice); 			 // copy train elems surrogate values
       
	cudaMemcpy(dev_query_ydata, query_ydata, QUERYELEMS * sizeof(double), cudaMemcpyHostToDevice); 		 // copy query elems surrogate values
        // cudaMemcpy(dev_queries, queries, QUERYELEMS * sizeof(query_t), cudaMemcpyHostToDevice);			 // copy query elems structs

	cudaMemcpy(dev_query_mem, query_buf, QUERYELEMS * PROBDIM * sizeof(double), cudaMemcpyHostToDevice);

	/* Each thread block will have #rows = #Training Element blocks and #cols = some multiple of 32.
	 * Assume that we divide the Training Elements into 2^5 = 32 Training Element blocks, where
	 * each block contains 2^15 Training Elements
	 */
	train_block_size = pow(2, 15);
	ntrain_blocks = TRAINELEMS / train_block_size;
	dim3 block_size(ntrain_blocks, 32);

	int *dev_train_block_size, *dev_ntrain_blocks;
	cudaMalloc((void**)&dev_train_block_size, sizeof(int));
	cudaMalloc((void**)&dev_ntrain_blocks, sizeof(int));
	cudaMemcpy(&dev_train_block_size, &train_block_size, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(&dev_ntrain_blocks, &ntrain_blocks, sizeof(int), cudaMemcpyHostToDevice);

	/* COMPUTATION PART */
        double t_start = gettime();

	
	knn_kernel<<<QUERYELEMS / 32, block_size>>>(dev_mem, dev_ydata, dev_query_mem, dev_query_ydata, NNBS, dev_sse, dev_err, dev_ntrain_blocks, dev_train_block_size);
	
	// knn_kernel<<<QUERYELEMS / 32, block_size>>>(dev_mem, dev_ydata, dev_query_ydata, dev_queries, NNBS, dev_sse, dev_err, dev_ntrain_blocks, dev_train_block_size);


	// knn_kernel<<<8, 128>>>(dev_mem, dev_ydata, dev_query_ydata, dev_queries, NNBS, dev_sse, dev_err);
	// knn_kernel<<<8, 128>>>(dev_mem, dev_ydata, dev_query_ydata, dev_queries, NNBS, dev_sse, dev_err);
	// knn_kernel<<<32, 32>>>(dev_mem, dev_ydata, dev_query_ydata, dev_queries, NNBS, dev_sse, dev_err);
	// knn_kernel<<<16, 64>>>(dev_mem, dev_ydata, dev_query_ydata, dev_queries, NNBS, dev_sse, dev_err);

	// sync device and host before getting final time
	cudaDeviceSynchronize();
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
	// free(queries);
	free(buf);
        free(train_buf);
	free(query_buf);

        cudaFree(dev_mem);
	cudaFree(dev_ydata);
        cudaFree(dev_query_ydata);
        // cudaFree(dev_queries);

	cudaFree(dev_sse);
	cudaFree(dev_err);
	cudaFree(dev_ntrain_blocks);
	cudaFree(dev_train_block_size);

	return 0;
}
