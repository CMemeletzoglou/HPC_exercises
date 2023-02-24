#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_cuda.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

#define INF 1e99
/* IDEA : Calculate a matrix of size [QUERYELEMS, TRAINELEMS] where the element [i,j] is the
 * distance between the i-th query point and the j-th training point. Since this is a lightweight
 * task, it may be assigned to a device thread. 
 * Then we may find the k minimum values of each row. For each one of those k values of a row, 
 * we will be preserving their index inside the matrix and thus the index of the point inside the TRAINELEMS array.
 * We may then predict the value of the query and calculate the errors, accumulating them in a device array
 * copyout those arrays to host, reduce them to the desired metrics and exit.
 */

// number of rows (in the overall matrix) to calculate in parallel
#define QUERY_BLOCK_SIZE 		16

// Each thread block should calculate a tile of this "global" matrix. Since the max size of a thread block is 1024 we
// derive the TRAIN_BLOCK_SIZE (which is the number of columns (in the overall matrix) to calculate in parallel)
// in the following manner.
#define TRAIN_BLOCK_SIZE 		(1024 / QUERY_BLOCK_SIZE)

// The amount of tiles (thread blocks) required to calculate a full block row of the matrix
#define ROW_THREAD_BLOCKS 		(TRAINELEMS / TRAIN_BLOCK_SIZE)

__device__ void thread_block_reduction(double *dist_vec, int *global_nn_idx, double *global_nn_dist, int k)
{	
        __shared__ int trainel_idx_buf[QUERY_BLOCK_SIZE * TRAIN_BLOCK_SIZE];

        int curr_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int next_idx;

	for(int neigh = 0; neigh < k; neigh++) // all threads will participate in finding each of the k neighbors
        {
                // Initially the buffer line for each query (indexed by threadIdx.y)
                // should contain the indexes of the local training point (threadIdx.x)
                // since the minimum distance it knows yet is the one that this thread calculated.
                // Thus, on the left handside, we index the buffer cell that corresponds to the current thread
                // and on the right handside we store the index of the local training point.

                // Because the training point evaluated at each column does not change, on the right handside we
                // may also store the position of the thread, inside the thread block. We may later recover the threadIdx.x
                // (i.e. the index of the local training point), by subtracting the offset introduced by the local query (i.e. threadIdx.y * blockDim.x).
                // We do this last step so we may easily index the dist_vec array, without having to calculate the offset indroduced by the local query
                // at each iteration.
                trainel_idx_buf[curr_idx] = curr_idx;
                __syncthreads();

                for (int j = 0; j < (int)log2f(blockDim.x); j++)
                {
                        next_idx = curr_idx + (1 << j);
                        int _next = trainel_idx_buf[next_idx];
                        int _curr = trainel_idx_buf[curr_idx];

                        __syncthreads();
                        // same as threadIdx.x % pow(2, j+1) && threadIdx.x < len, only a lot faster
                        if (!(threadIdx.x & ((1 << (j+1)) - 1)))
                                trainel_idx_buf[curr_idx] = (dist_vec[_curr] < dist_vec[_next]) ? _curr : _next;
                        __syncthreads();
                }

		if (threadIdx.x == 0)
		{
			// Explains the indexing on the **left** hadside
                        // global_nn_idx is a tensor (3d) (x-dim: query | y-dim: thread block | z-dim: k_nn_candidates)
                        // threadIdx.y*gridDim.x*k : Jump over all the previous queries, that have k neighbor candidates for each thread block.
                        //                           This offset will move you to the first element of the (2d) slice of the global_nn_idx tensor
                        //                           that contains all candidate neighbors for the query indexed by threadIdx.y.
                        //                           We now have a matrix, where the i-th row will contain the k candidate neighbors, 
			// 			     calculated by the i-th thread block.
                        // blockIdx.x*k : Jump over all the previous thread blocks (inside the matrix), that will each store k candidate neighbors.
                        //                This offset will move you to the first element of the row that corresponds to the (1d) vector of this matrix.
                        //                We now have a 1d vector, where the j-th element will contain the j-th candidate neighbor,
			//	  	  calculated by the i-th thread block, at iteration j.
			//
                        // neigh : Iterates over this 1d vector and stores the current iteration's neighbor.
                        // 
                        // ***************************************************************************************************************************************************************
                        // ***************************************************************************************************************************************************************
                        // 
                        // Explains the indexing on the **right** handside
                        // blockIdx.x * blockDim.x : Each thread block has calculated the distances for TRAIN_BLOCK_SIZE (= blockDim.x) training elements.
                        //                           Thus, in order to get the global index of the first training element, this thread block has been assigned,
                        //                           we should jump over all the previous thread blocks.
                        // (trainel_idx_buf[curr_idx] - threadIdx.y * blockDim.x) : Recover threadIdx.x, of the thread that has been assigned the training element with
                        //                                                          the minimum distance from query point, as explained above.
                        //                                                          This is the index of the training element inside the thread block (aka threadIdx.x)
                        global_nn_idx[(threadIdx.y*gridDim.x + blockIdx.x)*k + neigh] = 
                                        blockIdx.x * blockDim.x + (trainel_idx_buf[curr_idx] - threadIdx.y * blockDim.x);

                        global_nn_dist[(threadIdx.y*gridDim.x + blockIdx.x)*k + neigh] = 
                                        dist_vec[trainel_idx_buf[curr_idx]];
                                        
			dist_vec[trainel_idx_buf[curr_idx]] = INF;
		}
        }
}

__global__ void compute_distances_kernel(double *mem, double *query_mem, int query_block_offset,
					 int *global_nn_idx, double *global_nn_dist, int k, int dim,
					 size_t trainel_block_size, size_t queryel_block_size)
{
	extern __shared__ char arr[];
	
	double *trainel_block = (double *)arr;
	double *query_block = (double *)(arr + trainel_block_size);
	double *dist_vec = (double *)(arr + trainel_block_size + queryel_block_size);

	int trainel_block_offset = blockIdx.x * blockDim.x;

	// same thread-block row threads -> same threadIdx.y -> compute with the same query (global ty)
	
        // Indexes for the global memory (be careful you only have the partial matrix in memory)
        int global_tx = blockIdx.x * blockDim.x + threadIdx.x; // matrix col this thread is in
        int global_ty = blockIdx.y * blockDim.y + threadIdx.y; // matrix row this thread is in
	
        // Indexes for the shared memory
	int local_trainel_idx = threadIdx.x; // thread-block col of current thread
	int local_query_idx = threadIdx.y; // thread-block row of current thread

	// load data into the proper shared memory regions from device global memory
	// only the thread block's "zero" thread loads these data
        if(threadIdx.x == 0 && threadIdx.y == 0) 
        {
		memcpy(trainel_block, mem + trainel_block_offset * dim, trainel_block_size);
		memcpy(query_block, query_mem + query_block_offset * dim, queryel_block_size);
        }
        
	__syncthreads();

	// each thread computes the distance for its query point with its training element
	// then it updates its respective position in the distances vector
	dist_vec[local_query_idx * blockDim.x + local_trainel_idx] = 
                	compute_dist(&query_block[local_query_idx * dim], &trainel_block[local_trainel_idx * dim], dim);
	__syncthreads();

        thread_block_reduction(dist_vec, global_nn_idx, global_nn_dist, k);
}

__global__ void reduce_distance_kernel(int *global_nn_idx, double *global_nn_dist, 
                                       int *out_global_nn_idx, double *out_global_nn_dist, 
                                       int len, int k, size_t dist_vec_size)
{	
        extern __shared__ char shared_arr[]; 			 // dynamically allocated shared memory
        double *dist_vec = (double *)shared_arr;
        int *idx_vec = (int *)(shared_arr + dist_vec_size);

        __shared__ int local_el_idx_buf[QUERY_BLOCK_SIZE * TRAIN_BLOCK_SIZE]; // static shared memory buffer

	// load data into the proper shared memory regions from device global memory
	// only the thread block's "zero" thread loads these data
        if(threadIdx.x == 0) 
        {
		memcpy(&(dist_vec[threadIdx.y*len]), &(global_nn_dist[len*gridDim.x*threadIdx.y + len*blockIdx.x]), len*sizeof(double));
		memcpy(&(idx_vec[threadIdx.y*len]), &(global_nn_idx[len*gridDim.x*threadIdx.y + len*blockIdx.x]), len*sizeof(int));
        }
	__syncthreads();

        // Decide on the number of iterations required to do the reduction to len elements
	int num_iter = log2f(blockDim.x);

        int curr_tid = threadIdx.y * blockDim.x  + threadIdx.x;
        int next_tid, curr_el, next_el;
	for(int neigh = 0; neigh < k; neigh++)
	{
                // First iteration will be performed on data twice the size of the thread block
                // so we may use all threads for this first step, we should only be careful with the indexing
                // If the length (len) of the data we want to reduce is less than the available number of threads for
                // each query (blockDim.x), we obviously should not perform this step
		__syncthreads();
                if (len == 2*blockDim.x)
                {
                        curr_el = 2*curr_tid;
                        next_el = curr_el + 1;
                        local_el_idx_buf[curr_tid] = (dist_vec[curr_el] < dist_vec[next_el]) 
                                		      ? curr_el
                                		      : next_el;
                }
                else
                        local_el_idx_buf[curr_tid] = curr_tid;

                __syncthreads();
                for (int j = 0; j < num_iter; j++)
                {
                        next_tid = curr_tid + (1 << j);
                        curr_el = local_el_idx_buf[curr_tid];
                        next_el = local_el_idx_buf[next_tid];
                        
                        __syncthreads();
                        // same as threadIdx.x % pow(2, j+1) && threadIdx.x < len, only a lot faster
                        if (!(threadIdx.x & ((1 << (j+1)) - 1)) && threadIdx.x < len)
                                local_el_idx_buf[curr_tid] = (dist_vec[curr_el] < dist_vec[next_el]) ? curr_el : next_el;
                        __syncthreads();
                }

                if (threadIdx.x == 0)
                {                        
			out_global_nn_idx[(threadIdx.y*gridDim.x + blockIdx.x)*k + neigh] = idx_vec[local_el_idx_buf[curr_tid]];
			out_global_nn_dist[(threadIdx.y*gridDim.x + blockIdx.x)*k + neigh] = dist_vec[local_el_idx_buf[curr_tid]];
			dist_vec[local_el_idx_buf[curr_tid]] = INF;
                }
	}
}

__global__ void predict_query_values(double *dev_ydata, double *dev_query_ydata, int *dev_nn_idx, int query_block_start, 
				     int k, double *dev_yp_vals, double *dev_sse, double *dev_err)
{
        double sum = 0.0;
        double yp;

	int tid = threadIdx.x; // running with a 1D Thread Block
        int query_idx = tid;
        int g_query_idx = query_block_start + query_idx;

	for(int i = 0; i < k; i++)
		sum += dev_ydata[dev_nn_idx[query_idx*k + i]];
                
	// Predict the value
	yp = sum / k;

#if defined(DEBUG)
	dev_yp_vals[g_query_idx] = yp; // write the computed value to the global helper array, if in DEBUG mode
#endif
	// Compute error metrics. The dev_sser and dev_err arrays, will be copied-out to the CPU, who will reduce their values
	dev_sse[g_query_idx] = (dev_query_ydata[g_query_idx] - yp) * (dev_query_ydata[g_query_idx] - yp);
        dev_err[g_query_idx] = 100.0 * fabs((yp - dev_query_ydata[g_query_idx]) / dev_query_ydata[g_query_idx]);
}

int main(int argc, char **argv)
{
	/* Load all data from files in memory */
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}
	char *trainfile = argv[1];
	char *queryfile = argv[2];

        assert(TRAIN_BLOCK_SIZE > 32);

	// int dev;
	// cudaGetDevice(&dev); // get GPU device number
	// cudaDeviceProp prop;
	// cudaGetDeviceProperties(&prop, dev); // get cuda Device Properties
	// size_t shared_mem_size = prop.sharedMemPerBlock; // shared memory size

#if defined(DEBUG)
        /* Create/Open an output file */
        FILE *fpout = fopen("output.knn_cuda.txt","w");
#endif
        int vector_size = PROBDIM + 1;
	double *dev_mem, *dev_ydata, *dev_query_ydata, *dev_query_mem, *dev_nn_dist, *dev_temp_nn_dist, *dev_sse, *dev_err;
	double *dev_yp_vals = NULL;
	int *dev_nn_idx, *dev_temp_nn_idx;

	// ******************************************************************
	// ************************** Host mallocs **************************
	// ******************************************************************
	double *mem = (double *)malloc(TRAINELEMS * vector_size * sizeof(double));	 // Training Element vectors
	double *ydata = (double *)malloc(TRAINELEMS * sizeof(double));	  	  	 // Training Element Surrogate values
	double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double)); // Query Element vectors
	double *query_ydata = (double*)malloc(QUERYELEMS * sizeof(double));		 // Query Element Surrogate values

	double *train_buf = (double*)malloc(TRAINELEMS * PROBDIM * sizeof(double));
	double *query_buf = (double*)malloc(QUERYELEMS * PROBDIM * sizeof(double));

	double *buf = (double *)malloc(QUERYELEMS * sizeof(double));

        double *temp_dist = (double*)malloc((TRAINELEMS) * QUERY_BLOCK_SIZE * sizeof(double));
        int *temp_idx = (int*)malloc((TRAINELEMS) * QUERY_BLOCK_SIZE * sizeof(int));

#if defined(DEBUG)
	double *yp_vals = (double*)malloc(QUERYELEMS * sizeof(double));
#endif

	// ******************************************************************
	// ************************** Load Data *****************************
	// ******************************************************************
	load_binary_data(trainfile, mem, TRAINELEMS * (PROBDIM + 1));
	load_binary_data(queryfile, query_mem, QUERYELEMS * vector_size);

	extract_vectors(mem, train_buf, TRAINELEMS, PROBDIM + 1, PROBDIM);
	// construct a "pure" query elements array to pass to the device
	extract_vectors(query_mem, query_buf, QUERYELEMS, PROBDIM + 1, PROBDIM);

	// ******************************************************************
	// ************************** Device mallocs ************************
	// ******************************************************************
	// allocate global memory on the device for the training element and query vectors

        cudaMalloc((void**)&dev_mem, TRAINELEMS * PROBDIM * sizeof(double));	 	 // Device Training Element vectors
        cudaMalloc((void**)&dev_ydata, TRAINELEMS * sizeof(double));			 // Device Training Element Surrogate values

	cudaMalloc((void**)&dev_query_mem, QUERYELEMS * PROBDIM * sizeof(double)); 	 // array to host "pure" query element vectors
        cudaMalloc((void**)&dev_query_ydata, QUERYELEMS * sizeof(double));		 // Device Query Element Surrogate values

	/* Allocate enough space for each thread block to store the k nearest neighbors it found (partial reduction results)
	 * We must preserve both the index of the nearest neighbors and their distance from the query point
	 * since we will later reduce the k*ROW_THREAD_BLOCKS nearest neighbors to the final k for each query point.
	 */
	
        // TODO: Since all the initial calculations for the distances between
        //       each query and each training element is temporarily stored in the shared memory of each thread block
        //       and then before updating the arrays below, we reduce to 32 neighbors, we may only allocate
        //       ROW_THREAD_BLOCKS * QUERY_BLOCK_SIZE * NNBS
	cudaMalloc((void**)&dev_nn_dist, TRAINELEMS * QUERY_BLOCK_SIZE * sizeof(double));
	cudaMalloc((void**)&dev_nn_idx, TRAINELEMS * QUERY_BLOCK_SIZE * sizeof(int));

        // Temporary arrays that should have enough space to store the output of each reduction step.
        // The first reduction step will reduce each 128 vector of distances to 32 neighbors,
        // thus we only require one fourth the space that is needed to store the initial output of the
        // compute_distances_kernel, which is QUERY_BLOCK_SIZE*ROW_THREAD_BLOCKS*NNBS.
        // But because we need them to be interchangable with the dev_nn_dist and dev_nn_idx, we must
        // allocate the same size with the original arrays.
        cudaMalloc((void**)&dev_temp_nn_dist, TRAINELEMS * QUERY_BLOCK_SIZE * sizeof(double));
        cudaMalloc((void**)&dev_temp_nn_idx, TRAINELEMS * QUERY_BLOCK_SIZE * sizeof(int));

	cudaMalloc((void **)&dev_sse, QUERYELEMS * sizeof(double));
	cudaMalloc((void **)&dev_err, QUERYELEMS * sizeof(double));

#if defined(DEBUG)
	cudaMalloc((void **)&dev_yp_vals, QUERYELEMS * sizeof(double));
#endif

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

	// ***************************************************************************
	// ************************** Copyin data to device **************************
	// ***************************************************************************
	cudaMemcpy(dev_mem, train_buf, TRAINELEMS * PROBDIM * sizeof(double), cudaMemcpyHostToDevice); 		 // copy train elems
	cudaMemcpy(dev_ydata, ydata, TRAINELEMS * sizeof(double), cudaMemcpyHostToDevice); 			 // copy train elems surrogate values
       
	cudaMemcpy(dev_query_mem, query_buf, QUERYELEMS * PROBDIM * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_query_ydata, query_ydata, QUERYELEMS * sizeof(double), cudaMemcpyHostToDevice); 		 // copy query elems surrogate values

	dim3 block_size(TRAIN_BLOCK_SIZE, QUERY_BLOCK_SIZE, 1);
	dim3 grid_dim(ROW_THREAD_BLOCKS, 1, 1);

	size_t dist_vector_size = QUERY_BLOCK_SIZE * TRAIN_BLOCK_SIZE * sizeof(double);
	size_t trainel_block_size = TRAIN_BLOCK_SIZE * PROBDIM * sizeof(double);
	size_t queryel_block_size = QUERY_BLOCK_SIZE * PROBDIM * sizeof(double);

	size_t shared_mem_size = dist_vector_size + trainel_block_size + queryel_block_size;

	/* Each thread block's shared memory is comprised of :
	 * 
	 *   - a trainel_block -> TRAIN_BLOCK_SIZE * PROBDIM * sizeof(double)
	 *   - a queryel_block -> QUERY_BLOCK_SIZE * PROBDIM * sizeof(double)
	 *   - a dist_vector -> QUERY_BLOCK_SIZE * TRAIN_BLOCK_SIZE * sizeof(double)
	 */
	assert(block_size.x % 2 == 0);

	float num_thread_blocks;
        int len;
        size_t reduction_shared_mem_size;
        double sse = 0.0f, err_sum = 0.0f;
        double *_dtemp;
        int *_itemp;

	/* COMPUTATION PART */
        double t_start = gettime();
	for(int i = 0; i < QUERYELEMS; i += QUERY_BLOCK_SIZE)
	{
        	compute_distances_kernel<<<grid_dim, block_size, shared_mem_size>>>(dev_mem, dev_query_mem, i, 
		 					dev_nn_idx, dev_nn_dist, NNBS, PROBDIM, trainel_block_size, queryel_block_size);  // compute a "chunk" of rows
                // Check for any cuda errors from the above kernel call
                assert(cudaGetLastError() == 0);

		num_thread_blocks = ROW_THREAD_BLOCKS * NNBS / (2 * TRAIN_BLOCK_SIZE);

		dim3 reduction_block_size(TRAIN_BLOCK_SIZE, QUERY_BLOCK_SIZE, 1);
		while(1)
		{
			// Calculate the amount of elements in each thread block (per query)
			len = num_thread_blocks * (2 * TRAIN_BLOCK_SIZE) / ceil(num_thread_blocks);

			// Fix the num_thread_blocks, so if there are less than TRAIN_BLOCK_SIZE elements left
			// you will still use 1 thread block in order to reduce them to NNBS neighbors
			num_thread_blocks = ceil(num_thread_blocks);

			// Calculate the number of thread blocks required
			dim3 reduction_grid_dim(num_thread_blocks, 1, 1);

			// For each query we will load in the shared memory 2*TRAIN_BLOCK_SIZE nearest neighbors (i.e. their distances and idx)
			// The shared memory will be structured as:
			// [query0 distances][query1 distances]...[query0 indexes][query1 indexes]...
			// Calculate the shared memory size
			// (REMINDER : len takes into account that we will require 2*blockDim.x distances for the first iteration)
			size_t reduction_dist_vector_size = QUERY_BLOCK_SIZE * len * sizeof(double);
			size_t reduction_idx_vector_size = QUERY_BLOCK_SIZE * len * sizeof(int);
			reduction_shared_mem_size = reduction_dist_vector_size + reduction_idx_vector_size;

			reduce_distance_kernel<<<reduction_grid_dim, reduction_block_size, reduction_shared_mem_size>>>
				        (dev_nn_idx, dev_nn_dist, dev_temp_nn_idx, dev_temp_nn_dist, len, NNBS, reduction_dist_vector_size);
			// Check for any cuda errors from the above kernel call
			assert(cudaGetLastError() == 0);

                        // Swap the pointers of the input and output device arrays
                        _dtemp = dev_nn_dist; dev_nn_dist = dev_temp_nn_dist; dev_temp_nn_dist = _dtemp;
                        _itemp = dev_nn_idx; dev_nn_idx = dev_temp_nn_idx; dev_temp_nn_idx = _itemp;

			// Update the loop-control variable
			if (num_thread_blocks == 1)
				break;
                        
			num_thread_blocks = num_thread_blocks * NNBS / (2 * block_size.x);
		}

		// Find yp for each query and error metrics
		predict_query_values<<<1, QUERY_BLOCK_SIZE>>>(dev_ydata, dev_query_ydata, dev_nn_idx, i, NNBS, dev_yp_vals, dev_sse, dev_err);
		// Check for any cuda errors from the above kernel call
		assert(cudaGetLastError() == 0);
	}

	// sync device and host before getting final time
	cudaDeviceSynchronize();
        double t_sum = gettime() - t_start;

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
	printf("Average time/query = %lf secs\n", t_sum / QUERYELEMS);

#if defined(DEBUG)
	cudaMemcpy(yp_vals, dev_yp_vals, QUERYELEMS * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = 0; i < QUERYELEMS; i++)
		fprintf(fpout,"%.5f %.5f %.2f\n", query_ydata[i], yp_vals[i], buf[i]);
#endif

	/* CLEANUP */

#if defined(DEBUG)
	/* Close the output file */
	fclose(fpout);
	free(yp_vals);
	cudaFree(dev_yp_vals);
#endif

	free(mem);
	free(ydata);
	free(query_mem);
	free(query_ydata);
	free(buf);
        free(train_buf);
	free(query_buf);
        
        free(temp_dist);
        free(temp_idx);

        cudaFree(dev_mem);
	cudaFree(dev_ydata);
	cudaFree(dev_query_mem);
        cudaFree(dev_query_ydata);
	cudaFree(dev_nn_dist);
	cudaFree(dev_nn_idx);
        cudaFree(dev_temp_nn_dist);
        cudaFree(dev_temp_nn_idx);
	cudaFree(dev_sse);
	cudaFree(dev_err);

	return 0;
}
