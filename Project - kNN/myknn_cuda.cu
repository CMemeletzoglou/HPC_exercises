#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func.h"
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

// TODO : Generalize this
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
        double next_dist;

        int curr_idx = threadIdx.y * blockDim.x + threadIdx.x;
        int next_idx;

	for(int neigh = 0; neigh < k; neigh++) // all threads will participate in finding each of the k neighbors
        {
                // Initially the buffer line for each query (indexed by threadIdx.y) ([0, QUERY_BLOCK_SIZE-1])
                // should contain the indexes of the local training point (threadIdx.x) ([0, TRAIN_BLOCK_SIZE-1])
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
                        next_idx = curr_idx + pow(2, j);
                        int _next = trainel_idx_buf[next_idx];
                        int _curr = trainel_idx_buf[curr_idx];
                        __syncthreads();

                        if ((threadIdx.x % (int)pow(2, j+1)) == 0)
                        {
                                trainel_idx_buf[curr_idx] = (dist_vec[_curr] < dist_vec[_next]) ? _curr : _next;
                        }
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
        __syncthreads();
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

		/* __CHANGE__: We had forgotten to use the query_block_offset argument passed,
		 * for the calculation of the query_mem "loading starting point".
		 */
		memcpy(query_block, query_mem + query_block_offset * dim, queryel_block_size);
        }
        
	__syncthreads();

	// each thread computes the distance for its query point with its training element
	// then it updates its respective position in the distances vector

	/* __CHANGE__: Prior to this change each thread called compute_dist with
	 * &query_block[local_query_idx] and &trainel_block[local_trainel_idx],
	 * which is WRONG.
	 * For example, ff a thread needs to compute the distance of query point 1 and
	 * training element 1, we must not index the respective shared memory arrays,
	 * using 1, because those arrays are 1D arrays whose elements are doubles,
	 * but we must think of them as **"vectors" of size dim**.
	 * So, when a thread needs the training element "1" it does not need
	 * trainel_block[1] but trainel_block[1*dim], to skip the previous vector(s).
	 */
	dist_vec[local_query_idx * blockDim.x + local_trainel_idx] = 
                	compute_dist(&query_block[local_query_idx * dim], &trainel_block[local_trainel_idx * dim], dim);
        // printf("threadIdx.x = %d, threadIdx.y = %d | wrote at dist_vec[%d] -> %.5f\n", 
        //                 threadIdx.x, 
        //                 threadIdx.y, 
        //                 local_query_idx * blockDim.x + local_trainel_idx, 
        //                 compute_dist(&query_block[local_query_idx * dim], &trainel_block[local_trainel_idx * dim], dim));

	__syncthreads();

        // thread_block_reduction(dist_vec, global_nn_idx, global_nn_dist, k);

        global_nn_idx[threadIdx.y * TRAINELEMS + (blockIdx.x * blockDim.x + threadIdx.x)] = 
                        blockIdx.x * blockDim.x + threadIdx.x;

        global_nn_dist[threadIdx.y * TRAINELEMS + (blockIdx.x * blockDim.x + threadIdx.x)] = 
                        dist_vec[local_query_idx * blockDim.x + local_trainel_idx];
}

__global__ void reduce_distance_kernel(int *global_nn_idx, double *global_nn_dist, int len, int k, size_t dist_vec_size)
{
        __shared__ int buf[QUERY_BLOCK_SIZE * TRAIN_BLOCK_SIZE]; // static shared memory buffer
	
        extern __shared__ char shared_arr[]; 			 // dynamically allocated shared memory
        double *dist_vec = (double *)shared_arr;
        int *idx_vec = (int *)(shared_arr + dist_vec_size);

        // Indexes for the global memory (be careful you only have the partial matrix in memory)
        int global_tx = blockIdx.x * blockDim.x + threadIdx.x; // matrix col this thread is in
        int global_ty = blockIdx.y * blockDim.y + threadIdx.y; // matrix row this thread is in
	
        // Indexes for the shared memory
	int local_neigh_idx = threadIdx.x; // thread-block col of current thread -> local "candidate neighbor"
	int local_query_idx = threadIdx.y; // thread-block row of current thread -> local query point under reduction (same as global)

	// load data into the proper shared memory regions from device global memory
	// only the thread block's "zero" thread loads these data
        if(threadIdx.x == 0) 
        {
		memcpy(dist_vec + threadIdx.y * len, global_nn_dist + (threadIdx.y * gridDim.x * k + blockIdx.x * k), len * sizeof(double));
		memcpy(idx_vec + threadIdx.y * len, global_nn_idx + (threadIdx.y * gridDim.x + blockIdx.x) * k, len * sizeof(int));
        }

	__syncthreads();

        // Decide on the number of iterations required to do the reduction to len elements
	int num_iter = log2f(len);

        int thread_block_local_tid = threadIdx.y * blockDim.x  + threadIdx.x;
        int curr_idx = thread_block_local_tid;
        int next_idx;
	for(int neigh = 0; neigh < k; neigh++)
	{
                // first iteration will be performed on data twice the size of the thread block
                // so we may use all threads for this first step, we should only be careful with the indexing
                // If the length (len) of the data we want to reduce is less than the available number of threads for
                // each query (blockDim.x), we obviously should not perform this step
                buf[thread_block_local_tid] = threadIdx.y * blockDim.x  + threadIdx.x; // = thread_block_local_tid
                
                if (len > blockDim.x)
                {
		        __syncthreads();
                        buf[thread_block_local_tid] = (dist_vec[2*thread_block_local_tid] < dist_vec[2*thread_block_local_tid+1]) 
                                		      ? 2 * thread_block_local_tid
                                		      : 2 * thread_block_local_tid + 1;
                        --num_iter;
                }

                for (int j = 0; j < num_iter; j++) // reduce the number of threads
                {
                        __syncthreads();
                        // TODO: Replace mod with something more efficient (?)
                        if (threadIdx.x % (int)pow(2, j+1) == 0 && threadIdx.x < len)
                        {
                                next_idx = curr_idx + pow(2, j);
                                buf[curr_idx] = (dist_vec[buf[curr_idx]] < dist_vec[buf[next_idx]]) ? curr_idx : next_idx;
                        }
                }

                // Only threads with threadIdx.x == 0 does the last reduction, thus there is no need to __synchthreads here.
                if (threadIdx.x == 0)
                {
                        // TODO: Recover the threadIdx.x from buf[0] (for each query point)

                        // Store the best k nearest neighbors calculated by this thread to the global 3d tensor
                        // of size [QUERY_BLOCK_SIZE, gridDim.x, k].
                        // As a reminder we may think about the current size of nn_dist and nn_idx,
                        // which is [QUERY_BLOCK_SIZE, (len * gridDim.x) / k, k], 
                        // where (len * gridDim.x) / k -> the number of thread blocks we used on the previous iteration.
                        // len * gridDim.x -> the number of elements stored on the previous call of this kernel
                        // (len * gridDim.x) / k -> number of thread blocks that wrote those elements on the previous kernel call, 
                        // since each wrote k neighbors
                        
			global_nn_dist[(threadIdx.y * gridDim.x + blockIdx.x) * k + neigh] =
                                                dist_vec[buf[curr_idx]];
			
			global_nn_idx[(threadIdx.y * gridDim.x + blockIdx.x) * k + neigh] =  
                                                blockIdx.x * blockDim.x + (buf[curr_idx] - threadIdx.y * blockDim.x);
			
			dist_vec[buf[curr_idx]] = INF;
                }
	}
}

__global__ void predict_query_values(double *dev_ydata, double *dev_query_ydata, int *dev_nn_idx, int query_block_start, int k, double *dev_sse, double *dev_err)
{
	// double neigh_vals[NNBS];
        double sum = 0.0;
        double yp;

	// if(tid <)
        // each thread runs for a query (thus the global thread id is equal to the query id inside the quey block)
	int tid = threadIdx.x; // running with a 1D Thread Block
        int query_idx = query_block_start + tid;

	for(int i = 0; i < k; i++)
		sum += dev_ydata[dev_nn_idx[query_idx * k + i]]; // TODO : tid seems wrong
                
	// call predict_value
	yp = sum / k;

	// compute error metrics
	dev_sse[query_idx] = (dev_query_ydata[query_idx] - yp) * (dev_query_ydata[query_idx] - yp);
        dev_err[query_idx] = 100.0 * fabs((yp - dev_query_ydata[query_idx]) / dev_query_ydata[query_idx]);
        // printf("Thread %d computes for query %d | err = %.5f | sse = %.5f\n", tid, query_idx, dev_err[query_idx], dev_sse[query_idx]);
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
        FILE *fpout = fopen("output.knn.txt","w");
#endif
        int vector_size = PROBDIM + 1;
	double *dev_mem, *dev_ydata, *dev_query_ydata, *dev_query_mem, *dev_nn_dist, *dev_sse, *dev_err;
	int *dev_nn_idx;
	
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

	// ******************************************************************
	// ************************** Load Data *****************************
	// ******************************************************************
	load_binary_data(trainfile, mem, TRAINELEMS * (PROBDIM + 1));
	load_binary_data(queryfile, query_mem, QUERYELEMS * vector_size);

	extract_vectors(mem, train_buf, TRAINELEMS, PROBDIM + 1, PROBDIM);

	// construct a "pure" query elements array to pass to the device
        // TODO: Have the same in/out buffer -> overwriting
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
	// cudaMalloc((void**)&dev_nn_dist, ROW_THREAD_BLOCKS * (QUERY_BLOCK_SIZE * NNBS) * sizeof(double));
	// cudaMalloc((void**)&dev_nn_idx, ROW_THREAD_BLOCKS * (QUERY_BLOCK_SIZE * NNBS) * sizeof(int));
	
	cudaMalloc((void**)&dev_nn_dist, TRAINELEMS * QUERY_BLOCK_SIZE * sizeof(double));
	cudaMalloc((void**)&dev_nn_idx, TRAINELEMS * QUERY_BLOCK_SIZE * sizeof(int));

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
	 *   
	 *   - a queryel_block -> QUERY_BLOCK_SIZE * PROBDIM * sizeof(double)
	 *
	 *   - a dist_vector -> QUERY_BLOCK_SIZE * TRAIN_BLOCK_SIZE * sizeof(double)
	 */

	assert(block_size.x % 2 == 0);
	float num_thread_blocks;
        int len;
        size_t reduction_shared_mem_size;
        double sse = 0.0f, err_sum = 0.0f;
	/* COMPUTATION PART */
        double t_start = gettime();
        // int i = 0;
	for(int i = 0; i < QUERYELEMS; i += QUERY_BLOCK_SIZE)
	{
        	compute_distances_kernel<<<grid_dim, block_size, shared_mem_size>>>(dev_mem, dev_query_mem, i, 
		 					dev_nn_idx, dev_nn_dist, NNBS, PROBDIM, trainel_block_size, queryel_block_size);  // compute a "chunk" of rows
                // Check for any cuda errors you might be missing
                // printf("compute_distances_kernel error code: %d\n", cudaGetLastError());
                assert(cudaGetLastError() == 0);
		
		/*
		num_thread_blocks = ROW_THREAD_BLOCKS * NNBS / (2 * block_size.x);
		// num_thread_blocks = TRAINELEMS/TRAIN_BLOCK_SIZE * NNBS / (2 * block_size.x);
		// 4 * num_thread_blocks = ROW_THREAD_BLOCKS
		// 4 * TRAINELEMS/TRAIN_BLOCK_SIZE * NNBS / (2 * block_size.x) = ROW_THREAD_BLOCKS
		// 4 * NNBS / (2 * block_size.x) = 1
		// 2 * NNBS / block_size.x = 1
		// 2 * 32 / TRAIN_BLOCK_SIZE = 1
		// TRAIN_BLOCK_SIZE = 64 ισχύει
		dim3 reduction_block_size(block_size.x, QUERY_BLOCK_SIZE, 1);
		while(1)
		{
			// Calculate the amount of elements in each thread block (per query)
			len = num_thread_blocks * (2 * block_size.x) / ceil(num_thread_blocks);

			// Fix the num_thread_blocks, so if there are less than block_size.x elements left
			// you will still use 1 thread block in order to reduce them to NNBS neighbors
			num_thread_blocks = ceil(num_thread_blocks);

			// Calculate the number of thread blocks required
			// TODO: What if on the first iteration len <= NNBS
			dim3 reduction_grid_dim(num_thread_blocks, 1, 1);

			// For each query we will load in the shared memory 2*block_size.x nearest neighbors (i.e. their distances and idx)
			// The shared memory will be structured as:
			// [query0 distances][query1 distances]...[query0 indexes][query1 indexes]...
			// Calculate the shared memory size
			// (REMINDER : len takes into account that we will require 2*blockDim.x distances for the first iteration)
			size_t reduction_dist_vector_size = QUERY_BLOCK_SIZE * len * sizeof(double);
			size_t reduction_idx_vector_size = QUERY_BLOCK_SIZE * len * sizeof(int);
			reduction_shared_mem_size = reduction_dist_vector_size + reduction_idx_vector_size;

			reduce_distance_kernel<<<reduction_grid_dim, reduction_block_size, reduction_shared_mem_size>>>
				(dev_nn_idx, dev_nn_dist, len, NNBS, reduction_dist_vector_size);

			// Check for any cuda errors you might be missing
			// printf("reduce_distance_kernel error code: %d\n", cudaGetLastError());
			assert(cudaGetLastError() == 0);

			printf("num_thread_blocks = %.1f | len = %d\n", num_thread_blocks, len);
			// update control variable
			if (num_thread_blocks == 1)
				break;
			num_thread_blocks = num_thread_blocks * NNBS / (2 * block_size.x);
		}

		// Find yp for each query and error metrics
		predict_query_values<<<1, QUERY_BLOCK_SIZE>>>(dev_ydata, dev_query_ydata, dev_nn_idx, i, NNBS, dev_sse, dev_err);
		// Check for any cuda errors you might be missing
		assert(cudaGetLastError() == 0);
		*/
                int num_neigh_per_query = ROW_THREAD_BLOCKS * NNBS * 2;
		cudaMemcpy(temp_dist, dev_nn_dist, num_neigh_per_query * QUERY_BLOCK_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(temp_idx, dev_nn_idx, num_neigh_per_query * QUERY_BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
                int pos;
		double yp = 0;
                double sum = 0.0;
                double dist;
		for (int q = 0; q < QUERY_BLOCK_SIZE; q++) // for each dist matrix row
                {
                        sum = 0;
			for (int neigh = 0; neigh < NNBS; neigh++) // find all neighbors
			{
				dist = compute_min_pos(&temp_dist[q*num_neigh_per_query], num_neigh_per_query, &pos);
                                // printf("neigh[%d] has a distance of %.5f\n", neigh, dist);
                                // printf("neigh[%d]_ydata : %.5f, expected : %.5f\n", neigh, ydata[pos], query_ydata[i + q]);
				temp_dist[q*(num_neigh_per_query) + pos] = INF;
				sum += ydata[temp_idx[q*(num_neigh_per_query) + pos]];
			}
                        
                        yp = sum / NNBS;
			// printf("Found : %.5f, expected : %.5f\n", yp, query_ydata[i + q]);
			sse += (query_ydata[i + q] - yp) * (query_ydata[i + q] - yp);
			// printf("sse = %.5f\n", sse);
                        err_sum += 100.0 * fabs((yp - query_ydata[i + q]) / query_ydata[i + q]);
                }
	}

	// sync device and host before getting final time
	cudaDeviceSynchronize();
        double t_sum = gettime() - t_start;
/*
	double sse = 0.0f, err_sum = 0.0f;
	cudaMemcpy(buf, dev_sse, QUERYELEMS * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < QUERYELEMS; i++)
		sse += buf[i];

	cudaMemcpy(buf, dev_err, QUERYELEMS * sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < QUERYELEMS; i++)
		err_sum += buf[i];
*/
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
	cudaFree(dev_sse);
	cudaFree(dev_err);

	return 0;
}
