#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_mpi.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

// TODO : maybe allocate these inside main and pass them as args to find_knn_value (?)
static double **xdata;
static double *ydata; // this must be changed

// double find_knn_value(double *p, int n, int knn)
double find_knn_value(query_t *q, int knn)
{
	double xd[knn * PROBDIM];    // the knn neighboring points/vectors of size PROBDIM

#if defined(SIMD)
	__attribute__((aligned(32))) double fd[knn];	// function values for the knn neighbors
#else 
	double fd[knn];
#endif	

	for (int i = 0; i < knn; i++)
		fd[i] = q->nn_val[i];
		// fd[i] = ydata[q->nn_idx[i]];
// TODO: Remove xd, since it is not used in predict value
//      for (int i = 0; i < knn; i++)
// 		for (int j = 0; j < PROBDIM; j++)
// 			xd[i * PROBDIM + j] = xdata[q->nn_idx[i]][j];

	return predict_value(PROBDIM, knn, xd, fd);
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

        // MPI Init
        int rank, nprocs;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        
        int vector_size = PROBDIM + 1;
        int local_ntrainelems = TRAINELEMS / nprocs;
        int trainelem_offset = rank * local_ntrainelems * vector_size;
        
        // correction for the last process
        if(rank == nprocs - 1)
                local_ntrainelems += TRAINELEMS % nprocs;

	int trainelems_chunk = local_ntrainelems * vector_size; // chunk size in scalar elements (i.e. doubles)

	double *mem = (double *)malloc(trainelems_chunk * sizeof(double));
	ydata = (double *)malloc(local_ntrainelems * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double));
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t)); 

        // read a part of training data
        load_binary_data_mpi(trainfile, mem, NULL, trainelems_chunk, trainelem_offset);

        // read all of the query data
        load_binary_data_mpi(queryfile, query_mem, queries, QUERYELEMS * vector_size, 0);

	/* Create handler arrays that will be used to separate xdata's PROBDIM vectors
	 * and the corresponding surrogate values, since we never need both
	 * in order to perform a computation.
	 * We either going to use the xdata of two points (ex. when calculating distance from one another)
	 * or use ydata (surrogates) (ex. when predicting the value of a query point)
	 */
	xdata = (double **)malloc(local_ntrainelems * sizeof(double*));

#if defined(SIMD)
	int posix_res;
	// Allocate new memory for the handler arrays, so that it is aligned and copy the data there
	// Align each xdata[i] to a 32 byte boundary so you may later use SIMD
	for (int i = 0; i < local_ntrainelems; i++)
	{
		posix_res = posix_memalign((void **)(&(xdata[i])), 32, PROBDIM * sizeof(double));
		assert(posix_res == 0);
	}
	copy_to_aligned(mem, xdata, vector_size, PROBDIM, TRAINELEMS);
#else
	// Assign to the handler arrays, pointers to the already allocated mem
	for (int i = 0; i < local_ntrainelems; i++)
		xdata[i] = &mem[i * vector_size];
#endif

	/* Configure and Initialize the ydata handler arrays */
	double *query_ydata = malloc(QUERYELEMS * sizeof(double));
	
	for (int i = 0; i < local_ntrainelems; i++)
	{
#if defined(SURROGATES)
		ydata[i] = mem[i * vector_size + PROBDIM];
#else
		ydata[i] = 0;
#endif
	}

	for (int i = 0; i < QUERYELEMS; i++)
	{
#if defined(SURROGATES)
		query_ydata[i] = query_mem[i * vector_size + PROBDIM];
#else
		query_ydata[i] = 0;
#endif
	}

	// assert(TRAINELEMS % train_block_size == 0);
	// assert(TRAINELEMS % local_ntrainelems == 0);

	/* COMPUTATION PART */
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err_sum = 0.0;

	int queryelems_blocksize = QUERYELEMS / nprocs;
	int rank_in_charge;
	MPI_Request request = MPI_REQUEST_NULL;
	MPI_Status status;

	int first_query = rank * queryelems_blocksize;
	int last_query = (rank + 1) * queryelems_blocksize - 1;

	if (rank == nprocs - 1)
		last_query = QUERYELEMS - 1;
	
	// Need enough space to store all query_t structs sent by every other rank.
	query_t *rcv_buf = (query_t *)malloc((nprocs - 1) * sizeof(query_t));

	int global_block_offset = rank * local_ntrainelems;
	/* Each rank is responsible for calculating the k neighbors of each query point,
	 * using only the training elements block it has been assigned. The block's boundaries are defined as:
	 * start = rank * local_ntrainelems * vector_size (i.e. global_train_offset) // NOT entirely correct...
	 * end = (rank + 1) * local_ntrainelems * vector_size .
	 * The calculation of each query point's neighbors, occurs inside compute_knn_brute_force.
	 *
	 * Within each block, each rank is responsible for :
	 * a) Calculating the k neighbors of **all** query points
	 * b) Sending each query that it is not responsible for, to the correct rank.
	 * c) Gathering collections of k neighbors for the subset of query points defined by [query_chunk_start, query_chunk_end].
	 *    These collections are calculated (step a) and sent (step b) by the other ranks (i.e. from other training elements blocks).
	 * d) Calculating the final k nearest neighbors, using the collections gathered at step (c). (reduction)
	 */

	t0 = gettime();

	// (a) and (b) Calculate and send the k neighbors found in the training block for **all** query points.

	// TODO: one more loop here for the case of "cache blocking"
	for (int i = 0; i < QUERYELEMS; i++)
	{
		compute_knn_brute_force(xdata, ydata, &(queries[i]), PROBDIM, NNBS, global_block_offset, 0, local_ntrainelems);

		rank_in_charge = get_rank_in_charge_of(i, queryelems_blocksize, nprocs);
		if (rank_in_charge != rank)
			MPI_Isend(&(queries[i]), 1 * sizeof(query_t), MPI_BYTE, rank_in_charge, i, MPI_COMM_WORLD, &request);
	}

	int rcv_buf_offset = 0;
	for (int i = first_query; i <= last_query; i++)
	{
		// (c) Gather the collections of k nearest neighbors sent by the other ranks.
		rcv_buf_offset = 0;
		for (int j = 0; j < nprocs; j++)
		{
			if (j == rank)
				continue;
			MPI_Recv(&(rcv_buf[rcv_buf_offset++]), 1 * sizeof(query_t), MPI_BYTE, j, i, MPI_COMM_WORLD, &status);
		}

		// (d) Update the k neighbors of each query points under our control using the data we received from the other ranks.
		reduce_in_struct(&(queries[i]), rcv_buf, nprocs-1);	
	}
	t1 = gettime();
	t_sum = t1 - t0;
        
        // Initialize environment for metric calculations
#if defined(DEBUG)
        // Preserve yp, err for each query, so you may later write the output file
        char *filename = "output.knn_mpi.txt";

        // Open and initialize the ouput file for all ranks
        MPI_File f;
        MPI_Offset base;
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
        MPI_File_get_position(f, &base);

	double yp[last_query - first_query + 1], err[last_query - first_query + 1];
	int local_idx = 0;
#else
        // Only care about current err and yp, since err will be accumulated in err_sum, and yp will be used to calculate the err
        // and sse and thus in next iteration may be overwritten
	double yp, err;
#endif

        // Calculate yp and the errors/metrics for all queries under the rank's responsibility
        // Should preserve the values in DEBUG mode.
	for (int i = first_query; i <= last_query; i++)
	{
		t0 = gettime();
#if defined(DEBUG)
		yp[local_idx] = find_knn_value(&(queries[i]), NNBS);
#else
		yp = find_knn_value(&(queries[i]), NNBS);
#endif
		t1 = gettime();
		t_sum += t1 - t0;

#if defined(DEBUG)
		sse += (query_ydata[i] - yp[local_idx]) * (query_ydata[i] - yp[local_idx]);
		err[local_idx] = 100.0 * fabs((yp[local_idx] - query_ydata[i]) / query_ydata[i]);
		err_sum += err[local_idx];		
		local_idx++;
#else
		sse += (query_ydata[i] - yp) * (query_ydata[i] - yp);
		err = 100.0 * fabs((yp - query_ydata[i]) / query_ydata[i]);
#endif
	}

#if defined(DEBUG)
        // Write the output file
	int curr_char_offset = 0, global_char_offset = 0;

        const int max_len = 30;
        char buf[max_len*(last_query - first_query + 1)];
        
        // Collect all local data you want to write to the file into a char buffer
        for (int i = first_query; i <= last_query; i++)
        {
                local_idx = i - first_query;
                snprintf(buf + curr_char_offset, max_len, "%.5f %.5f %.2f\n", query_ydata[i], yp[local_idx], err[local_idx]);
                curr_char_offset = strlen(buf);
        }

        // At this point each rank has determined the amount of bytes it needs to write in the output file.
        // Since each rank is in charge of a set of queries in a blocking fashion 
        // (i.e. rank 0 is in charge of queries [0, queryelems_blocksize), rank 1 -> [queryelems_blocksize, 2*queryelems_blocksize - 1], ...).
        // We may Exscan those offsets in order for each rank to get it's global offset (in number of chars), in the shared file.
        MPI_Exscan(&curr_char_offset, &global_char_offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Write data to the output file, using those offsets
        MPI_File_write_at_all(f, base + global_char_offset, buf, strlen(buf), MPI_CHAR, MPI_STATUS_IGNORE);
#endif
        
	/* CALCULATE AND DISPLAY RESULTS */

	// Reduce all metrics to the root rank
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &t_sum, &t_sum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &err_sum, &err_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &sse, &sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	

        if(rank == 0)
        {
                double mse = sse / QUERYELEMS;
                double ymean = compute_mean(query_ydata, QUERYELEMS);
                double var = compute_var(query_ydata, QUERYELEMS, ymean);
                double r2 = 1 - (mse / var);

                printf("Results for %d query points\n", QUERYELEMS);
                printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
                printf("MSE = %.6f\n", mse);
                printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

                printf("Total time = %lf secs\n", t_sum);
                // TODO: Print average time of first query per block
                printf("Average time/query = %lf secs\n", t_sum / QUERYELEMS);
        }

	/* CLEANUP */

#if defined(DEBUG)
	MPI_File_close(&f);
#endif

#if defined(SIMD)
	for (int i = 0; i < local_ntrainelems; i++)
		free(xdata[i]);
#endif
	free(xdata);
	free(mem);
	free(query_ydata);
	free(query_mem);
	free(queries);
	
	free(ydata); // new
	free(rcv_buf);

	MPI_Finalize();
	return 0;
}
