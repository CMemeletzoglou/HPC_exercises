#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_mpi.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

// My L1d = 192 KiB, and I want to cache the maximum amount of training points.
// Each training point has a size of: PROBDIM * sizeof(double) = 16 * 8 = 128 bytes.
// Thus, in L1d I may preserve in L1d cache 192,000 / 128 = 1,500 training points simultaneously.
// I also need to be able to store in cache the query point aswell (!!)
// and have a block size that will evenly devide TRAINELEMS.
// The easy solution is to get the max power of 2 that is less that 1,500,
// since TRAINELEMS is also a power of 2.
// #define train_block_size 128

// TODO : maybe allocate these inside main and pass them as args to find_knn_value (?)
static double **xdata;
static double ydata[TRAINELEMS];

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
		fd[i] = ydata[q->nn_idx[i]];

	for (int i = 0; i < knn; i++) 
		for (int j = 0; j < PROBDIM; j++)
			xd[i * PROBDIM + j] = xdata[q->nn_idx[i]][j];

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
        double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double));
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t)); 

        // read a part of training data
        load_binary_data_mpi(trainfile, mem, NULL, trainelems_chunk, trainelem_offset);

        // read all of the query data
        load_binary_data_mpi(queryfile, query_mem, queries, QUERYELEMS * vector_size, 0);

#if defined(DEBUG)
	/* Create/Open an output file */
	// FILE *fpout = fopen("output.knn_mpi.txt","w");
	char *fout = "output.knn_mpi.txt";
#endif
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
	copy_to_aligned(mem, xdata, (PROBDIM+1), PROBDIM, TRAINELEMS);
#else
	// Assign to the handler arrays, pointers to the already allocated mem
	for (int i = 0; i < local_ntrainelems; i++)
		xdata[i] = &mem[i*(PROBDIM + 1)];
#endif

	/* Configure and Initialize the ydata handler arrays */
	double *query_ydata = malloc(QUERYELEMS * sizeof(double));
	
	for (int i = 0; i < local_ntrainelems; i++)
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

	// assert(TRAINELEMS % train_block_size == 0);
	// assert(TRAINELEMS % local_ntrainelems == 0);

	/* COMPUTATION PART */
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	int queryelems_blocksize = QUERYELEMS / nprocs;
	int rank_in_charge;
	MPI_Request request = MPI_REQUEST_NULL;
	MPI_Status status;

	int first_query = rank * queryelems_blocksize;
	int last_query = (rank + 1) * queryelems_blocksize - 1;
	
	if (rank == nprocs - 1)
		last_query = QUERYELEMS - 1;
	
	// receive buffer for a single query
	// I will need enough space to store all query_t structs sent by every other rank.
	query_t *rcv_buf = (query_t *)malloc((nprocs-1)*sizeof(query_t));

	// int global_train_offset = rank * local_ntrainelems;
	int global_train_offset = trainelem_offset;
	/* Each rank is responsible for calculating the k neighbors of each query point,
	 * using only the training elements block it has been assigned. The block's boundaries are defined as:
	 * start = rank * local_ntrainelems  (i.e. global_train_offset) // NOT entirely correct...
	 * end = (rank + 1) * local_ntrainelems .
	 * The calculation of each query point's neighbors, occurs inside compute_knn_brute_force.
	 *
	 * Within each block, each rank is responsible for :
	 * a) Calculating the k neighbors of all query points
	 * b) Sending each query that it is not responsible for, to the correct rank.
	 * c) Gathering collections of k neighbors for the subset of query points defined by [query_chunk_start, query_chunk_end].
	 *    These collections are calculated (step a) sent (step b) by the other ranks (i.e. from other training elements blocks).
	 * d) Calculating the final k nearest neighbors, using the collections gathered at step (b). (reduction)
	 */

	t0 = gettime();
	// (a) and (b) Calculate and send the k neighbors found in the training block for **all** query points.
	for (int i = 0; i < QUERYELEMS; i++)
	{
		compute_knn_brute_force_mpi(xdata, &(queries[i]), PROBDIM, NNBS, global_train_offset, local_ntrainelems);
		rank_in_charge = get_rank_in_charge_of(i, queryelems_blocksize, nprocs);
		if (rank_in_charge != rank)
			MPI_Isend(&(queries[i]), 1*sizeof(query_t), MPI_BYTE, rank_in_charge, i, MPI_COMM_WORLD, &request);
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
			MPI_Recv(&(rcv_buf[rcv_buf_offset++]), 1*sizeof(query_t), MPI_BYTE, j, i, MPI_COMM_WORLD, &status);
		}
	
		// (d) Update the k neighbors of each query points under our control using the data we received from the other ranks.
		reduce_in_struct(&(queries[i]), rcv_buf, nprocs);
	
	}

	/* TODO: DELETE MEEEEE */
	if (rank == 0)
		printf("You may have forgot to remove exit(0)!!\n");
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	exit(0);
	
	t1 = gettime();
	t_sum = t1 - t0;
	for (int i = 0; i < QUERYELEMS; i++)
	{
		t0 = gettime();
		double yp = find_knn_value(&(queries[i]), NNBS);
		t1 = gettime();
		t_sum += t1 - t0;
		
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
	for (int i = 0; i < TRAINELEMS; i++)
		free(xdata[i]);
#endif
	free(xdata);
	free(mem);
	free(query_ydata);
	free(query_mem);
	free(queries);

	return 0;
}
