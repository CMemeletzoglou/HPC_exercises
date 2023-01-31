#include <mpi.h>
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
	double fd[knn];	      	      // function values for the knn neighbors

	for (int i = 0; i < knn; i++)
		fd[i] = ydata[nn_x[i]];

	for (int i = 0; i < knn; i++) 
		for (int j = 0; j < PROBDIM; j++)
			xd[i * PROBDIM + j] = xdata[nn_x[i]][j];

	return predict_value(PROBDIM, knn, xd, fd, p, nn_d);
}

int main(int argc, char *argv[])
{
        if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

        // MPI Init
        int rank, nprocs;
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        
	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *mem = (double *)malloc(TRAINELEMS * (PROBDIM + 1) * sizeof(double));
	xdata = (double **)malloc(TRAINELEMS * sizeof(double *));

        // each MPI rank will have only some of the query elements
        int local_qelems = QUERYELEMS / nprocs;

        // correction for the last process
        if(rank == nprocs - 1)
                local_qelems += QUERYELEMS % nprocs;

        int chunk_size = QUERYELEMS / nprocs;

        // allocate space for the rank's query points
        double *query_mem = (double *)malloc(local_qelems * (PROBDIM + 1) * sizeof(double));

        load_binary_data(trainfile, mem, TRAINELEMS * (PROBDIM + 1));

        for (int i = 0; i < TRAINELEMS; i++)
        {
		xdata[i] = mem + i * (PROBDIM + 1); //&mem[i*PROBDIM];
#if defined(SURROGATES)
		ydata[i] = mem[i * (PROBDIM + 1) + PROBDIM];
#else
		ydata[i] = 0;
#endif
        }

        // load the local_qelems query points for each rank, in an MPI I/O way
        // load_binary_data_mpi(queryfile, query_mem, rank, local_qelems * (PROBDIM + 1));
        load_binary_data_mpi(queryfile, query_mem, QUERYELEMS, PROBDIM, chunk_size * (PROBDIM + 1));

#if defined(DEBUG)
        MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, "output.knn_mpi.txt", MPI_MODE_WRONLY, MPI_INFO_NULL, &f);        
#endif
        // allocate space for the target function values for the rank's query points
	double *y = malloc(local_qelems * sizeof(double));
        // double local_ysum = 0.0f;

        for (int i = 0; i < local_qelems; i++)
        {
#if defined(SURROGATES)
                y[i] = query_mem[i * (PROBDIM + 1) + PROBDIM];
#else
                y[i] = 0.0;
#endif
                // local_ysum += y[i];
        }

        double t0, t1, t_start, t_end, t_first = 0.0f, t_sum = 0.0f;
        double local_sse = 0.0f, total_sse = 0.0f;
        double local_err = 0.0f, total_err = 0.0f;
        double total_r2;

        if(rank == 0)
                t_start = gettime();

        for (int i = 0; i < local_qelems; i++)
	{	/* requests */
		t0 = gettime();
		double yp = find_knn_value(&query_mem[i * (PROBDIM + 1)], PROBDIM, NNBS);
		t1 = gettime();

                t_sum += (t1 - t0);

                // if (i == 0)
                if (((rank * local_qelems) + i) == 0)
			t_first = (t1-t0);

		local_sse += (y[i] - yp) * (y[i] - yp);

// #if defined(DEBUG)
// 		for (int k = 0; k < PROBDIM; k++)
// 			fprintf(fpout, "%.5f ", query_mem[i * (PROBDIM + 1) + k]);
// #endif

		local_err += 100.0 * fabs((yp - y[i]) / y[i]);
// #if defined(DEBUG)
// 		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
// #endif
		// err_sum += err;
	}

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allreduce(&local_sse, &total_sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // reduce the local sse vars        
        double mse = total_sse / QUERYELEMS;

        /* The problem here is that some error metrics used below, cannot be calculated from the 
         * global query array pieces, that each rank has. For example, while there is a workaround
         * for the calculation of ymean, using the helper variable local_ysum (see line 98),
         * there is no such workaround for the calculation of var, where the mean must be known.
         * So at this point, we can either run each rank's query vector and calculate the sum
         * of the squared differences of each element from the vector's mean, then reduce them
         * and them calculate the total var
         * 
         * or
         * 
         * gather the query vector into rank 0, and perform the calculation there..
         * This gather operation seems inevitable...
         */
        double *query_buf = NULL;
        int *rcounts = NULL;
        int *displs = NULL;
        if (rank == 0)
        {
                query_buf = (double *)malloc(QUERYELEMS * (PROBDIM + 1) * sizeof(double));
                rcounts = (int *)malloc(nprocs * sizeof(int));
                displs = (int *)malloc(nprocs * sizeof(int));

                for (int i = 0; i < nprocs; i++)
                {
                        rcounts[i] = local_qelems;
                        displs[i] = 0;
                }
                
        }

        MPI_Gatherv(y, local_qelems, MPI_DOUBLE, query_buf, rcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if(rank == 0) // only rank 0 prints
        {
                t_end = gettime();

                double t_sum = t_end - t_start;

                double ymean = compute_mean(query_buf, QUERYELEMS);
                printf("ymean = %.5f\n", ymean);


                printf("Results for %d query points\n", QUERYELEMS);
                printf("APE = %.2f %%\n", total_err / QUERYELEMS);
                printf("MSE = %.6f\n", mse);
                printf("R2 = 1 - (MSE/Var) = %.6lf\n", total_r2);

                printf("Total time = %lf secs\n", t_sum);
                printf("Time for 1st query = %lf secs\n", t_first);
                printf("Time for 2..N queries = %lf secs\n", t_sum - t_first);
                printf("Average time/query = %lf secs\n", (t_sum - t_first) / (QUERYELEMS - 1));
        }

	free(mem);
	free(xdata);
	free(query_mem);
	free(y);

        free(query_buf);
        free(rcounts);

        MPI_Finalize(); // destroy/finalize the MPI environment

        return 0;
}
