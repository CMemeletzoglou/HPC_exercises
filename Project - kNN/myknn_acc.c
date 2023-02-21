#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_acc.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

// #define TRAIN_BLOCK_SIZE 64
// #define QUERY_BLOCK_SIZE 16

// double find_knn_value(double *p, int n, int knn)	
double find_knn_value(query_t *restrict q, int knn)
{
	double sum_v = 0.0;
	#pragma acc loop seq
	for (int i = 0; i < knn; i++)
		sum_v += q->nn_val[i];

	return sum_v / knn;
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
        
	// assert(TRAINELEMS % TRAIN_BLOCK_SIZE == 0);

#if defined(DEBUG)
	/* Create/Open an output file */
	FILE *fpout = fopen("output.knn.txt","w");
#endif
        int vector_size = PROBDIM + 1;

        double *mem = (double *)malloc(TRAINELEMS * vector_size * sizeof(double));
	double *ydata = (double *)malloc(TRAINELEMS * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * vector_size * sizeof(double));
	query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));

	load_binary_data(trainfile, mem, NULL, TRAINELEMS*(PROBDIM+1));
	load_binary_data(queryfile, query_mem, queries, QUERYELEMS * vector_size);

	/* Configure and Initialize the ydata handler arrays */
	double *query_ydata = malloc(QUERYELEMS * sizeof(double));
	
	for (int i = 0; i < TRAINELEMS; i++)
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

        double *train_buf = (double *)malloc(TRAINELEMS * PROBDIM * sizeof(double));
        double *query_buf = (double *)malloc(QUERYELEMS * PROBDIM * sizeof(double));

        // extract pure vectors to be passed to the GPU
        extract_vectors(mem, train_buf, TRAINELEMS, PROBDIM + 1, PROBDIM);
	extract_vectors(query_mem, query_buf, QUERYELEMS, PROBDIM + 1, PROBDIM);
	
	/* COMPUTATION PART */
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	/* For each training elements block, we calculate each query point's k neighbors,
	 * using the training elements, that belong to the current training element block.
	 * The calculation of each query point's neighbors, occurs inside compute_knn_brute_force.
	 */

	double *global_nn_dist = (double *)malloc(QUERYELEMS * TRAINELEMS * sizeof(double));

	double t_start = gettime();


        #pragma acc data copyin(train_buf[:TRAINELEMS * PROBDIM], query_buf[:QUERYELEMS * PROBDIM], ydata[:TRAINELEMS], \
                                query_ydata[:QUERYELEMS], queries[:QUERYELEMS]) create(global_nn_dist[:QUERYELEMS * TRAINELEMS]) \
				copyout(sse, err_sum)			
        {
		// construct global distance matrix
		#pragma acc parallel loop collapse(2)
		for (int i = 0; i < QUERYELEMS; i++)
			for (int train_el = 0; train_el < TRAINELEMS; train_el++) // for each training element
				global_nn_dist[i * TRAINELEMS + train_el] = compute_dist(&(query_buf[i*PROBDIM]), &(train_buf[train_el * PROBDIM]), PROBDIM);
				// global_nn_dist[i * TRAINELEMS + train_el] = compute_dist(queries[i].x, &(train_buf[train_el * PROBDIM]), PROBDIM);
		
		// reduce the length of its rows
		int pos;
		double min_k_val;
		#pragma acc parallel loop private(pos, min_k_val)
		for (int query_idx = 0; query_idx < QUERYELEMS; query_idx++)
		{
			for (int neigh = 0; neigh < NNBS; neigh++)
			{
				min_k_val = compute_min_pos(global_nn_dist + query_idx * TRAINELEMS, TRAINELEMS, &pos);
				queries[query_idx].nn_dist[neigh] = min_k_val;
				queries[query_idx].nn_idx[neigh] = pos;
				queries[query_idx].nn_val[neigh] = ydata[pos];

				global_nn_dist[query_idx * TRAINELEMS + pos] = INF;
			}
		}
		
		// calculate query values
		#pragma acc parallel loop private(err, err_sum, sse) reduction(+ : err_sum, sse)
		for (int i = 0; i < QUERYELEMS; i++)
		{
			double yp = find_knn_value(&(queries[i]), NNBS);
			sse += (query_ydata[i] - yp) * (query_ydata[i] - yp);
			err_sum += 100.0 * fabs((yp - query_ydata[i]) / query_ydata[i]);
		}
		t_sum = gettime() - t_start;
	}


	// #if defined(DEBUG)
	// 		fprintf(fpout,"%.5f %.5f %.2f\n", query_ydata[i], yp, err);
	// #endif

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

	free(queries);
	free(query_ydata);
	free(query_mem);

	free(ydata);
	free(mem);

        free(train_buf);
        free(query_buf);
	free(global_nn_dist);

	return 0;
}
