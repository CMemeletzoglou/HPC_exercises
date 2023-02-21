#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_acc.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

#define TRAIN_BLOCK_SIZE 64 // 2^14 -> 64 blocks
#define QUERY_BLOCK_SIZE 16

#define NUM_TRAIN_BLOCKS (TRAINELEMS / TRAIN_BLOCK_SIZE)

// double find_knn_value(double *p, int n, int knn)	
double find_knn_value(double *restrict ydata, int knn)
{
	double sum_v = 0.0;
	#pragma acc loop seq
	for (int i = 0; i < knn; i++)
		sum_v += ydata[i]; // change

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

	load_binary_data(trainfile, mem, TRAINELEMS*(PROBDIM+1));
	load_binary_data(queryfile, query_mem, QUERYELEMS * vector_size);

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

        double *train_vectors = (double *)malloc(TRAINELEMS * PROBDIM * sizeof(double));
        double *query_vectors = (double *)malloc(QUERYELEMS * PROBDIM * sizeof(double));

        // extract pure vectors to be passed to the GPU
        extract_vectors(mem, train_vectors, TRAINELEMS, PROBDIM + 1, PROBDIM);
	extract_vectors(query_mem, query_vectors, QUERYELEMS, PROBDIM + 1, PROBDIM);
	
	/* COMPUTATION PART */
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	/* For each training elements block, we calculate each query point's k neighbors,
	 * using the training elements, that belong to the current training element block.
	 * The calculation of each query point's neighbors, occurs inside compute_knn_brute_force.
	 */

	double *global_nn_dist = (double *)malloc(TRAINELEMS * sizeof(double));
	int *global_nn_idx = (int *)malloc(TRAINELEMS * TRAIN_BLOCK_SIZE * sizeof(int));

	int *reduced_nn_idx = (int*)malloc(NNBS * NUM_TRAIN_BLOCKS * sizeof(int));
	double *reduced_nn_dist = (double *)malloc(NNBS * NUM_TRAIN_BLOCKS * sizeof(double));
        
        #pragma acc data copyin(train_vectors[:TRAINELEMS * PROBDIM], query_vectors[:QUERYELEMS * PROBDIM], ydata[:TRAINELEMS], query_ydata[:QUERYELEMS]) \
			 create(global_nn_dist[:TRAINELEMS], global_nn_idx[:TRAINELEMS]) \
			 create(reduced_nn_dist[:NNBS*NUM_TRAIN_BLOCKS], reduced_nn_idx[:NNBS*NUM_TRAIN_BLOCKS]) \
                         copyout(sse, err_sum)
        {
		double t_start = gettime();

		for (int i = 0; i < QUERYELEMS; i++) // choose query
		{	
                        #pragma acc parallel loop collapse(2)
			for(int train_block = 0; train_block < TRAINELEMS; train_block += TRAIN_BLOCK_SIZE)
			{
				// calculate distances for the chosen query point with each training point in this block
                                for (int train_el = 0; train_el < TRAIN_BLOCK_SIZE; train_el++)
                                {
                                        int g_train_el_idx = train_block + train_el;
                                        global_nn_dist[g_train_el_idx] = compute_dist(&(query_vectors[i * PROBDIM]), &(train_vectors[g_train_el_idx * PROBDIM]), PROBDIM);
					global_nn_idx[g_train_el_idx] = g_train_el_idx;
				}
			}
			// REDUCE
			// This has a race condition and thus cannot be parallelized
			
			#pragma acc parallel loop
			for(int train_block = 0; train_block < TRAINELEMS; train_block += TRAIN_BLOCK_SIZE)
                        {
                                int train_block_idx = train_block / TRAIN_BLOCK_SIZE;
                                #pragma acc loop seq
				for (int neigh = 0; neigh < NNBS; neigh++)
				{
                                        int pos;
					reduced_nn_dist[train_block_idx*NNBS + neigh] = compute_min_pos(&global_nn_dist[train_block], TRAIN_BLOCK_SIZE, &pos);
                                        reduced_nn_idx[train_block_idx*NNBS + neigh] = train_block + pos;
                                        global_nn_dist[train_block + pos] = INF; // the race condition
				}
			}
			
                        // 2nd stage reduction
                        // #pragma acc loop seq
			#pragma acc parallel num_gangs(1) vector_length(1) present(ydata[:TRAINELEMS], reduced_nn_idx[:NNBS * NUM_TRAIN_BLOCKS], \
											reduced_nn_dist[:NNBS * NUM_TRAIN_BLOCKS])
                        {
                                int pos;
                                double sum = 0.0, yp;
				#pragma acc loop seq
                                for (int neigh = 0; neigh < NNBS; neigh++)
                                {
                                        compute_min_pos(reduced_nn_dist, NUM_TRAIN_BLOCKS*NNBS, &pos);
                                        sum += ydata[reduced_nn_idx[pos]];
                                        reduced_nn_dist[pos] = INF;
                                }
                                yp = sum / NNBS;
                                sse += (query_ydata[i] - yp) * (query_ydata[i] - yp);
                                err_sum += 100.0 * fabs((yp - query_ydata[i]) / query_ydata[i]);
                        }
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

	free(query_ydata);
	free(query_mem);

	free(ydata);
	free(mem);

        free(train_vectors);
        free(query_vectors);
	free(global_nn_idx);
	free(global_nn_dist);

	free(reduced_nn_idx);
	free(reduced_nn_dist);

	return 0;
}
