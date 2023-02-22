#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "func_acc.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

// kernel 2 (find the 32 best neighbors for each train block) : 
// train_block_size = 2^6  * NNBS -> 2^11          || 2^8 * NNBS -> 2^13        || 2^10 * NNBS -> 2^15          || 2^12 * NNBS -> 2^17

// kernel 3 (find final 32 neighbors):                          
// NUM_TRAIN_BLOCKS = 2^14 * NNBS * NNBS -> 2^24   || 2^8 * NNBS * NNBS -> 2^18 || 2^10 * NNBS * NNBS -> 2^20   || 2^8 * NNBS * NNBS -> 2^18
#define TRAIN_BLOCK_SIZE 8192
#define QUERY_BLOCK_SIZE 64

#define NUM_TRAIN_BLOCKS (TRAINELEMS / TRAIN_BLOCK_SIZE)

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
        // check that the size of the training element block divides the TRAINELEMS training elements
	assert(TRAINELEMS % TRAIN_BLOCK_SIZE == 0); 

	// size of the loaded vectors from the input files (PRBODIM + 1 for the surrogate value)
        int loaded_vector_size = PROBDIM + 1; 

        double *mem = (double *)malloc(TRAINELEMS * loaded_vector_size * sizeof(double));
	double *ydata = (double *)malloc(TRAINELEMS * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * loaded_vector_size * sizeof(double));

	load_binary_data(trainfile, mem, TRAINELEMS*(PROBDIM+1));
	load_binary_data(queryfile, query_mem, QUERYELEMS * loaded_vector_size);

	double *query_ydata = malloc(QUERYELEMS * sizeof(double));

	/***************************************************************************
	 ******************************** Device arrays ****************************
	 ****************************************************************************/
	double *train_vectors = (double *)malloc(TRAINELEMS * PROBDIM * sizeof(double));
        double *query_vectors = (double *)malloc(QUERYELEMS * PROBDIM * sizeof(double));

	double *global_nn_dist = (double *)malloc(QUERY_BLOCK_SIZE * TRAINELEMS * sizeof(double));
	int *global_nn_idx = (int *)malloc(QUERY_BLOCK_SIZE * TRAINELEMS * sizeof(int));

        int *reduced_nn_idx = (int*)malloc(QUERYELEMS * NUM_TRAIN_BLOCKS * NNBS * sizeof(int));
	double *reduced_nn_dist = (double *)malloc(QUERYELEMS * NUM_TRAIN_BLOCKS * NNBS * sizeof(double));

	int helper_vec_size = 0;
	double *err_vals = NULL, *yp_vals = NULL;
#if defined(DEBUG)
	/* Create/Open an output file */
	FILE *fpout = fopen("output.knn_acc.txt","w");
	err_vals = (double *)malloc(QUERYELEMS * sizeof(double));
	yp_vals = (double *)malloc(QUERYELEMS * sizeof(double));
	helper_vec_size = QUERYELEMS;
#endif

	for (int i = 0; i < TRAINELEMS; i++)
	{
#if defined(SURROGATES)
		ydata[i] = mem[i * loaded_vector_size + PROBDIM];
#else
		ydata[i] = 0;
#endif
	}

	for (int i = 0; i < QUERYELEMS; i++)
	{
#if defined(SURROGATES)
		query_ydata[i] = query_mem[i * loaded_vector_size + PROBDIM];
#else
		query_ydata[i] = 0;
#endif
	}

        // extract pure vectors (coordinates without surrogate value) to be passed to the GPU
        extract_vectors(mem, train_vectors, TRAINELEMS, PROBDIM + 1, PROBDIM);
	extract_vectors(query_mem, query_vectors, QUERYELEMS, PROBDIM + 1, PROBDIM);
	
	/* COMPUTATION PART */
	double t0, t1, t_first = 0.0, t_sum = 0.0;
	double sse = 0.0;
	double err, err_sum = 0.0;

	// Explicit ACC data region. Copyin all required data, create the necessary temporary arrays, and copyout the necessary output data
        #pragma acc data copyin(train_vectors[:TRAINELEMS * PROBDIM], query_vectors[:QUERYELEMS * PROBDIM], ydata[:TRAINELEMS], query_ydata[:QUERYELEMS]) \
			 create(global_nn_dist[:TRAINELEMS * QUERY_BLOCK_SIZE], global_nn_idx[:TRAINELEMS * QUERY_BLOCK_SIZE]) \
			 create(reduced_nn_dist[:QUERYELEMS * NUM_TRAIN_BLOCKS * NNBS], reduced_nn_idx[:QUERYELEMS * NUM_TRAIN_BLOCKS * NNBS]) \
	   		 create(err_vals[:helper_vec_size], yp_vals[:helper_vec_size]) \
			 copyout(sse, err_sum, err_vals[:helper_vec_size], yp_vals[:helper_vec_size])
        {
		double t_start = gettime();
		/* We use a two level blocking algorithm : we use a block of query points and a block of training elements 
		 * in order to perform the required calculations.
		 * There are two major reasons supporting this decision :
		 * 	a) By using a block of query elements and by calculating the distances of all its elements, with all
		 * 	   training elements, we avoid computing in parallel all the rows of the (QUERYELEMS x TRAINELEMS)
		 *  	   distance matrix, where the i-th row containts the distances of the i-th query point with all 
		 * 	   training elements. We now only compute QUERY_BLOCK_SIZE rows in parallel. 
		 * 
		 *	   By following, this approach there is no gain in reduced computing time, but in storage space,
		 *	   since using the default problem parameters (2^20 training elements and 2^10 query points),
		 * 	   the (QUERYELEMS x TRAINELEMS) distance matrix, requires ~8.5 GBs of memory, thus it is a
		 * 	   non-scalable solution.
		 * 
		 * 	b) Even if we were to ignore the significant storage needs and to compute the (QUERYELEMS x TRAINELEMS)
		 * 	   distance matrix, we would need to reduce the TRAINELEMS calculated distance values of each row,
		 * 	   into NNBS neighbors. However, the procedure of finding the NNBS neighbors for each query point,
		 * 	   is not parallelizable since it is based on finding the NNBS min values (hence the #pragma acc loop seq
		 * 	   directive on the respective for loops). Therefore, a single thread would need to perform NNBS passes,
		 *         over TRAINELEMS distances, in order to find the NNBS minimum ones.
		 *
		 * 	   But if we use a blocking approach, not only on the query points, but on the training elements as well,
		 * 	   we can perform a two level reduction. Specifically, we compute the distance of each query point in the
		 * 	   query block, with all training elements in each training element block (TRAIN_BLOCK_SIZE distances).
		 * 	   Then, we can reduce the TRAIN_BLOCK_SIZE distances calculated in each training element block, into NNBS.
		 *
		 * 	   This way, if there are N training element blocks, for each query point we will get N * NNBS "candidate
		 * 	   neighbors", which can then be reduced into the final NNBS. By using this approach, we still cannot
		 * 	   run the computation of the NNBS neighbors in parallel, but we reduce the work that needs to done.
		 * 
		 * 	   Now the thread that performs the search of the NNBS neighbors, only needs to perform NNBS passes on 
		 * 	   N * NNBS values, instead of NNBS passes on TRAINELEMS values, which is significantly less.
		 *
		 * Therefore, for each query block, we iterate over all training element blocks, in order to compute
		 * each query point's distances with the training elements in the current training element block.
		 * However, the computation of the distances between the training element blocks, is made in a parallel manner.
		 */
		for (int query_block = 0; query_block < QUERYELEMS; query_block += QUERY_BLOCK_SIZE) 		// choose query block
		{	
		        #pragma acc parallel loop collapse(3)
			for(int train_block = 0; train_block < TRAINELEMS; train_block += TRAIN_BLOCK_SIZE) 	// choose training block
			{
                                for (int query_el = 0; query_el < QUERY_BLOCK_SIZE; query_el++) 		// choose query point
                                {
                                        // calculate distances for the chosen query point with each training point in this block
                                        for (int train_el = 0; train_el < TRAIN_BLOCK_SIZE; train_el++)
                                        {
						int g_query_el_idx = query_block + query_el;	// global query element index
                                                int g_train_el_idx = train_block + train_el;	// global training element index
                                                global_nn_dist[query_el*TRAINELEMS + g_train_el_idx] = 
                                                        compute_dist(&(query_vectors[g_query_el_idx * PROBDIM]), &(train_vectors[g_train_el_idx * PROBDIM]), PROBDIM);
                                                global_nn_idx[query_el*TRAINELEMS + g_train_el_idx] = g_train_el_idx;
                                        }
                                }
			}
                        /* Reduce the TRAINELEMS calculated distances, for each query point, into N * NNBS, where N
			 * is the number of training element blocks.
			 * We use the global_nn_dist and global_nn_idx arrays, as a "scratchpad memory" which stores
			 * all intermediate results. Then, using the data contained in the forementioned arrays,
			 * we store the final NNBS "candindate" neighbors, for each training element block, into
			 * the reduced_nn_dist and reduced_nn_idx arrays.
			 */
                        #pragma acc parallel loop collapse(2)
			for(int train_block = 0; train_block < TRAINELEMS; train_block += TRAIN_BLOCK_SIZE) // choose training block
                        {
                                for (int query_el = 0; query_el < QUERY_BLOCK_SIZE; query_el++)		    // choose query point
				{
					int g_query_el_idx = query_block + query_el;			    // global query point index
                                        int train_block_idx = train_block / TRAIN_BLOCK_SIZE;		    // training element block index
                                        #pragma acc loop seq
                                        for (int neigh = 0; neigh < NNBS; neigh++)
                                        {
                                                int pos;
                                                reduced_nn_dist[g_query_el_idx * NUM_TRAIN_BLOCKS*NNBS + train_block_idx * NNBS + neigh] =						
						 		compute_min_pos(&global_nn_dist[query_el*TRAINELEMS + train_block], TRAIN_BLOCK_SIZE, &pos);

                                                reduced_nn_idx[g_query_el_idx * NUM_TRAIN_BLOCKS*NNBS + train_block_idx * NNBS + neigh] =
								train_block + pos;
                                                
						// race condition which prevents the implementation of a parallel NNBS min search approach
						global_nn_dist[query_el * TRAINELEMS + train_block + pos] = INF; 
                                        }
				}
			}                              
                }
		/* At this point, for each query point we have calculated the N * NNBS "candidate neighbors", where N
		 * is the number of training element blocks.
		 * Thus, we proceed to the calculation of the final NNBS neighbors for each query point.
		 * Then, we can predict the value of the query point, using the values of its NNBS neighbors.
		 */
                double sum = 0.0, yp;
                int pos;

                #pragma acc parallel loop private(pos, yp, err) firstprivate(sum) reduction(+ : sse, err_sum)
                for (int query_el = 0; query_el < QUERYELEMS; query_el++)
                {
                        #pragma acc loop seq
                        for (int neigh = 0; neigh < NNBS; neigh++)
                        {
                                compute_min_pos(&reduced_nn_dist[query_el * NUM_TRAIN_BLOCKS * NNBS], NUM_TRAIN_BLOCKS * NNBS, &pos);
                                sum += ydata[reduced_nn_idx[query_el * NUM_TRAIN_BLOCKS * NNBS + pos]]; // gather "neigh" neighbor value
                                reduced_nn_dist[query_el * NUM_TRAIN_BLOCKS*NNBS + pos] = INF;
                        }
			/* Calculate the estimated value for the given query point, using the surrogate values of its NNBS neighbors.
			 * We also need to compute the necessary error metrics, and write the necessary data into the yp_vals and err_vals
			 * arrays, which are used by the host in order to write the output logging file.
			 */
                        yp = sum / NNBS; 
			sse += (query_ydata[query_el] - yp) * (query_ydata[query_el] - yp);
			err = 100.0 * fabs((yp - query_ydata[query_el]) / query_ydata[query_el]);
		#if defined(DEBUG)
			yp_vals[query_el] = yp;
			err_vals[query_el] = err;
		#endif
			err_sum += err;
		}		
		t_sum = gettime() - t_start;
	}

#if defined(DEBUG)
	for (int i = 0; i < QUERYELEMS; i++)
		fprintf(fpout,"%.5f %.5f %.2f\n", query_ydata[i], yp_vals[i], err_vals[i]);
#endif
	/* CALCULATE AND DISPLAY RESULTS */

	double mse = sse / QUERYELEMS;
	double ymean = compute_mean(query_ydata, QUERYELEMS);
	double var = compute_var(query_ydata, QUERYELEMS, ymean);
	double r2 = 1 - (mse / var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);
	
	// total computing time without considering the time necessary to copyin/copyout data to/from the GPU
	printf("Total Computing time = %lf secs\n", t_sum); 
	printf("Average time/query = %lf secs\n", t_sum / QUERYELEMS);

	/* CLEANUP */

#if defined(DEBUG)
	/* Close the output file */
	fclose(fpout);
	free(err_vals);
	free(yp_vals);
#endif
	free(mem);
	free(ydata);
	free(query_mem);
	free(query_ydata);

        free(train_vectors);
        free(query_vectors);

	free(global_nn_dist);
	free(global_nn_idx);
	free(reduced_nn_idx);
	free(reduced_nn_dist);

	return 0;
}
