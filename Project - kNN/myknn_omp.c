#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "func.h"

#ifndef PROBDIM
#define PROBDIM 2
#endif

static double **xdata;
static double *ydata;

double find_knn_value(query_t *q, int knn)
{
#if defined(SIMD)
	__attribute__((aligned(32))) double fd[knn];	// function values for the knn neighbors
#else 
	double fd[knn];
#endif	

	for (int i = 0; i < knn; i++)
		fd[i] = ydata[q->nn_idx[i]];

        return predict_value(fd, knn);
}

int main(int argc, char *argv[])
{
	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	omp_set_dynamic(0); // set OpenMP dynamic mode to false, i.e. use the explicitly defined number of threads
	omp_set_num_threads(omp_get_max_threads()); // run using the maximum supported number of threads

        int L1d_size, train_block_size = 1;
	get_L1d_size(&L1d_size); // get L1d cache size
	// calculate the appropriate train block size as the previous power of 2
	if(L1d_size > 0)
		train_block_size = pow(2, floor(log2((L1d_size * 1000) / (PROBDIM * sizeof(double)))));

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *mem = (double *)malloc(TRAINELEMS * (PROBDIM + 1) * sizeof(double));
	ydata = (double*)malloc(TRAINELEMS * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * (PROBDIM + 1) * sizeof(double));	
        query_t *queries = (query_t *)malloc(QUERYELEMS * sizeof(query_t));
	
#if defined(SIMD)
	int posix_res;
        // Malloc aligned space for query.x data 
        for (int i = 0; i < QUERYELEMS; i++)
        {
                posix_res = posix_memalign((void **)(&(queries[i].x)), 32, PROBDIM * sizeof(double));
                assert(posix_res == 0);
        }
#endif

	xdata = (double **)malloc(TRAINELEMS * sizeof(double *));

	load_binary_data(trainfile, mem, NULL, TRAINELEMS*(PROBDIM+1));
	load_binary_data(queryfile, query_mem, queries, QUERYELEMS * (PROBDIM + 1));

#if defined(SIMD)
	// Allocate new memory for the handler arrays, so that it is aligned and copy the data there
	// Align each xdata[i] to a 32 byte boundary so you may later use SIMD
	for (int i = 0; i < TRAINELEMS; i++)
	{
		posix_res = posix_memalign((void **)(&(xdata[i])), 32, PROBDIM * sizeof(double));
		assert(posix_res == 0);
	}
	copy_to_aligned(mem, xdata, (PROBDIM+1), PROBDIM, TRAINELEMS);
#else
	// Assign to the handler arrays, pointers to the already allocated mem
	for (int i = 0; i < TRAINELEMS; i++)
		xdata[i] = &mem[i*(PROBDIM + 1)];
#endif

	/* Configure and Initialize the ydata handler arrays */
	for (int i = 0; i < TRAINELEMS; i++)
	{
#if defined(SURROGATES)
		ydata[i] = mem[i * (PROBDIM + 1) + PROBDIM];
#else
		ydata[i] = 0;
#endif
	}

	double *query_ydata = malloc(QUERYELEMS * sizeof(double));

	for (int i = 0; i < QUERYELEMS; i++)
	{
#if defined(SURROGATES)
		query_ydata[i] = query_mem[i * (PROBDIM + 1) + PROBDIM];
#else
		query_ydata[i] = 0;
#endif
	}

	assert(TRAINELEMS % train_block_size == 0);

#if defined(DEBUG)
	FILE *fpout = fopen("output.knn_omp.txt","w");
	double *yp_vals = malloc(QUERYELEMS * sizeof(double));
	double *err_vals = malloc(QUERYELEMS * sizeof(double));
#endif
	
        /* COMPUTATION PART */
        double t0, t1, t_start, t_end, t_sum = 0.0, t_total;
        double sse = 0.0;
        double err_sum = 0.0;

	size_t nthreads;

	t_start = gettime();
        /* Parallel + Blocking Query Point k-nearest neighbors calculation.
         * For each block of Training Points of size train_block_size,
         * find each Query Point's k neighbors using the Training Points of the
         * current block.
         * We divide the iterations within each thread block to the available threads,
         * since the k-neighbor calculation for each Query Point, does not depend
         * on the k-neighbor calculation of the other Query Points.
         */
	#pragma omp parallel reduction(+ : sse, err_sum, t_sum) private(t0, t1) 
	{
		for (int train_offset = 0; train_offset < TRAINELEMS; train_offset += train_block_size)
		{
			#pragma omp for nowait
			for (int i = 0; i < QUERYELEMS; i++)
				compute_knn_brute_force(xdata, ydata, &(queries[i]), PROBDIM, NNBS, train_offset, 0, train_block_size);			
		}

		size_t tid = omp_get_thread_num();

		#pragma omp single		
		nthreads = omp_get_num_threads();

		size_t start = tid * (QUERYELEMS / nthreads);
		size_t end = (tid + 1) * (QUERYELEMS / nthreads);
		if (tid == nthreads - 1)
			end = QUERYELEMS;

		/* After having found each Query Point's k-nearest neighbors, we proceed
         	 * with the calculation of the estimated value for the target function,
         	 * for each Query Point.
         	 * 
		 * We explicitly manage the chunks of work assigned to each thread, 
		 * using the variables start and end, because if we just use a
		 * parallel for worksharing construct, if the DEBUG mode is enabled,
		 * a race condition would appear on the writing of the output logging file.
		 *
		 * Therefore, we use the variables start and end and the thread-local arrays
		 * yp, err, to gather the debugging info from each thread's computations.
		 * Then the thread-local information, is written on the shared arrays
		 * yp_vals, err_vals, where each thread writes in a different region of the
		 * arrays, defined by the start and end variables.
		 */	
	#if defined(DEBUG)
		double yp[end - start + 1], err[end - start + 1];
		int idx = 0;
	#else
		double yp;
	#endif
		for (int i = start; i < end; i++) 	/* requests */
		{
			t0 = gettime();
		#if defined(DEBUG)
                	yp[idx] = find_knn_value(&(queries[i]), NNBS);
		#else
                	yp = find_knn_value(&(queries[i]), NNBS);
		#endif
                	t1 = gettime();

                        t_sum += t1 - t0;

                #if defined(DEBUG)
			sse += (query_ydata[i] - yp[idx]) * (query_ydata[i] - yp[idx]);
			err[idx] = 100.0 * fabs((yp[idx] - query_ydata[i]) / query_ydata[i]);
			err_sum += err[idx];
			idx++;
		#else
			sse += (query_ydata[i] - yp) * (query_ydata[i] - yp);
			err_sum += 100.0 * fabs((yp - query_ydata[i]) / query_ydata[i]);
		#endif
		}
	#if defined(DEBUG)
		idx = 0;
		for (int i = start; i < end; i++)
		{
			yp_vals[i] = yp[idx];
			err_vals[i] = err[idx++];
		}
	#endif
	}
        t_end = gettime();
        // total running time (parallel blocking neighbor find + query point function value estimation calculation)
        t_total = t_end - t_start; 

#if defined(DEBUG)
	for (int i = 0; i < QUERYELEMS; i++)
		fprintf(fpout,"%.5f %.5f %.2f\n", query_ydata[i], yp_vals[i], err_vals[i]);
#endif
        
	double mse = sse / QUERYELEMS;
	double ymean = compute_mean(query_ydata, QUERYELEMS);
	double var = compute_var(query_ydata, QUERYELEMS, ymean);
	double r2 = 1 - (mse / var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	printf("Total Computing time = %lf secs\n", t_total);
	printf("Average time/query = %lf secs\n", t_total / QUERYELEMS);

#if defined(SIMD)
	for (int i = 0; i < QUERYELEMS; i++)
		free(queries[i].x);
#endif
        free(queries);
        free(query_ydata);
	free(query_mem);
	
#if defined(SIMD)
	for (int i = 0; i < TRAINELEMS; i++)
		free(xdata[i]);
#endif
	free(xdata);
	free(ydata);        
	free(mem);

#if defined(DEBUG)
	free(yp_vals);
	free(err_vals);
        fclose(fpout);
#endif

        return 0;
}
