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

	omp_set_dynamic(0); // set OpenMP dynamic mode to false, i.e. use the explicitly defined number of threads
	omp_set_num_threads(omp_get_max_threads()); // run using the maximum supported number of threads

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *mem = (double *)malloc(TRAINELEMS * (PROBDIM + 1) * sizeof(double));
	double *query_mem = (double *)malloc(QUERYELEMS * (PROBDIM + 1) * sizeof(double));	
	
	xdata = (double **)malloc(TRAINELEMS * sizeof(double *));
	
	load_binary_data(trainfile, mem, TRAINELEMS*(PROBDIM+1));

	for (int i = 0; i < TRAINELEMS; i++)
	{
		xdata[i] = mem + i * (PROBDIM + 1); //&mem[i*PROBDIM];
        #if defined(SURROGATES)
		ydata[i] = mem[i * (PROBDIM + 1) + PROBDIM];
        #else
		ydata[i] = 0;
        #endif
	}

	load_binary_data(queryfile, query_mem, QUERYELEMS * (PROBDIM + 1));

#if defined(DEBUG)
	FILE *fpout = fopen("output.knn_omp.txt","w");
#endif
	double *y = malloc(QUERYELEMS * sizeof(double));

	for (int i = 0; i < QUERYELEMS; i++)
	{
	#if defined(SURROGATES)
                y[i] = query_mem[i * (PROBDIM + 1) + PROBDIM];
        #else
                y[i] = 0.0;
        #endif
	}

	double t0, t1, t_start, t_end, t_first = 0.0, t_sum = 0.0;
        double sse = 0.0;
        double err_sum = 0.0;

	// -------------------------------------
	// -------------new arrays (?)----------
	// -------------------------------------
#if defined(DEBUG)
	double *yp_vals = malloc(QUERYELEMS * sizeof(double));
	double *err_vals = malloc(QUERYELEMS * sizeof(double));
#endif

	size_t nthreads;

	t_start = gettime();
	#pragma omp parallel reduction(+ : sse, err_sum, t_sum) private(t0, t1) 
	{
		size_t tid = omp_get_thread_num();

		#pragma omp single
		nthreads = omp_get_num_threads();

		size_t start = tid * (QUERYELEMS / nthreads);
		size_t end = (tid + 1) * (QUERYELEMS / nthreads);
		if (tid == nthreads - 1)
			end = QUERYELEMS;
	
	#if defined(DEBUG)
		double yp[end - start + 1], err[end - start + 1];
		size_t idx = 0;
	#else
		double yp;
	#endif
		for (int i = start; i < end; i++) 	/* requests */
		{
			t0 = gettime();
		#if defined(DEBUG)
                	yp[idx] = find_knn_value(&query_mem[i * (PROBDIM + 1)], PROBDIM, NNBS);
		#else
                	yp = find_knn_value(&query_mem[i * (PROBDIM + 1)], PROBDIM, NNBS);
		#endif
                	t1 = gettime();

			t_sum += (t1 - t0);
			
			if (i == 0)
				t_first = (t1 - t0);
		#if defined(DEBUG)
			sse += (y[i] - yp[idx]) * (y[i] - yp[idx]);
			err[idx] = 100.0 * fabs((yp[idx] - y[i]) / y[i]);
			err_sum += err[idx];
			idx++;
		#else
			sse += (y[i] - yp) * (y[i] - yp);
			err_sum += 100.0 * fabs((yp - y[i]) / y[i]);
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

#if defined(DEBUG)
	for (int i = 0; i < QUERYELEMS; i++)
	{
		for (int k = 0; k < PROBDIM; k++)
			fprintf(fpout, "%.5f ", query_mem[i * (PROBDIM + 1) + k]);

		fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp_vals[i], err_vals[i]);
	}
#endif


	double mse = sse / QUERYELEMS;
	double ymean = compute_mean(y, QUERYELEMS);
	double var = compute_var(y, QUERYELEMS, ymean);
	double r2 = 1 - (mse / var);

	printf("Results for %d query points\n", QUERYELEMS);
	printf("APE = %.2f %%\n", err_sum / QUERYELEMS);
	printf("MSE = %.6f\n", mse);
	printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

	printf("Total time = %lf secs\n", t_end - t_start);
	printf("Time for 1st query = %lf secs\n", t_first);
	printf("Time for 2..N queries = %lf secs\n", t_end - t_start - t_first);
	printf("Average time/query = %lf secs\n", (t_sum - t_first) / (QUERYELEMS - 1));

	free(mem);
	free(xdata);
	free(query_mem);
	free(y);

	// new
#if defined(DEBUG)
	free(yp_vals);
	free(err_vals);
#endif

	return 0;
}
