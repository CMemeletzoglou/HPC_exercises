#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <assert.h>

#define NNBS 32

// struct that will preserve the k nearest neighbors for each query.
// Will be used in order to load training data in a blocking fashion,
// iterating over all query points. Thus for each query point we will
// need to preserve the k nearest neighbors that have been found so far
// in the preceding blocks.
typedef struct query_s
{
	double *x; // Query's coordinates + surrogate value 
	int nn_idx[NNBS]; // The index (< TRAINELEMS) of the k nearest neighbors
	double nn_d[NNBS]; // The distance between the query point and each one of the k nearest neighbors
} query_t;

/* I/O routines */
void store_binary_data(char *filename, double *data, int n)
{
	FILE *fp;
	fp = fopen(filename, "wb");
	if (fp == NULL)
	{
		printf("fopen(%s, \"wb\") FAILED!\n", filename);
		exit(1);
	}
	size_t nelems = fwrite(data, sizeof(double), n, fp);
	assert(nelems == n); // check that all elements were actually written
	fclose(fp);
}

void load_binary_data(const char *filename, double *data, query_t *query_status, const int n)
{
	FILE *fp;
	fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		printf("fopen(%s, \"wb\") FAILED!\n", filename);
		exit(1);
	}
	size_t nelems = fread(data, sizeof(double), n, fp);
	assert(nelems == n); // check that all elements were actually read
	fclose(fp);

	// If queries are loaded, initialize the query_status structs
	if (query_status != NULL)
	{
		for (int i = 0; i < QUERYELEMS; i++)
			for (int j = 0; j < NNBS; j++)
			{
				query_status[i].x = &data[i * (PROBDIM + 1)];
				query_status[i].nn_idx[j] = -1;
				query_status[i].nn_d[j] = 1e99 - j;
			}
	}
}

double read_nextnum(FILE *fp)
{
	double val;

	int c = fscanf(fp, "%lf", &val);
	if (c <= 0) {
		fprintf(stderr, "fscanf returned %d\n", c);
		exit(1);
	}
	return val;
}

/* Timer */
double gettime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double) (tv.tv_sec+tv.tv_usec/1000000.0);
}

/* Function to approximate */
double fitfun(double *x, int n)
{
	double f = 0.0;
	int i;

#if 1
	for(i=0; i<n; i++)	/* circle */
		f += x[i]*x[i];
#endif
#if 0
	for(i=0; i<n-1; i++) {	/*  himmelblau */
		f = f + pow((x[i]*x[i]+x[i+1]-11.0),2) + pow((x[i]+x[i+1]*x[i+1]-7.0),2);
	}
#endif
#if 0
	for (i=0; i<n-1; i++)   /* rosenbrock */
		f = f + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
#endif
#if 0
	for (i=0; i<n; i++)     /* rastrigin */
		f = f + pow(x[i],2) + 10.0 - 10.0*cos(2*M_PI*x[i]);
#endif

	return f;
}


/* random number generator  */
#define SEED_RAND()     srand48(1)
#define URAND()         drand48()

#ifndef LB
#define LB -1.0
#endif
#ifndef UB
#define UB 1.0
#endif

double get_rand(int k)
{
	return (UB-LB)*URAND()+LB;
}


/* utils */
double compute_min(double *v, int n)
{
	int i;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) vmin = v[i];

	return vmin;
}

double compute_max(double *v, int n)
{
	int i;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) vmax = v[i];

	return vmax;
}

double compute_sum(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s;
}

double compute_sum_pow(double *v, int n, int p)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i], p);

	return s;
}

double compute_mean(double *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

double compute_std(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return sqrt(s/(n-1));
}

double compute_var(double *v, int n, double mean)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

double compute_dist(double *v, double *w, int n)
{
	int i;
	double s = 0.0;
	for (i = 0; i < n; i++) {
		s += pow(v[i]-w[i],2);
	}

	return sqrt(s);
}

double compute_max_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}

double compute_min_pos(double *v, int n, int *pos)
{
	int i, p = 0;
	double vmin = v[0];
	for (i = 1; i < n; i++)
		if (v[i] < vmin) {
			vmin = v[i];
			p = i;
		}

	*pos = p;
	return vmin;
}

double compute_root(double dist, int norm)
{
	if (dist == 0) return 0;

	switch (norm) {
	case 2:
		return sqrt(dist);
	case 1:
	case 0:
		return dist;
	default:
		return pow(dist, 1 / (double) norm);
	}
}

double compute_distance(double *pat1, double *pat2, int lpat, int norm)
{
	register int i;
	double dist = 0.0;

	for (i = 0; i < lpat; i++) {
		double diff = 0.0;

		diff = pat1[i] - pat2[i];

		switch (norm) {
		double   adiff;

		case 2:
			dist += diff * diff;
			break;
		case 1:
			dist += fabs(diff);
			break;
		case 0:
			if ((adiff = fabs(diff)) > dist)
			dist = adiff;
			break;
		default:
			dist += pow(fabs(diff), (double) norm);
			break;
		}
	}

	return dist;	// compute_root(dist);
}

// npat -> TRAINELEMS
// lpat -> PROBDIM
void compute_knn_brute_force(double **xdata, query_t *q, int dim, int k, int train_data_offset, int num_train_data)
{
	int i, max_i;
	double max_d, new_d;

	// find K neighbors
	max_d = compute_max_pos(q->nn_d, k, &max_i);
	for (i = train_data_offset; i < train_data_offset + num_train_data; i++) // i runs inside each training block's boundaries
	{
		new_d = compute_dist(q->x, xdata[i], dim); // euclidean
		if (new_d < max_d) // add point to the list of knns, replace element max_i
		{	
			q->nn_idx[max_i] = i;
			q->nn_d[max_i] = new_d;
		}
		max_d = compute_max_pos(q->nn_d, k, &max_i);
	}

/* 	// sort the knn list using bubble sort
	int temp_idx, j;
	double temp_d;
	for (i = (k - 1); i > 0; i--)
	{
		for (j = 1; j <= i; j++)
		{
			if (q->nn_d[j-1] > q->nn_d[j])
			{
				temp_d = q->nn_d[j-1];
				q->nn_d[j-1] = q->nn_d[j];
				q->nn_d[j] = temp_d;
				
				temp_idx = q->nn_idx[j-1];
				q->nn_idx[j-1] = q->nn_idx[j];
				q->nn_idx[j] = temp_idx;
			}
		}
	} */
}


/* compute an approximation based on the values of the neighbors */
double predict_value(int dim, int knn, double *xdata, double *ydata)
{
	int i;
	double sum_v = 0.0;
	// plain mean (other possible options: inverse distance weight, closest value inheritance)
	for (i = 0; i < knn; i++)
		sum_v += ydata[i];

	return sum_v / knn;
}