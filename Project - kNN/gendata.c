#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.h"


int main(int argc, char *argv[])
{
	// Allocate enough space to store either training data or query data
	double *mem = (double *)malloc(fmax(TRAINELEMS, QUERYELEMS) * (PROBDIM + 1) * sizeof(double));

	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	SEED_RAND();	/* the training set is fixed */

	// Initialize mem for training data
	for (int i = 0; i < TRAINELEMS; i++)
	{
		for (int k = 0; k < PROBDIM; k++)
			mem[i * (PROBDIM + 1) + k] = get_rand(k);

		mem[i * (PROBDIM + 1) + PROBDIM] = fitfun(&mem[i * (PROBDIM + 1)], PROBDIM);
	}

	store_binary_data(trainfile, mem, TRAINELEMS * (PROBDIM + 1));
	printf("%d data points written to %s!\n", TRAINELEMS, trainfile);

	// Initialize mem for query data
	for (int i = 0; i < QUERYELEMS; i++)
	{
		for (int k = 0; k < PROBDIM; k++)
			mem[i * (PROBDIM + 1) + k] = get_rand(k);

		mem[i * (PROBDIM + 1) + PROBDIM] = fitfun(&mem[i * (PROBDIM + 1)], PROBDIM);
	}

	store_binary_data(queryfile, mem, QUERYELEMS * (PROBDIM + 1));
	printf("%d data points written to %s!\n", QUERYELEMS, queryfile);

	free(mem);
	return 0;
}
