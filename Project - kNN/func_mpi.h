#include "mpi.h"
#include "func.h"

void load_binary_data_mpi(const char *filename, double *data, query_t *queries, const int N, int offset)
{
	int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// Open the file (collective call)
	MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

        // Calculate the offset for each rank
        MPI_Offset base;
        MPI_File_get_position(f, &base);

        MPI_Offset data_offset = offset * sizeof(double);

	// Collectively Read the data
        MPI_Status status;
        MPI_File_read_at_all(f, base + data_offset, data, N, MPI_DOUBLE, &status); // blocking collective call

        // If queries are loaded, initialize the queries structs
	if (queries != NULL)
	{
		size_t posix_res;
		for (int i = 0; i < QUERYELEMS; i++)
		{
			posix_res = posix_memalign((void **)(&(queries[i].x)), 32, PROBDIM * sizeof(double));
			assert(posix_res == 0);

			for (int k = 0; k < PROBDIM; k++)
				queries[i].x[k] = data[i * (PROBDIM + 1) + k];

			for (int j = 0; j < NNBS; j++)
				queries[i].nn_idx[j] = -1;

			for (int j = 0; j < NNBS; j++)
				queries[i].nn_d[j] = 1e99 - j;
		}
	}

        // Close the file
        MPI_File_close(&f);        
}

// Store mpi
// mode == 0 -> write trainelements
// mode == 1 -> write queryelements
void store_binary_data_mpi(const char *filename, double *data, const int N, int offset, int mode)
{
        int rank, nprocs;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

	// Open the file (collective call)
	MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
        MPI_File_preallocate(f, (mode == 0 ? TRAINELEMS : QUERYELEMS) * sizeof(double));

        // Calculate the offset for each rank
        MPI_Offset base;
        MPI_File_get_position(f, &base);

        MPI_Offset data_offset = offset * sizeof(double);

	// Collectively Read the data
        MPI_Status status;
        MPI_File_write_at_all(f, base + data_offset, data, N, MPI_DOUBLE, &status); // blocking collective call
        
        MPI_File_close(&f); 
}
