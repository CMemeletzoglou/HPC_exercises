#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

typedef struct Diagnostics_s
{
        double time;
        double heat;
} Diagnostics;

typedef struct Diffusion2D_s
{
        double D_, L_, T_;
        int N_, Ntot_, real_N_;
        double dr_, dt_, fac_;
        int rank_, procs_;
        int local_N_;
        double *rho_, *rho_tmp_;
        Diagnostics *diag_;
} Diffusion2D;

void initialize_density(Diffusion2D *D2D)
{
        int real_N_ = D2D->real_N_;
        int local_N_ = D2D->local_N_;
        double *rho_ = D2D->rho_;
        double dr_ = D2D->dr_;
        double L_ = D2D->L_;
        int rank_ = D2D->rank_;
        int procs_ = D2D->procs_;
        int gi, gj;

        /// Initialize rho(x, y, t=0).
        double bound = 0.25 * L_;

        for (int i = 1; i <= local_N_; ++i) // row traversal loop
        {
                // global matrix row index
                gi = floor((double)rank_ / sqrt(procs_)) * local_N_ + i; 

                for (int j = 1; j <= local_N_; ++j) // column traversal loop
                {
                        gj = ( rank_ % (int)sqrt(procs_) ) * local_N_ + j; // global matrix column index

                        /* initialize each cell of the rho_ matrix, to 0 or 1, depending on its position
                         * with respect to the area defined by the bound constant
                         */
                        if (fabs((gi - 1) * dr_ - 0.5 * L_) < bound && fabs((gj - 1) * dr_ - 0.5 * L_) < bound)
                        {
                                rho_[i*real_N_ + j] = 1;
                        }
                        else
                        {
                                rho_[i*real_N_ + j] = 0;
                        }
                }
        }
}

void init(Diffusion2D *D2D,
                const double D,
                const double L,
                const int N,
                const int T,
                const double dt,
                const int rank,
                const int procs)
{
        D2D->D_ = D;
        D2D->L_ = L;
        D2D->N_ = N;
        D2D->T_ = T;
        D2D->dt_ = dt;
        D2D->rank_ = rank;
        D2D->procs_ = procs;

        // Real space grid spacing.
        D2D->dr_ = D2D->L_ / (D2D->N_ - 1);

        // Stencil factor.
        D2D->fac_ = D2D->dt_ * D2D->D_ / (D2D->dr_ * D2D->dr_);

        // Number of rows per process = Number of rows per square tile
        D2D->local_N_ = (int)sqrt(procs) * (D2D->N_ / D2D->procs_); 

        // Actual dimension of a row (+2 for the ghost cells). (**num_cols +2**)
        D2D->real_N_ = D2D->local_N_ + 2;

        // Total number of cells inside each square tile
        D2D->Ntot_ = (D2D->local_N_ + 2) * (D2D->local_N_ + 2);

        D2D->rho_ = (double *)calloc(D2D->Ntot_, sizeof(double));
        D2D->rho_tmp_ = (double *)calloc(D2D->Ntot_, sizeof(double));
        D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

        // Check that the timestep satisfies the restriction for stability.
        if (D2D->rank_ == 0)
                printf("timestep from stability condition is %lf\n", D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

        initialize_density(D2D);
}

/* Helper function to gather data from a column vector, into the provided buffer (2nd argument).
 * The column index is passed as the 3rd argument. We move to the first element of the column vector
 * using the col_index, as an offset from the first element of the rho_ 2d matrix.
 * 
 * The 2d rho_ matrix is stored, as a linear array, therefore the rho_ + col_index, actually gives
 * the element rho_[0][col_index].
 * 
 * Each rank (MPI Node) has a K x K square submatrix (with K = local_N_) of the global N x N rho_,
 * but it has two extra columns and two extra rows, which are used as buffers for the data received
 * from the neighbouring ranks.
 * 
 * Therefore, the actual data for each MPI Node, are contained in the (K-2) x (K-2) submatrix of the K x K matrix.
 * gather_column_data needs to ignore the first and the last "superficial" rows. So, its for loop starts from row
 * index = 1 to local_N, which is the rows containing the actual data.
 *
 * Each element of the column vector, has an offset of real_N_ from the previous element of the same column. 
 * Thus, we need to make a jump of real_N_, in each new iteration of the for loop. Each element retrieved is stored
 * into the buf buffer.
 */
void gather_column_data(Diffusion2D *D2D, double *buf, int col_index)
{
        int local_N_ = D2D->local_N_;
        int real_N_ = D2D->real_N_;
        double *rho_start_elmnt = D2D->rho_ + col_index;
        
        for(int i = 1; i <= local_N_; i++)
        {
                *(buf + i-1) = *(rho_start_elmnt + (i * real_N_));                                
        }
}

/* Similarly, to gather_column_data, scatter_column_data operates on a column vector of the rho_ matrix.
 *
 * The function updates a column vector of the rho_ matrix with the data contained in the buf buffer (2nd argument).
 */
void scatter_column_data(Diffusion2D *D2D, double *buf, int col_index)
{
        int local_N_ = D2D->local_N_;
        int real_N_ = D2D->real_N_;
        double *rho_start_elmnt = D2D->rho_ + col_index;
        
        for(int i = 1; i <= local_N_; i++)
        {
                *(rho_start_elmnt + (i * real_N_)) = *(buf + i-1);
        }
}

void advance(Diffusion2D *D2D)
{
        int real_N_ = D2D->real_N_;
        int local_N_ = D2D->local_N_;
        double *rho_ = D2D->rho_;
        double *rho_tmp_ = D2D->rho_tmp_;
        double fac_ = D2D->fac_;
        int rank_ = D2D->rank_;
        int procs_ = D2D->procs_;

        MPI_Request req[8];
        MPI_Status status[8];

        // we have sqrt(procs_) square tiles, i.e. sqrt(procs_) MPI nodes
        int sqrt_procs = (int)sqrt(procs_);

        // Compute the IDs for each rank's neighbouring ranks
        int below_rank = ((rank_ + sqrt_procs) < procs_) ? rank_ + sqrt_procs : MPI_PROC_NULL;
        int upper_rank = ((rank_ - sqrt_procs) >= 0) ? rank_ - sqrt_procs : MPI_PROC_NULL;

        int right_rank = ( ( ( (rank_ + 1) % sqrt_procs) ) == 0) ? MPI_PROC_NULL : rank_ + 1;
        int left_rank = ( ( ( rank_ % sqrt_procs) ) == 0 ) ? MPI_PROC_NULL : rank_ - 1;

        double *data_buf = calloc(local_N_, sizeof(double));        
        double *rcv_buf = calloc(local_N_, sizeof(double));

        for (int i = 0; i < 8; i++)
                req[i] = MPI_REQUEST_NULL;

        // *************************************************************************
        //                              COMMUNICATION PART
        // *************************************************************************

        // Exchange ALL necessary ghost cells with neighboring ranks.
        if (upper_rank != MPI_PROC_NULL)
        {
                MPI_Irecv(&rho_[0 * real_N_ + 1], local_N_, MPI_DOUBLE, upper_rank, 100, MPI_COMM_WORLD, &req[0]);
                MPI_Isend(&rho_[1 * real_N_ + 1], local_N_, MPI_DOUBLE, upper_rank, 100, MPI_COMM_WORLD, &req[1]);
        }
        
        if(below_rank != MPI_PROC_NULL)
        {
                MPI_Irecv(&rho_[(local_N_+1)*real_N_+1], local_N_, MPI_DOUBLE, below_rank, 100, MPI_COMM_WORLD, &req[2]);
                MPI_Isend(&rho_[local_N_*real_N_+1], local_N_, MPI_DOUBLE, below_rank, 100, MPI_COMM_WORLD, &req[3]);
        }

        if (right_rank != MPI_PROC_NULL)
        {
                MPI_Irecv(rcv_buf, local_N_, MPI_DOUBLE, right_rank, 100, MPI_COMM_WORLD, &req[4]);

                gather_column_data(D2D, data_buf, local_N_);
                MPI_Isend(data_buf, local_N_, MPI_DOUBLE, right_rank, 100, MPI_COMM_WORLD, &req[5]);
        }

        if (left_rank != MPI_PROC_NULL)
        {
                MPI_Irecv(rcv_buf, local_N_, MPI_DOUBLE, left_rank, 100, MPI_COMM_WORLD, &req[6]);

                gather_column_data(D2D, data_buf, 1);
                MPI_Isend(data_buf, local_N_, MPI_DOUBLE, left_rank, 100, MPI_COMM_WORLD, &req[7]);
        }

        // *************************************************************************
        //                              COMPUTATION PART
        // *************************************************************************

        // Central differences in space, forward Euler in time with Dirichlet
        // boundaries.
        for (int i = 2; i < local_N_; ++i)
        {
                for (int j = 2; j < local_N_; ++j)
                {
                        rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                                fac_
                                                *
                                                (
                                                + rho_[i*real_N_ + (j+1)]
                                                + rho_[i*real_N_ + (j-1)]
                                                + rho_[(i+1)*real_N_ + j]
                                                + rho_[(i-1)*real_N_ + j]
                                                - 4.*rho_[i*real_N_ + j]
                                                );
                }
        }

        // Wait in order to ensure data from neighbouring ranks, have arrived
        MPI_Waitall(8, req, status);
        if(status[4].MPI_SOURCE == right_rank)  
                scatter_column_data(D2D, rcv_buf, local_N_+1);
        
        if(status[6].MPI_SOURCE == left_rank)  
                scatter_column_data(D2D, rcv_buf, 0);

        // update first and last column of each rank
        for (int i = 1; i <= local_N_; ++i)
        {
                for (int j = 1; j <= local_N_; j += local_N_- 1)
                {
                        rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                                fac_
                                                *
                                                (
                                                + rho_[i*real_N_ + (j+1)]
                                                + rho_[i*real_N_ + (j-1)]
                                                + rho_[(i+1)*real_N_ + j]
                                                + rho_[(i-1)*real_N_ + j]
                                                - 4.*rho_[i*real_N_ + j]
                                                );
                }
        }

        // Update the first and the last rows of each rank.
        for (int i = 1; i <= local_N_; i += local_N_- 1)
        {
                for (int j = 1; j <= local_N_; ++j)
                {
                        rho_tmp_[i*real_N_ + j] = rho_[i*real_N_ + j] +
                                                fac_
                                                *
                                                (
                                                + rho_[i*real_N_ + (j+1)]
                                                + rho_[i*real_N_ + (j-1)]
                                                + rho_[(i+1)*real_N_ + j]
                                                + rho_[(i-1)*real_N_ + j]
                                                - 4.*rho_[i*real_N_ + j]
                                                );
                }
        }

        // Swap rho_ with rho_tmp_. This is much more efficient,
        // because it does not copy element by element, just replaces storage
        // pointers.
        double *tmp_ = D2D->rho_tmp_;
        D2D->rho_tmp_ = D2D->rho_;
        D2D->rho_ = tmp_;

        // free the dynamically allocated buffers
        free(data_buf);
        free(rcv_buf);
}

void compute_diagnostics(Diffusion2D *D2D, const int step, const double t)
{
        int real_N_ = D2D->real_N_;
        int local_N_ = D2D->local_N_;
        double *rho_ = D2D->rho_;
        double dr_ = D2D->dr_;
        int rank_ = D2D->rank_;

        double heat = 0.0;
        for(int i = 1; i <= local_N_; ++i)
                for(int j = 1; j <= local_N_; ++j)
                        heat += rho_[i*real_N_ + j] * dr_ * dr_;

        MPI_Reduce(rank_ == 0? MPI_IN_PLACE: &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 

        if (rank_ == 0)
        {
        #if DEBUG
                printf("t = %lf heat = %lf\n", t, heat);
        #endif
                D2D->diag_[step].time = t;
                D2D->diag_[step].heat = heat;
        }
}

void write_diagnostics(Diffusion2D *D2D, const char *filename)
{

        FILE *out_file = fopen(filename, "w");
        for (int i = 0; i < D2D->T_; i++)
                fprintf(out_file, "%f\t%f\n", D2D->diag_[i].time, D2D->diag_[i].heat);
        fclose(out_file);
}

/* Helper function to gather the rho_ submatrix controlled by each rank, into the
 * buffer specified by the second argument. Called in write_density_mpi .
 */
void collect_useful_data(Diffusion2D *D2D, double* buf)
{
        int local_N_ = D2D->local_N_;
        int real_N_ = D2D->real_N_;
        double* rho_ = D2D->rho_;

        for (int i = 1; i <= local_N_; ++i)
                for (int j = 1; j <= local_N_; ++j)
                        buf[(i-1)*local_N_ + (j-1)] = rho_[i*real_N_ + j];
}

/* Helper function to construct the rho_ submatrix controlled by each rank,
 * using the data provided from buf.  Called in read_density_mpi .
 */
void construct_rho(Diffusion2D *D2D, double* buf)
{
        int local_N_ = D2D->local_N_;
        int real_N_ = D2D->real_N_;
        double* rho_ = D2D->rho_;

        for (int i = 1; i <= local_N_; ++i)
                for (int j = 1; j <= local_N_; ++j)
                        rho_[i * real_N_ + j] = buf[(i - 1) * local_N_ + (j - 1)];
}

void write_density_mpi(Diffusion2D *D2D, char *filename)
{
        int local_N_ = D2D->local_N_;
        int rank_ = D2D->rank_;
        int local_ntot_ = local_N_ * local_N_;
        int nprocs_ = D2D->procs_;

        int sqrt_procs = (int)sqrt(nprocs_);
        int data_len = local_ntot_ * sizeof(double);

        /* BUFFER DATA */

        double* buf = (double*)malloc(data_len);
        collect_useful_data(D2D, buf); // gather each rank's data

        /* CALCULATE INITIAL OFFSET FOR EACH PROCESS */

        // Each row of the global rho_ matrix, contains sqrt(nprocs_) ranks.
        // Each rank writes data_len bytes (local_ntot_ doubles).
        int block_row_size = data_len * sqrt_procs;
        int rank_row_size = local_N_ * sizeof(double);

        // find the block row index of each rank and then "skip" the blocks written by the previous ranks
        int block_row_offset = floor((double)rank_ / sqrt_procs) * block_row_size;

        // skip one line, for each of the previous ranks of the same block row
        int inrow_offset = (rank_ % sqrt_procs) * rank_row_size;

        /* SET THE MPI VIEW and USE THAT VIEW FOR MPI_File_write_all */

        // Open the file (collective call)
        MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &f);

        MPI_Offset ntotal_bytes = data_len * nprocs_;
        MPI_File_preallocate(f, ntotal_bytes);

        // Calculate the initial offset for each rank
        MPI_Offset base;
        MPI_File_get_position(f, &base);

        // initial rank offset
        MPI_Offset rank_offset = block_row_offset + inrow_offset;

        // Create the view
        MPI_Datatype VEC_DOUBLE_WITH_HOLES;
        MPI_Type_vector(local_N_, local_N_, D2D->N_, MPI_DOUBLE, &VEC_DOUBLE_WITH_HOLES);
        MPI_Type_commit(&VEC_DOUBLE_WITH_HOLES);
        MPI_File_set_view(f, rank_offset, MPI_DOUBLE, VEC_DOUBLE_WITH_HOLES, "native", MPI_INFO_NULL);

        // Write the data - MIND THE VIEW!!!
        MPI_Status status;
        MPI_File_write_all(f, buf, local_ntot_, MPI_DOUBLE, &status); // blocking collective call
        // printf("Rank : %d, New rank offset for line %d = %lld and wrote %ld bytes\n", rank_, 0, rank_offset, status._ucount);
        
        /* FREE */
        // Close the file - free buffer
        MPI_File_close(&f);
        free(buf);
        MPI_Type_free(&VEC_DOUBLE_WITH_HOLES);
}

void read_density_mpi(Diffusion2D *D2D, char *filename)
{
        int local_N_ = D2D->local_N_;
        int rank_ = D2D->rank_;
        int local_ntot_ = local_N_ * local_N_;
        int nprocs_ = D2D->procs_;

        int sqrt_procs = (int)sqrt(nprocs_);
        int data_len = local_ntot_ * sizeof(double);

        /* ALLOCATE BUFFER TO SAVE THE READ DATA INTO  */
        double* buf = (double*)malloc(data_len);

        /* CALCULATE INITIAL OFFSET FOR EACH PROCESS */

        // Each row of the global rho_ matrix, contains sqrt(nprocs_) ranks.
        // Each rank writes data_len bytes (local_ntot_ doubles).
        int block_row_size = data_len * sqrt_procs;
        int rank_row_size = local_N_ * sizeof(double);

        // find the block row index of each rank and then "skip" the blocks written by the previous ranks
        int block_row_offset = floor((double)rank_ / sqrt_procs) * block_row_size;

        // skip one line, for each of the previous ranks of the same block row
        int inrow_offset = (rank_ % sqrt_procs) * rank_row_size;

        /* SET THE MPI VIEW and USE THAT VIEW FOR MPI_File_read_all */

        // Open the file (collective call)
        MPI_File f;
        MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &f);

        // Calculate the initial offset for each rank
        MPI_Offset base;
        MPI_File_get_position(f, &base);

        // initial rank offset
        MPI_Offset rank_offset = block_row_offset + inrow_offset;

        // Create the view
        MPI_Datatype VEC_DOUBLE_WITH_HOLES;
        MPI_Type_vector(local_N_, local_N_, D2D->N_, MPI_DOUBLE, &VEC_DOUBLE_WITH_HOLES);
        MPI_Type_commit(&VEC_DOUBLE_WITH_HOLES);
        MPI_File_set_view(f, rank_offset, MPI_DOUBLE, VEC_DOUBLE_WITH_HOLES, "native", MPI_INFO_NULL);

        // Read the data - MIND THE VIEW!!!
        MPI_Status status;
        MPI_File_read_all(f, buf, local_ntot_, MPI_DOUBLE, &status); // blocking collective call
        
        // Copy the data into rho_
        construct_rho(D2D, buf);

        /* FREE */
        // Close the file - free buffer
        MPI_File_close(&f);
        free(buf);
        MPI_Type_free(&VEC_DOUBLE_WITH_HOLES);
}

int main(int argc, char* argv[])
{
        int rank, procs;
        MPI_Init(&argc, &argv); // initialize the MPI enviroment
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &procs);

        const double D  = 1;
        const double L  = 1;
        const int  N  = 1*16;
        const int T = 1000;
        const double dt = 1e-9;

        Diffusion2D system;

        init(&system, D, L, N, T, dt, rank, procs); // initialize the 2d diffusion system

        int restart_step = -1;
        int break_at_step = -100;
        int first_step = 0;

        if (argc == 2)
        {
                restart_step = atoi(argv[1]);
        }
        else if (argc == 3)
        {
                restart_step = atoi(argv[1]);
                break_at_step = atoi(argv[2]);
                if(break_at_step > T)
                        break_at_step = T;
        }

        if (restart_step >= 0) // >=0 to be able to restart from the very first step
        {
                read_density_mpi(&system, (char *)"density_mpi.dat.restart");
                first_step = restart_step;
        }

        double t0 = MPI_Wtime();
        int step;
        for (step = first_step; step < T; ++step)
        {
                advance(&system);
        #ifndef _PERF_
                compute_diagnostics(&system, step, dt * step);
        #endif
                if (step == break_at_step)
                        break;
        }
        double t1 = MPI_Wtime();

        if (rank == 0)
                printf("Timing: %d %lf\n", N, t1-t0);

        if (step == break_at_step)
                write_density_mpi(&system, (char *)"density_mpi.dat.restart");
        else
                write_density_mpi(&system, (char *)"density_mpi.dat");

        #ifndef _PERF_
        if (rank == 0)
        {
                char diagnostics_filename[256];
                sprintf(diagnostics_filename, "diagnostics_mpi_%d.dat", procs);
                write_diagnostics(&system, diagnostics_filename);
        }
        #endif

        MPI_Finalize();
        return 0;
}
