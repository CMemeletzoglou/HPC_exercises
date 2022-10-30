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
    int N_ = D2D->N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    double L_ = D2D->L_;
    int rank_ = D2D->rank_;
    int procs_ = D2D->procs_;
    int gi;

    /// Initialize rho(x, y, t=0).
    double bound = 0.25 * L_;

    for (int i = 1; i <= local_N_; ++i) {
        gi = rank_ * (N_ / procs_) + i; // convert local index to global index
        for (int j = 1; j <= N_; ++j) {
            if (fabs((gi - 1)*dr_ - 0.5*L_) < bound && fabs((j-1)*dr_ - 0.5*L_) < bound) {
                rho_[i*real_N_ + j] = 1;
            } else {
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

    // Number of rows per process.
    D2D->local_N_ = D2D->N_ / D2D->procs_;

    // Small correction for the last process.
    if (D2D->rank_ == D2D->procs_ - 1)
        D2D->local_N_ += D2D->N_ % D2D->procs_;

    // Actual dimension of a row (+2 for the ghost cells).
    D2D->real_N_ = D2D->N_ + 2;

    // Total number of cells.
    D2D->Ntot_ = (D2D->local_N_ + 2) * (D2D->N_ + 2);

    D2D->rho_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->rho_tmp_ = (double *)calloc(D2D->Ntot_, sizeof(double));
    D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

    // Check that the timestep satisfies the restriction for stability.
    if (D2D->rank_ == 0)
        printf("timestep from stability condition is %e\n", D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

    initialize_density(D2D);
}

void advance(Diffusion2D *D2D)
{
    int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double *rho_tmp_ = D2D->rho_tmp_;
    double fac_ = D2D->fac_;
    int rank_ = D2D->rank_;
    int procs_ = D2D->procs_;

    // Non-blocking MPI
    MPI_Request req[4];
    MPI_Status status[4];

    int prev_rank = rank_ - 1;
    int next_rank = rank_ + 1;

    // Exchange ALL necessary ghost cells with neighboring ranks.
    if (prev_rank >= 0) {
        // TODO:MPI
        MPI_Irecv(&rho_[           0*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[0]);
        MPI_Isend(&rho_[           1*real_N_+1], N_, MPI_DOUBLE, prev_rank, 100, MPI_COMM_WORLD, &req[1]);
    }
    else {
        // the purpose of this part will become
        // clear when using asynchronous communication.
        req[0] = MPI_REQUEST_NULL;
        req[1] = MPI_REQUEST_NULL;
    }

    if (next_rank < procs_) {
        // TODO:MPI
        MPI_Irecv(&rho_[(local_N_+1)*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[2]);
        MPI_Isend(&rho_[    local_N_*real_N_+1], N_, MPI_DOUBLE, next_rank, 100, MPI_COMM_WORLD, &req[3]);
    }
    else {
        // the purpose of this part will become 
        // clear when using asynchronous communication.
        req[2] = MPI_REQUEST_NULL;
        req[3] = MPI_REQUEST_NULL;
    }

    // Central differences in space, forward Euler in time with Dirichlet
    // boundaries.
    for (int i = 2; i < local_N_; ++i) {
        for (int j = 1; j <= N_; ++j) {
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

    // ensure boundaries have arrived
    MPI_Waitall(4, req, status);

    // Update the first and the last rows of each rank.
    for (int i = 1; i <= local_N_; i += local_N_- 1) {
        for (int j = 1; j <= N_; ++j) {
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
}

void compute_diagnostics(Diffusion2D *D2D, const int step, const double t)
{
    int N_ = D2D->N_;
    int real_N_ = D2D->real_N_;
    int local_N_ = D2D->local_N_;
    double *rho_ = D2D->rho_;
    double dr_ = D2D->dr_;
    int rank_ = D2D->rank_;

    double heat = 0.0;
    for(int i = 1; i <= local_N_; ++i)
        for(int j = 1; j <= N_; ++j)
            heat += rho_[i*real_N_ + j] * dr_ * dr_;

    // TODO:MPI, reduce heat (sum)
    MPI_Reduce(rank_ == 0? MPI_IN_PLACE: &heat, &heat, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 


    if (rank_ == 0) {
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


void write_density_mpi(Diffusion2D *D2D, char *filename)
{
    // TODO: add your MPI I/O code here, write rho_ to disk
}

void read_density_mpi(Diffusion2D *D2D, char *filename)
{
    // TODO: add your MPI I/O code here, load rho_ from disk
}

int main(int argc, char* argv[])
{
    int rank, procs;
    //TODO:MPI Initialize MPI, number of ranks (rank) and number of processes (nprocs) involved in the communicator
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    const double D  = 1;
    const double L  = 1;
    const int  N  = 1024*4;
    const int T = 1000;
    const double dt = 1e-9;

    Diffusion2D system;

    init(&system, D, L, N, T, dt, rank, procs);

    int restart_step = 0;
    if (argc == 2) restart_step = atoi(argv[2]);

    int break_at_step = -100;
    int first_step = 0;
    if (restart_step > 0)
    {
        read_density_mpi(&system, (char *)"density_mpi.dat.restart");
        first_step = restart_step;
    }

    double t0 = MPI_Wtime();
    int step;
    for (step = first_step; step < T; ++step) {
        advance(&system);
#ifndef _PERF_
        compute_diagnostics(&system, step, dt * step);
#endif
        if (step == break_at_step) break;
    }
    double t1 = MPI_Wtime();

    if (rank == 0)
        printf("Timing: %d %lf\n", N, t1-t0);


    if (step == break_at_step)
        write_density_mpi(&system, (char *)"density_mpi.dat.restart");
    else
        write_density_mpi(&system, (char *)"density_mpi.dat");

#ifndef _PERF_
    if (rank == 0) {
        char diagnostics_filename[256];
        sprintf(diagnostics_filename, "diagnostics_mpi_%d.dat", procs);
        write_diagnostics(&system, diagnostics_filename);
    }
#endif

    MPI_Finalize();
    return 0;
}
