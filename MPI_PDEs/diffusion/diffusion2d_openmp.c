#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif


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
        int rank_;
        double *rho_, *rho_tmp_;
        Diagnostics *diag_;
} Diffusion2D;

void initialize_density(Diffusion2D *D2D)
{
        int real_N_ = D2D->real_N_;
        int N_ = D2D->N_;
        double *rho_ = D2D->rho_;
        double dr_ = D2D->dr_;
        double L_ = D2D->L_;

        /// Initialize rho(x, y, t=0).
        double bound = 0.25 * L_;

        #pragma omp parallel for
        for (int i = 1; i <= N_; ++i)
        {
                for (int j = 1; j <= N_; ++j)
                {
                        if (fabs((i - 1) * dr_ - 0.5 * L_) < bound
                                && fabs((j - 1) * dr_ - 0.5 * L_) < bound)
                        {
                                rho_[i * real_N_ + j] = 1;
                        } 
                        else
                        {
                                rho_[i * real_N_ + j] = 0;
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
                const int rank)
{
        D2D->D_ = D;
        D2D->L_ = L;
        D2D->N_ = N;
        D2D->T_ = T;
        D2D->dt_ = dt;
        D2D->rank_ = rank;

        // Real space grid spacing.
        D2D->dr_ = D2D->L_ / (D2D->N_ - 1);

        // Stencil factor.
        D2D->fac_ = D2D->dt_ * D2D->D_ / (D2D->dr_ * D2D->dr_);

        // Actual dimension of a row (+2 for the ghost cells).
        D2D->real_N_ = N + 2;

        // Total number of cells.
        D2D->Ntot_ = (D2D->N_ + 2) * (D2D->N_ + 2);

        D2D->rho_ = (double *)calloc(D2D->Ntot_, sizeof(double));
        D2D->rho_tmp_ = (double *)calloc(D2D->Ntot_, sizeof(double));
        D2D->diag_ = (Diagnostics *)calloc(D2D->T_, sizeof(Diagnostics));

        // Check that the timestep satisfies the restriction for stability.
        printf("timestep from stability condition is %lf\n", D2D->dr_ * D2D->dr_ / (4.0 * D2D->D_));

        initialize_density(D2D);
}

void advance(Diffusion2D *D2D)
{
        int N_ = D2D->N_;
        int real_N_ = D2D->real_N_;
        double *rho_ = D2D->rho_;
        double *rho_tmp_ = D2D->rho_tmp_;
        double fac_ = D2D->fac_;

        // Central differences in space, forward Euler in time with Dirichlet
        // boundaries.
        #pragma omp parallel for
        for (int i = 1; i <= N_; ++i)
        {
                for (int j = 1; j <= N_; ++j)
                {
                        rho_tmp_[i * real_N_ + j] =
                                rho_[i * real_N_ + j]
                                + fac_ * (rho_[i * real_N_ + (j + 1)]
                                        + rho_[i * real_N_ + (j - 1)]
                                        + rho_[(i + 1) * real_N_ + j]
                                        + rho_[(i - 1) * real_N_ + j]
                                        - 4.0 * rho_[i * real_N_ + j]);
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
        double *rho_ = D2D->rho_;
        double dr_ = D2D->dr_;

        double heat = 0.0;
        for (int i = 1; i <= N_; ++i)
                for (int j = 1; j <= N_; ++j)
                        heat += dr_ * dr_ * rho_[i * real_N_ + j];

        #if DEBUG
        printf("t = %lf heat = %lf\n", t, heat);
        #endif
        D2D->diag_[step].time = t;
        D2D->diag_[step].heat = heat;
}

void write_diagnostics(Diffusion2D *D2D, const char *filename)
{

        FILE *out_file = fopen(filename, "w");
        for (int i = 0; i < D2D->T_; i++)
                fprintf(out_file, "%f\t%f\n", D2D->diag_[i].time, D2D->diag_[i].heat);
        fclose(out_file);
}


#if !defined(_OPENMP)
int omp_get_num_threads()
{
        return 1;
}

#include <sys/time.h>
double omp_get_wtime()
{
        struct timeval t;
        gettimeofday(&t, NULL);
        return (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
}
#endif

int main(int argc, char* argv[])
{
        if (argc < 6)
        {
                printf("Usage: %s D L T N dt\n", argv[0]);
                return 1;
        }

        int nthreads = 1;
        #pragma omp parallel
        {
                #pragma omp master
                nthreads = omp_get_num_threads();
        }
        printf("Running with %d threads\n", nthreads);

        const double D = atof(argv[1]);		// coefficient = 1
        const double L = atoi(argv[2]);		// domain [-L, L]  (L=1)
        const int N = atoi(argv[3]);		// NxN grid (N=1024)
        const int T = atoi(argv[4]);		// number of timesteps (T=1000)
        const double dt = atof(argv[5]);	// dt = 0.00000001

        Diffusion2D system;

        init(&system, D, L, N, T, dt, 0);

        double t0 = omp_get_wtime();
        for (int step = 0; step < T; ++step)
        {
                advance(&system);
                #ifndef _PERF_
                        compute_diagnostics(&system, step, dt * step);
                #endif
        }
        double t1 = omp_get_wtime();

        printf("Timing: %d %lf\n", N, t1-t0);

        #ifndef _PERF_
                char diagnostics_filename[256];
        #if !defined(_OPENMP)
                strcpy(diagnostics_filename, "diagnostics_serial.dat");
        #else
                sprintf(diagnostics_filename, "diagnostics_openmp_%d.dat", nthreads);
        #endif
                write_diagnostics(&system, diagnostics_filename);
        #endif

        return 0;
}
