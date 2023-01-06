#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"

typedef std::size_t size_type;

class Diffusion2D
{
        public:
                Diffusion2D(const float D, const float rmax, const float rmin, const size_type N)
                        : D_(D), rmax_(rmax), rmin_(rmin), N_(N)
                {
                        N_tot = N_*N_;

                        /// real space grid spacing
                        dr_ = (rmax_ - rmin_) / (N_ - 1);

                        /// dt < dx*dx / (4*D) for stability
                        dt_ = dr_ * dr_ / (6 * D_);

                        /// stencil factor
                        fac_ = dt_ * D_ / (dr_ * dr_);

                        rho_ = new float[N_tot];
                        rho_tmp = new float[N_tot];

                        std::fill(rho_, rho_+N_tot,0.0);
                        std::fill(rho_tmp, rho_tmp+N_tot,0.0);

                        InitializeSystem();
                }

                ~Diffusion2D()
                {
                        delete[] rho_;
                        delete[] rho_tmp;
                }

                void PropagateDensity(int steps);

                float GetMoment()
                {
                        float sum = 0;

                        for(size_type i = 0; i < N_; ++i)
                                for(size_type j = 0; j < N_; ++j)
                                {
                                        float x = j*dr_ + rmin_;
                                        float y = i*dr_ + rmin_;
                                        sum += rho_[i*N_ + j] * (x*x + y*y);
                                }

                        return dr_*dr_*sum;
                }
                float GetTime() const {return time_;}

                void WriteDensity(const std::string file_name) const;

        private:
                void InitializeSystem();

                const float D_, rmax_, rmin_;
                const size_type N_;
                size_type N_tot;

                float dr_, dt_, fac_;

                float time_;

                float *rho_, *rho_tmp;
};

void Diffusion2D::WriteDensity(const std::string file_name) const
{
        std::ofstream out_file;
        out_file.open(file_name.c_str(), std::ios::out);
        if(out_file.good())
        {
                for(size_type i = 0; i < N_; ++i)
                {
                        for(size_type j = 0; j < N_; ++j)
                                out_file << (i*dr_+rmin_) << '\t' << (j*dr_+rmin_) << '\t' << rho_[i*N_ + j] << "\n";

                        out_file << "\n";
                }
        }

        out_file.close();
}

void Diffusion2D::PropagateDensity(int steps)
{
        using std::swap;
        /// Dirichlet boundaries; central differences in space, forward Euler
        /// in time

        for(int s = 0; s < steps; ++s)
        {
                for(size_type i = 0; i < N_; ++i)
                        for(size_type j = 0; j < N_; ++j)
                        {
                                rho_tmp[i*N_ + j] =
                                        rho_[i*N_ + j]
                                        +
                                        fac_
                                        *
                                        (
                                                (j == N_-1 ? 0 : rho_[i*N_ + (j+1)])
                                                +
                                                (j == 0 ? 0 : rho_[i*N_ + (j-1)])
                                                +
                                                (i == N_-1 ? 0 : rho_[(i+1)*N_ + j])
                                                +
                                                (i == 0 ? 0 : rho_[(i-1)*N_ + j])
                                                -
                                                4*rho_[i*N_ + j]
                                        );
                        }
                swap(rho_tmp,rho_);

                time_ += dt_;
        }
}

void Diffusion2D::InitializeSystem()
{
        time_ = 0;

        /// initialize rho(x,y,t=0)
        float bound = 1./2;

        for(size_type i = 0; i < N_; ++i)
        {
                for(size_type j = 0; j < N_; ++j)
                {
                        if(std::fabs(i*dr_+rmin_) < bound && std::fabs(j*dr_+rmin_) < bound)
                        {
                                rho_[i*N_ + j] = 1;
                        }
                        else
                        {
                                rho_[i*N_ + j] = 0;
                        }
                }
        }
}

int main(int argc, char* argv[])
{
        if(argc != 2)
        {
                std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
                return 1;
        }

        const float D = 1;
        const float tmax = 0.01;
        const float rmax = 1;
        const float rmin = -1;

        const size_type N_ = 1 << std::atoi(argv[1]);
        const int steps_between_measurements = 100;

        Diffusion2D System(D, rmax, rmin, N_);

        float time = 0;

        timer runtime;
        runtime.start();

        while(time < tmax)
        {
                System.PropagateDensity(steps_between_measurements);
                time = System.GetTime();
                float moment = System.GetMoment();
                std::cout << time << '\t' << moment << std::endl;
        }

        runtime.stop();

        double elapsed = runtime.get_timing();

        std::cerr << argv[0] << "\t N=" <<N_ << "\t time=" << elapsed << "s" << std::endl;

        std::string density_file = "Density.dat";
        System.WriteDensity(density_file);

        return 0;
}
