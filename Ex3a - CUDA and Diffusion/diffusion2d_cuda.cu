#include <iostream>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>
#include <cassert>
#include <vector>
#include <cmath>
#include "timer.hpp"

typedef std::size_t size_type;

// used to specify the dimensions of a each Thread Block
static const int diffusion_block_x = 16;
static const int diffusion_block_y = 16;

__global__ void diffusion_kernel(float * rho_out, float const * rho, float fac, int N)
{
        /* We use a **grid-stride loop** approach (with two nested for loops, one for each grid dimension), 
         * in order to provide a generic implementation of the CUDA Kernel. 
         * This is needed to account for the case where the Kernel is launched using a grid whose 
         * total thread count is less than the total number of data cells (Ntot).
         * (In the "default" code where we use 16 x 16 Thread Blocks, this is not an issue, however
         * we chose to implement it, in order to provide a complete solution)
         *
         * We use the variable xstride to calculate the grid-stride loop's x-axis step
         * (in each grid's x-axis, there are gridDim.x Thread Blocks containing blockDim.x threads).
         * We calculate the grid-stride loop's y-axis step, in a similar manner.
         *
         * The values of xstride and ystride are used as a grid offset for the calculation of a
         * thread's global (x,y) index pair (stored in thread_x, thread_y).
         * 
         * The "default" code uses a square 2D array of Thread Blocks, where each Thread Block
         * is a 16 x 16 array of threads. Thus, xstride is equal to ystride. 
         * We generalize the code by using two separate variables, which allows for a non-square grid.
         *
         * Since the xstride variable, "runs through" a line of grids and the ystride "runs through"
         * a column of grids, we need to use **ystride** as a loop step for the grid-rows and **xstride**
         * as a loop step for the grid-columns. (Again, in the "default" case this difference does not exist)
         */
        int xstride = gridDim.x * blockDim.x;
        int ystride = gridDim.y * blockDim.y;

        for(int grid_i = 0; grid_i < N; grid_i+=ystride)
        {
                for(int grid_j = 0; grid_j < N; grid_j+=xstride)
                {
                        /* In order to efficiently calculate the new values of the rho_ 2D array,
                        *  we split the 2D data into small square tiles.
                        *
                        * Each square tile is a (diffusion_block_x x diffusion_block_y) array.
                        * (or a (blockDim.x x blockDim.y) array, since on kernel launch we set
                        * blockDim.x = diffusion_block_x and blockDim.y = diffusion_block_y)
                        *
                        * A data-tile block is updated by the threads of the Thread Block,
                        * having the same block coordinates as the data-tile block.
                        * Thus, we use one thread per square tile element (= one data cell of rho_)
                        *
                        * Therefore, we need to calculate the global (x,y) coordinates of each thread.
                        * 
                        * To compute the x coordinate of a thread, we calculate blockIdx.x * TILE_WIDTH,
                        * i.e. blockIdx.x * blockDim.x, which corresponds to the index of the 
                        * first element inside the corresponding Thread Block, in the x-"axis".
                        *
                        * Then, using threadIdx.x as an offset we "locate" the data cell to be updated
                        * by the current thread (that is the case when the grid covers the entire rho_ array).
                        *
                        * However, in order to account for the generalized case where the grid is smaller
                        * than the data-grid, we offset each **global** thread index by the current data-grid
                        * we are currently acting upon. The first Thread Block of each iteration is at
                        * position (grid_i, grid_j).
                        *                        
                        * We calculate the y coordinate of a thread, in the same manner.
                        *
                        * However, since the thread_x coordinate "runs through" a line and the thread_y
                        * coordinate "runs through" a column, the actual global index pair (x,y) is
                        * (thread_y, thread_x) and **not** (thread_x, thread_y).
                        */
                        int thread_x = grid_j + (blockIdx.x * blockDim.x + threadIdx.x); // column index
                        int thread_y = grid_i + (blockIdx.y * blockDim.y + threadIdx.y); // row index
 
                        if(thread_x < N && thread_y < N) // stay inside the N x N array's bounds
                        {
                                rho_out[thread_y * N + thread_x] =
                                        rho[thread_y * N + thread_x]
                                        +
                                        fac
                                        *
                                        (
                                                (thread_x == N-1 ? 0 : rho[thread_y * N + (thread_x + 1)])
                                                +
                                                (thread_x == 0 ? 0 : rho[thread_y * N + (thread_x - 1)])
                                                +
                                                (thread_y == N-1 ? 0 : rho[(thread_y + 1) * N + thread_x])
                                                +
                                                (thread_y == 0 ? 0 : rho[(thread_y - 1) * N + thread_x])
                                                -
                                                4 * rho[thread_y * N + thread_x]
                                        );
                        }
                }
        }
}

class Diffusion2D
{
        public:
                Diffusion2D(const float D, const float rmax, const float rmin, const size_type N)
                        : D_(D), rmax_(rmax), rmin_(rmin), N_(N), N_tot(N*N), d_rho_(0), d_rho_tmp_(0), rho_(N_tot)
                {
                        /// real space grid spacing
                        dr_ = (rmax_ - rmin_) / (N_ - 1);

                        /// dt < dx*dx / (4*D) for stability
                        dt_ = dr_ * dr_ / (6 * D_);

                        /// stencil factor
                        fac_ = dt_ * D_ / (dr_ * dr_);

                        /* Allocate global memory on Device
                         * Since we allocate **global** memory (shared among all GPU threads),
                         * there is no need to have "ghost cells"
                         * (i.e. buffers to share data between threads)
                         */
                        cudaMalloc((void **)&d_rho_, N_tot * sizeof(float));
                        cudaMalloc((void **)&d_rho_tmp_, N_tot * sizeof(float));

                        // set the allocated device memory blocks, to zero
                        cudaMemset(d_rho_, 0, N_tot * sizeof(float));
                        cudaMemset(d_rho_tmp_, 0, N_tot * sizeof(float));

                        InitializeSystem();
                }

                ~Diffusion2D()
                {
                        cudaFree(d_rho_tmp_);
                        cudaFree(d_rho_);
                }

                void PropagateDensity(int steps);

                float GetMoment()
                {
                        // Get data (rho_) from the GPU device
                        cudaMemcpy(&rho_[0], d_rho_, N_tot * sizeof(float), cudaMemcpyDeviceToHost);

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

                float *d_rho_, *d_rho_tmp_;
                mutable std::vector<float> rho_;
};

void Diffusion2D::WriteDensity(const std::string file_name) const
{
        // Get data (rho_) from the GPU device
        cudaMemcpy(&rho_[0], d_rho_, N_tot * sizeof(float), cudaMemcpyDeviceToHost);

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
        /// Dirichlet boundaries; central differences in space, forward Euler in time.

        // Define grid_size and block_size

        // Each Thread Block is a 16x16 2d array, i.e. each Thread Block contains 256 threads   
        dim3 block_size(diffusion_block_x, diffusion_block_y, 1);

        /* Each grid is a 2d array of Thread Blocks.
         * The grid's dimensions are (N_ / diffusion_block_x) x (N_ / diffusion_block_y).
         * Therefore, we seperate the total N_ x N_ array, into grids with the formentioned dimensions.
         * Also, the total number of threads blocks is again (N_ / diffusion_block_x) x (N_ / diffusion_block_y).
         */
        dim3 grid_size(N_ / diffusion_block_x, N_ / diffusion_block_y, 1);

        for (int s = 0; s < steps; ++s)
        {
                /* Kernel launch using the defined grid_size and block_size dim3 variables
                 * in the execution configuration.
                 */
                diffusion_kernel<<<grid_size, block_size>>>(d_rho_tmp_, d_rho_, fac_, N_);   
                
                /* Swap the addresses where the device memory pointers point to 
                 * (**NOT** the respective memory blocks' contents).
                 * The pointer swap is performed on the CPU.
                 * However, these pointers point to device memory,
                 * hence they cannot be dereferenced by Host code.
                 * On the next kernel launch, the GPU code will "see"
                 * the changes made by std::swap.
                 */            
                swap(d_rho_, d_rho_tmp_);
                time_ += dt_;
        }
}

void Diffusion2D::InitializeSystem()
{
        time_ = 0;

        /// initialize rho(x,y,t=0)
        float bound = 1./2;

        // Initialize the N x N array (this happens on CPU memory)
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
        // Copy the initialized N x N array to the GPU device
        cudaMemcpy(d_rho_, &rho_[0], rho_.size() * sizeof(float), cudaMemcpyHostToDevice);
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
                cudaDeviceSynchronize(); // wait for the CUDA kernel to finish
                time = System.GetTime();
                float moment = System.GetMoment();
                std::cout << time << '\t' << moment << std::endl;
        }

        runtime.stop();

        double elapsed = runtime.get_timing();

        std::cerr << argv[0] << "\t N=" <<N_ << "\t time=" << elapsed << "s" << std::endl;

        std::string density_file = "Density_cuda.dat";
        System.WriteDensity(density_file);

        return 0;
}
