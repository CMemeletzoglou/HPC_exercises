#include <iostream>
#include <tuple>
#include "timer.hpp"

typedef struct ant_s
{
        std::tuple<int, int> position;
} ant_t;

typedef struct grid_cell_s
{
        float pher_amount;
        bool ant_present;
        ant_t ant;
} grid_cell_t;

class AntColonySystem
{
        public:
                AntColonySystem(const std::size_t N) : N(N), N_tot(N*N)
                {
                        // allocate memory for the grid
                        grid = new (std::nothrow) grid_cell_t[N_tot];
                }

                ~AntColonySystem()
                {
                        // deallocate the grid's memory block
                        delete[] grid;
                }

                float get_time() const { return curr_time; }

                void advance_system(float time);

        private:
                void initialize_system();

                const std::size_t N, N_tot;

                float curr_time;

                grid_cell_t *grid;
};

void AntColonySystem::initialize_system()
{
        // assume pherormone is in [0,1], with 0 being completely empty
        // and 1 corresponding to completely full of pherormone

        srand(time(NULL));

        for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                        grid[i * N + j].pher_amount =  static_cast<float> (rand()) / static_cast<float> (RAND_MAX);
                        grid[i * N + j].ant_present = rand() % 2;

                        /* if the cell has an ant assigned to it, set the ant's coordinates 
                         * to that of the cell
                         */
                        if(grid[i * N + j].ant_present)
                                grid[i * N + j].ant.position = std::make_tuple(i, j);
                }
}

void AntColonySystem::advance_system(float time)
{
        /* Assume that an ant can move only in a "cross" fashion, i.e.
         * one cell forward, backward, left, right.
         */

        // iterate for each ant (?)
        for (int i = 0; i < N_tot; i++)
        {
                // check if the ant can move etc
        }
}








int main(int argc, char **argv)
{
        if(argc != 2)
        {
                std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
                return 1;
        }

        const std::size_t N = 1 << std::atoi(argv[1]);
        const float tmax = 0.01; // max sim time (value?)

        // create a N x N AntColonySystem
        AntColonySystem system(N);

        float time = 0;
        timer runtime;

        runtime.start();

        while(time < tmax)
        {
                time = system.get_time();
                
                // call sim function
                system.advance_system(time);
        }

        runtime.stop();

        double time_elapsed = runtime.get_timing();

        std::cerr << argv[0] << "\t N=" << N << "\t time=" << time_elapsed << "s" << std::endl;

        return 0;
}