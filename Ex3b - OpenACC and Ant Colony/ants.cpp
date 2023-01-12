#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <assert.h>
#include "timer.hpp"


typedef struct grid_cell_s
{
        float pher_amount;
        int cell_ants = 0;
} grid_cell_t;

class AntColonySystem
{
        public:
                AntColonySystem(const std::size_t N, const int ant_count) : N(N), N_tot(N * N), ant_count(ant_count)
                {
                        // allocate memory for the grid
                        grid = new grid_cell_t[N_tot];
                        grid_tmp = new grid_cell_t[N_tot];
                        initialize_system();
                }

                ~AntColonySystem()
                {
                        // deallocate the grid's memory block
                        delete[] grid;
                        delete[] grid_tmp;
                }

                float get_time() const { return curr_time; }

                void advance_system(const int);

                void write_grid_status(const std::string) const;
        private:
                void initialize_system();

                void move_ants();
                void update_pheromone();

                int choose_next_cell(const std::vector<grid_cell_t*> vec);

                const std::size_t N, N_tot;
                int ant_count;

                const float dt = 1e-3; 
                float curr_time = 0.0f;

                grid_cell_t *grid, *grid_tmp;
};

void AntColonySystem::write_grid_status(const std::string filename) const
{
        std::ofstream out_file;
        out_file.open(filename.c_str(), std::ios::out);
        if(out_file.good())
        {
                for (std::size_t i = 0; i < N; i++)
                        for (std::size_t j = 0; j < N; j++)
                                out_file << "Cell[" << i << ", " << j << "] : "
                                         << "\t"
                                         << "pheromone = "
                                         << grid[i * N + j].pher_amount << "\t\t\t"
                                         << "Has ant : "
                                         << grid[i * N + j].cell_ants << "\n";

                out_file << "\n\n**************************Ants**************************\n\n";

                int cnt = 0;
                for (std::size_t i = 0; i < N; i++)
                    for (std::size_t j = 0; j < N; j++)
                        for(int a = 0; a < grid[i*N + j].cell_ants; a++)
                                out_file << "Ant " << cnt++ << "\t Position : (" << i
                                                << ", " << j << ")\n";

                out_file << "\nEND GRID STATUS\n\n";
        }

        out_file.close();
}

void AntColonySystem::initialize_system()
{
        // assume pheromone is in [0,1], with 0 being completely empty
        // and 1 corresponding to completely full of pheromone

        // Initialize the amount of pheromone in each cell - randomly
        for (std::size_t i = 0; i < N; i++)
                for (std::size_t j = 0; j < N; j++)
                        grid[i * N + j].pher_amount = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

        // Place the ant_count ants inside randomly on the grid
        int ants_placed = 0;
        while(ants_placed < ant_count)
        {
                int rand_i = rand() % N;
                int rand_j = rand() % N;
                if(grid[rand_i * N + rand_j].cell_ants == 0)
                {
                        grid[rand_i * N + rand_j].cell_ants++; 
                        ants_placed++;
                }
        }      
        assert(ants_placed == ant_count);
}

// It either returns the index of the neighbouring cell to move onto
// or -1 to stay where you are
int AntColonySystem::choose_next_cell(const std::vector<grid_cell_t*> vec)
{
        std::vector<int> max_vec;
        int first_empty_cell_idx = -1;
        
        // find the first valid and non-occupied neighboring cell
        auto it = std::find_if(vec.begin(), vec.end(), [&](auto const &el)
                               { return (el != nullptr && el->cell_ants == 0); }); 

        if (it != vec.end())
                first_empty_cell_idx = std::distance(vec.begin(), it); // calculate index from iterator

        // If there is no empty neighbouring cell or the first empty cell is the last one
        if (first_empty_cell_idx == -1 || first_empty_cell_idx == static_cast<int>(vec.size() - 1))
                return first_empty_cell_idx;
        
        std::size_t max_cell_idx = first_empty_cell_idx;
        max_vec.push_back(max_cell_idx);

        // There are more than 1 neighbouring cells that are not currently occupied
        // Pick the one with the maximum amount of pheromone and move there
        for (std::size_t i = first_empty_cell_idx + 1; i < vec.size(); i++)
        {
                // skip over dummy or occupied cells
                if (vec[i] == nullptr || vec[i]->cell_ants) continue;

                // Preserve all the neighbours that have the same max amount of pheromone
                // so you may later decide in which one to move onto
                // if(vec[max_cell_idx]->pher_amount == vec[i]->pher_amount)
                if(std::abs(vec[max_cell_idx]->pher_amount - vec[i]->pher_amount) < 1e-7)
                        max_vec.push_back(i);
                else if(vec[max_cell_idx]->pher_amount < vec[i]->pher_amount)
                {
                        max_cell_idx = i; // set the new max
                        max_vec.clear(); // reset the vector
                        max_vec.push_back(i);
                }
        }

        /* Choose the cell with the maximum amount of pheromone to move onto.
         * The choice is either random  (if all the neighboring cells have the same amount of
         * pheromone) or not random (if there is only one cell with the max amount of pheromone.
         * In this case, valid_count = 1, therefore mod valid_count gives as 0, so we select
         * the first element of the max_vec, which is the one max pheromone cell)
         */
        return max_vec[rand() % max_vec.size()];
}

void AntColonySystem::move_ants()
{
        /* Assume that an ant can move only in a "cross" fashion, i.e.
         * one cell forward, backward, left, right.
         */
        std::vector<grid_cell_t*> neigh_cells(4, nullptr);
        std::vector<int> neigh_cell_idx(4, -1);
        int nneigh = 0;

        // Iterate over each grid cell
        for (std::size_t i = 0; i < N; i++)
                for (std::size_t j = 0; j < N; j++)
                {
                        std::fill(neigh_cells.begin(), neigh_cells.end(), nullptr);
                        std::fill(neigh_cell_idx.begin(), neigh_cell_idx.end(), -1);
                        nneigh = 0;
                        
                        // If there are ant(s) on the grid cell move one to a new position
                        if (grid[i*N + j].cell_ants)
                        {
                                /* Need to check if an ant has an upper, lower, left or right cell, because
                                * the ants on the grid's boundaries don't have all kinds of neighbors.
                                */
                                if (i > 1) // There is a cell above
                                {
                                        neigh_cell_idx[0] = (i-1)*N + j;
                                        neigh_cells[0] = &(grid[neigh_cell_idx[0]]);
                                        nneigh++;
                                }
                                if (j < N-1) // There is a cell on the right
                                {
                                        neigh_cell_idx[1] = i*N + (j+1);
                                        neigh_cells[1] = &(grid[neigh_cell_idx[1]]);
                                        nneigh++;
                                }
                                if (j > 1) // There is a cell on the left
                                {
                                        neigh_cell_idx[2] = i*N + (j-1);
                                        neigh_cells[2] = &(grid[neigh_cell_idx[2]]);
                                        nneigh++;
                                }
                                if (i < N-1) // There is a cell below
                                {
                                        neigh_cell_idx[3] = (i+1)*N + j;
                                        neigh_cells[3] = &(grid[neigh_cell_idx[3]]);
                                        nneigh++;
                                }

                                int next_cell_idx = choose_next_cell(neigh_cells);
                                if(next_cell_idx != -1) // there is a cell the current ant can move to
                                {
                                        // get the **global** index of the next cell, using the neighbor-local index
                                        next_cell_idx = neigh_cell_idx[next_cell_idx];
                                        grid_tmp[i*N + j].cell_ants = grid[i*N + j].cell_ants - 1; // vacate the current cell

                                        float pher_loss = grid[i*N + j].pher_amount / 2;
                                        grid_tmp[i*N + j].pher_amount += grid[i*N + j].pher_amount - pher_loss;

                                        float pher_incr = pher_loss / nneigh;
                                        for (std::size_t i = 0; i < neigh_cells.size(); i++)
                                                if (neigh_cells[i] != nullptr) 
                                                        grid_tmp[neigh_cell_idx[i]].pher_amount += pher_incr;

                                        // move to the chosen cell
                                        grid_tmp[next_cell_idx].cell_ants++; // Watch out when parallelizing
                                }
                                else // nowhere to move, the ant will stay in the current position
                                {
                                        next_cell_idx = i*N + j;
                                        grid_tmp[next_cell_idx].pher_amount += grid[next_cell_idx].pher_amount;
                                        grid_tmp[next_cell_idx].cell_ants = grid[next_cell_idx].cell_ants;
                                }
                        }
                        else // the current cell does not have an ant
                                grid_tmp[i*N + j].pher_amount += grid[i*N + j].pher_amount;
                }
}

void AntColonySystem::update_pheromone()
{
        /* This function updates the pheromone of each cell, **after** the ants have moved.
         * Assume that each cell occupied by an ant, gets a pheromone increase of 5%
         * and empty cells lose the 10% of their pheromone amount
         */
        for (std::size_t i = 0; i < N; i++)
                for (std::size_t j = 0; j < N; j++)
                {
                        if(grid[i * N + j].cell_ants)
                                grid[i*N + j].pher_amount += 0.05 * grid[i*N + j].pher_amount * grid[i*N + j].cell_ants;
                        else
                                grid[i*N + j].pher_amount -= 0.001 * grid[i*N + j].pher_amount;

                        if (grid[i*N + j].pher_amount < 0)
                                grid[i*N + j].pher_amount = 0;
                        else if (grid[i*N + j].pher_amount > 100)
                                grid[i*N + j].pher_amount = 100;
                }
}

void AntColonySystem::advance_system(const int steps)
{
        grid_cell_t t;
        t.cell_ants = 0;
        t.pher_amount = 0;
        
        for (int s = 0; s < steps; s++)
        {
                std::fill(grid_tmp, grid_tmp + N_tot, t); // fill grid_tmp with dummy cells

                move_ants();                
                std::swap(grid, grid_tmp);
                
                update_pheromone();
                //then update the pheromone amounts of each cell (vacated or newly occupied)
                //update the cells occupied by ants and the empty cells

                curr_time += dt; // advance time
        }
}

int main(int argc, char **argv)
{
        if(argc != 3)
        {
                std::cerr << "Usage: " << argv[0] << " <log2(size)>" << " ant count" << std::endl;
                return 1;
        }
        
        srand(time(NULL));

        const std::size_t N = 1 << std::atoi(argv[1]); // grid size is 2^argv[1]
        const int ant_count = std::atoi(argv[2]);
        const float tmax = 0.01; // max sim time (value?)
        const int steps_between_measurements = 100;

        // create a N x N AntColonySystem
        AntColonySystem system(N, ant_count);
        
        float time = 0;
        timer runtime;

        runtime.start();

        while(time < tmax)
        {
                // call sim function
                system.advance_system(steps_between_measurements);
                // advance time                
                time = system.get_time();
        }

        runtime.stop();

        double time_elapsed = runtime.get_timing();

        std::cerr << argv[0] << "\t N=" << N << "\t time=" << time_elapsed << "s" << std::endl;

        const std::string ant_grid_filename = "Ant_grid_serial_2.dat";
        system.write_grid_status(ant_grid_filename); // write logging data

        return 0;
}