#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <assert.h>
#include "timer.hpp"

typedef struct ant_s
{
        std::tuple<int, int> position; // the ant position's global index
} ant_t;

typedef struct grid_cell_s
{
        std::tuple<int, int> coords;
        float pher_amount;
        bool ant_present = false;
} grid_cell_t;

class AntColonySystem
{
        public:
                AntColonySystem(const std::size_t N, const int ant_count) : N(N), N_tot(N * N), ant_count(ant_count)
                {
                        // allocate memory for the grid
                        grid = new (std::nothrow) grid_cell_t[N_tot];
                        ants.reserve(ant_count);
                        initialize_system();
                }

                ~AntColonySystem()
                {
                        // deallocate the grid's memory block
                        delete[] grid;
                }

                float get_time() const { return curr_time; }

                void advance_system(const int);

                void write_grid_status(const std::string) const;
        private:
                void initialize_system();

                void move_ants();
                int choose_next_cell(const grid_cell_t **arr, int n = 4);
                void update_pheromone();

                const std::size_t N, N_tot;
                int ant_count;

                const float dt = 1e-3;
                float curr_time = 0.0f;

                grid_cell_t *grid;
                std::vector<ant_t> ants;
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
                                         << grid[i * N + j].ant_present << "\n";

                out_file << "\n\n**************************Ants**************************\n\n";

                for (int i = 0; i < ant_count; i++)
                        out_file << "Ant " << i << "\t Position : (" << std::get<0>(ants[i].position)
                                 << ", " << std::get<1>(ants[i].position) << ")\n";

                out_file << "\nEND GRID STATUS\n\n";
        }

        out_file.close();
}

void AntColonySystem::initialize_system()
{
        // assume pheromone is in [0,1], with 0 being completely empty
        // and 1 corresponding to completely full of pheromone
        for (std::size_t i = 0; i < N; i++)
                for (std::size_t j = 0; j < N; j++)
                {
                        grid[i * N + j].coords = std::make_tuple(i, j);
                        grid[i * N + j].pher_amount = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                }

        int ants_placed = 0;
        while(ants_placed < ant_count)
        {
                int rand_i = rand() % N;
                int rand_j = rand() % N;
                if(!grid[rand_i * N + rand_j].ant_present)
                {
                        ant_t ant; // create a new ant
                        ant.position = std::make_tuple(rand_i, rand_j); // add its coordinates
                        grid[rand_i * N + rand_j].ant_present = true; 

                        ants.emplace_back(ant); // add ant to ants vector
                        ants_placed++;
                }
        }        
}

void AntColonySystem::move_ants()
{
        /* Assume that an ant can move only in a "cross" fashion, i.e.
         * one cell forward, backward, left, right.
         */
        std::vector<grid_cell_t*> neigh_cells(4, nullptr);
        std::vector<int> neigh_cell_idx(4, -1);
        int num_neigh = 0;

        for (int curr_ant = 0; curr_ant < ant_count; curr_ant++) // for each ant
        {
                num_neigh = 0;

                // get the ant's (i, j) coordinate vector
                int ant_i = std::get<0>(ants[curr_ant].position);
                int ant_j = std::get<1>(ants[curr_ant].position);

                /* Need to check for existence of left, right, upper and lower cells
                 * since the ants placed on the grid's boundaries, do not have all
                 * kinds of neighbors.
                 * Store the neighboring cells into neigh_cell using the order :
                 * left, right, upper, below.
                 * Store the global index of each neighboring cell into the 
                 * corresponding cell of array neigh_cell_idx.
                 */
                if(ant_j - 1 >= 0) // if there exists a left cell
                {
                        neigh_cell_idx[0] = ant_i * N + (ant_j - 1);
                        neigh_cells[0] = &(grid[neigh_cell_idx[0]]);
                        num_neigh++;
                }
                if (ant_j + 1 <= static_cast<int>(N) - 1) // if there exists a right cell
                {
                        neigh_cell_idx[1] = ant_i * N + (ant_j + 1);
                        neigh_cells[1] = &(grid[neigh_cell_idx[1]]);
                        num_neigh++;
                }                        
                if (ant_i - 1 >= 0) // if there exists an upper cell
                {
                        neigh_cell_idx[2]= (ant_i - 1) * N + ant_j;
                        neigh_cells[2] = &(grid[neigh_cell_idx[2]]);
                        num_neigh++;
                }
                if (ant_i + 1 <= static_cast<int>(N) - 1) // if there exists a below cell
                {
                        neigh_cell_idx[3] = (ant_i + 1) * N + ant_j;
                        neigh_cells[3] = &(grid[neigh_cell_idx[3]]);
                        num_neigh++;
                }

                int next_cell_idx = choose_next_cell((const grid_cell_t**)&neigh_cells, 4);
                if(next_cell_idx != -1) // there is a cell the current ant can move to
                {
                        // get the **global** index of the next cell, using the neighbor-local index
                        next_cell_idx = neigh_cell_idx[next_cell_idx]; 

                        // vacate the current cell
                        grid[ant_i * N + ant_j].ant_present = false;

                        /* Update the pheromone amount on the newly vacated cell
                         * (it loses half of its current pheromone)
                         */
                        float pher_loss = grid[ant_i * N + ant_j].pher_amount / 2;
                        grid[ant_i * N + ant_j].pher_amount -= pher_loss;
                        
                        // pheromone increase for the neighboring cells
                        float pher_incr = pher_loss / num_neigh; 
                        
                        //distribute the pheromone loss, equally to all 4 neighboring cells
                        for (std::size_t i = 0; i < neigh_cells.size(); i++)
                                if(neigh_cells[i] != nullptr)
                                        grid[neigh_cell_idx[i]].pher_amount += pher_incr;
                        
                        // move ant to the chosen cell
                        ants[curr_ant].position = grid[next_cell_idx].coords;
                        // occupy the new cell
                        grid[next_cell_idx].ant_present = true;
                }              
                else // nowhere to move, the ant will stay in the current position
                        next_cell_idx = ant_i * N + ant_j;
        }
}

// It either returns the index of the neighbouring cell to move onto
// or -1 to stay where you are
int AntColonySystem::choose_next_cell(const grid_cell_t** arr, int n)
{
        std::vector<int> max_vec;
        int first_empty_cell_idx = -1;
        
        // find the first valid and non-occupied neighboring cell
        for (int i = 0; i < n; i++)
        {
                if(arr[i] != nullptr && !(arr[i]->ant_present))
                {
                        first_empty_cell_idx = -1; 
                        break;
                }
        }

        // If there is no empty neighbouring cell or the first empty cell is the last one
        if (first_empty_cell_idx == -1 || first_empty_cell_idx == static_cast<int>(n - 1))
                return first_empty_cell_idx;
        
        std::size_t max_cell_idx = first_empty_cell_idx;
        max_vec.push_back(max_cell_idx);

        // There are more than 1 neighbouring cells that are not currently occupied
        // Pick the one with the maximum amount of pheromone and move there
        for (int i = first_empty_cell_idx + 1; i < n; i++)
        {
                // skip over dummy or occupied cells
                if (arr[i] == nullptr || arr[i]->ant_present) continue;

                // Preserve all the neighbours that have the same max amount of pheromone
                // so you may later decide in which one to move onto
                // if(arr[max_cell_idx]->pher_amount == arr[i]->pher_amount)
                if(std::abs(arr[max_cell_idx]->pher_amount - arr[i]->pher_amount) < 1e-7)
                        max_vec.push_back(i);
                else if(arr[max_cell_idx]->pher_amount < arr[i]->pher_amount)
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

/* This function updates the pheromone of each cell, **after** the ants have moved.
 * Assume that each cell occupied by an ant, gets a pheromone increase of 5%
 * and empty cells lose the 10% of their current pheromone
 */
void AntColonySystem::update_pheromone()
{
        for (std::size_t i = 0; i < N; i++)
                for (std::size_t j = 0; j < N; j++)
                        if(grid[i * N + j].ant_present)
                                grid[i * N + j].pher_amount += 0.05 * grid[i * N + j].pher_amount;
                        else
                                grid[i * N + j].pher_amount -= 0.1 * grid[i * N + j].pher_amount;
}

void AntColonySystem::advance_system(const int steps)
{
        for (int s = 0; s < steps; s++)
        {
                move_ants();
                /* After moving the ants, update the pheromone amount of each cell
                 * (vacated or newly occupied). 
                 * The following call updates the cells occupied by ants and the empty cells
                 * of the grid
                 */
                update_pheromone();
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
        const int ant_count = std::atoi(argv[2]); // number of ants to place on the 2D grid
        const float tmax = 0.01; // max sim time
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

        const std::string ant_grid_filename = "Ant_grid_serial.dat";
        system.write_grid_status(ant_grid_filename); // write logging data

        return 0;
}