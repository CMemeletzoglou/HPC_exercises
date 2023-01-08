#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>
#include "timer.hpp"

typedef struct ant_s
{
        std::tuple<int, int> position;
} ant_t;

typedef struct grid_cell_s
{
        std::tuple<int, int> coords;
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

                void move_ants();
                void update_pherormone();

                const std::size_t N, N_tot;

                float curr_time;

                grid_cell_t *grid;
                int ant_count = 0;

                std::vector<ant_t> ants;
};

void AntColonySystem::initialize_system()
{
        // assume pherormone is in [0,1], with 0 being completely empty
        // and 1 corresponding to completely full of pherormone

        srand(time(NULL));

        for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                        grid[i * N + j].coords = std::make_tuple(i, j);
                        grid[i * N + j].pher_amount = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        grid[i * N + j].ant_present = rand() % 2;

                        /* if the cell has an ant assigned to it, set the ant's coordinates 
                         * to that of the cell
                         */
                        if(grid[i * N + j].ant_present)
                        {
                                grid[i * N + j].ant.position = std::make_tuple(i, j);
                                ant_count++;
                        }
                }
        
        //reserve enough space for the ants placed
        ants.reserve(ant_count);
}

void AntColonySystem::move_ants()
{
        /* Assume that an ant can move only in a "cross" fashion, i.e.
         * one cell forward, backward, left, right.
         */
        std::vector<grid_cell_t> neigh_cells;
        neigh_cells.reserve(4);

        // iterate for each ant (?)
        for (int curr_ant = 0; curr_ant < ant_count; curr_ant++)
        {
                // get the ant's coordinates
                int ant_i = std::get<0>(ants.at(curr_ant).position);
                int ant_j = std::get<1>(ants.at(curr_ant).position);

                // std::tuple<int, int> left_cell = std::make_tuple(i, j - 1);
                // std::tuple<int, int> right_cell = std::make_tuple(i, j + 1);
                // std::tuple<int, int> up_cell = std::make_tuple(i - 1, j);
                // std::tuple<int, int> down_cell = std::make_tuple(i + 1, j);

                neigh_cells.emplace_back(grid[ant_i * N + (ant_j - 1)]);     // add left cell
                neigh_cells.emplace_back(grid[ant_i * N + (ant_j + 1)]);     // add right cell
                neigh_cells.emplace_back(grid[(ant_i - 1) * N + ant_j]);     // add up cell
                neigh_cells.emplace_back(grid[(ant_i + 1) * N + ant_j]);     // add down cell

                // check if all of the neighboring cells have the same amount of pherormone
                if (std::adjacent_find(neigh_cells.begin(), neigh_cells.end(), [](auto const &left, auto const &right)
                                       { return left.pher_amount != right.pher_amount; }) == neigh_cells.end())
                                        // comparing floats like that.. not good practice
                {
                        /* Helper variables to make the below while loop stop, in case **all**
                         * of the 4 max pherormone neighboring cells, are occupied
                         */
                        int occupied_count = 0;
                        bool ant_moved = false;
                        while (!ant_moved && occupied_count < 4)
                        {
                                // select a random neighboring cell
                                int rand_index = rand() % neigh_cells.size();
                                if(neigh_cells.at(rand_index).ant_present == false)
                                {
                                        /* Update the pherormone amount on the cell to be
                                         * vacated (i.e. the ant's current cell)
                                         */
                                        float pher_loss = grid[ant_i * N + ant_j].pher_amount / 2;
                                        float pher_incr = pher_loss / 4;

                                        //distribute the pherormone loss, equally to all 4 neighboring cells
                                        grid[ant_i * N + (ant_j - 1)].pher_amount += pher_incr; // left cell
                                        grid[ant_i * N + (ant_j + 1)].pher_amount += pher_incr / 4; // right cell
                                        grid[(ant_i - 1) * N + ant_j].pher_amount += pher_incr / 4; // up cell
                                        grid[(ant_i + 1) * N + ant_j].pher_amount += pher_incr / 4; // down cell

                                        // move the ant
                                        ants.at(curr_ant).position = neigh_cells.at(rand_index).coords;

                                        int cell_i = std::get<0>(neigh_cells.at(rand_index).coords);
                                        int cell_j = std::get<1>(neigh_cells.at(rand_index).coords);

                                        // update the cell's status to occupied
                                        grid[cell_i * N + cell_j].ant_present = true;
                                        ant_moved = true;
                                }
                                else
                                        occupied_count++;
                        }
                }
                else
                {
                        auto max_pher_it = std::max_element(neigh_cells.begin(), neigh_cells.end(),
                                                        [](auto const &left, auto const &right)
                                                        {
                                                                return left.pher_amount < right.pher_amount;
                                                        });
                        float max_pher = (*max_pher_it).pher_amount;
                        int max_pher_cell_index = max_pher_it - neigh_cells.begin(); // get the index of the iterator

                        // check if the max_pher is empty
                        if(neigh_cells.at(max_pher_cell_index).ant_present == false)
                        {
                                /* Update the pherormone amount on the cell to be
                                 * vacated (i.e. the ant's current cell)
                                 */
                                float pher_loss = grid[ant_i * N + ant_j].pher_amount / 2;
                                float pher_incr = pher_loss / 4;

                                //distribute the pherormone loss, equally to all 4 neighboring cells
                                grid[ant_i * N + (ant_j - 1)].pher_amount += pher_incr; // left cell
                                grid[ant_i * N + (ant_j + 1)].pher_amount += pher_incr / 4; // right cell
                                grid[(ant_i - 1) * N + ant_j].pher_amount += pher_incr / 4; // up cell
                                grid[(ant_i + 1) * N + ant_j].pher_amount += pher_incr / 4; // down cell

                                // we can move there, so change the ant's coordinates
                                ants.at(curr_ant).position = neigh_cells.at(max_pher_cell_index).coords;

                                int cell_i = std::get<0>(neigh_cells.at(max_pher_cell_index).coords);
                                int cell_j = std::get<1>(neigh_cells.at(max_pher_cell_index).coords);

                                // update the cell's status to occupied
                                grid[cell_i * N + cell_j].ant_present = true;
                        }
                }
        }
}

void AntColonySystem::update_pherormone()
{
        /* Assume that each cell occupied by an ant, gets an pherormone increase of 5%
         * and empty cells lose the 10% of their pherormone amount
         */
        for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                        if(grid[i * N + j].ant_present)
                                grid[i * N + j].pher_amount += 0.05 * grid[i * N + j].pher_amount;
                        else
                                grid[i * N + j].pher_amount -= 0.1 * grid[i * N + j].pher_amount;

}

void AntColonySystem::advance_system(float time) // TODO: use the time arg (?)
{
        move_ants();
        //then update the pherormone amounts of each cell (vacated or newly occupied)

        //update the cells occupied by ants and the empty cells
        update_pherormone();
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