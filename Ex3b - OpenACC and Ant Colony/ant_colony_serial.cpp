#include <iostream>
#include <tuple>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
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
                AntColonySystem(const std::size_t N) : N(N), N_tot(N * N)
                {
                        // allocate memory for the grid
                        grid = new (std::nothrow) grid_cell_t[N_tot];

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
                void update_pherormone();

                const std::size_t N, N_tot;

                const float dt = 1e-3; // value (??)
                float curr_time = 0.0f;

                grid_cell_t *grid;
                int ant_count = 0;

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
                                         << "Pherormone = "
                                         << grid[i * N + j].pher_amount << "\t\t\t"
                                         << "Has ant : "
                                         << grid[i * N + j].ant_present << "\n";

                out_file << "\n\n**************************Ants**************************\n\n";

                for (int i = 0; i < ant_count; i++)
                        out_file << "Ant " << i << "\t Position : (" << std::get<0>(ants.at(i).position)
                                 << " ," << std::get<1>(ants.at(i).position) << ")\n";

                out_file << "\nEND GRID STATUS\n\n";
        }

        out_file.close();
}

void AntColonySystem::initialize_system()
{
        // assume pherormone is in [0,1], with 0 being completely empty
        // and 1 corresponding to completely full of pherormone

        srand(time(NULL));

        for (std::size_t i = 0; i < N; i++)
                for (std::size_t j = 0; j < N; j++)
                {
                        grid[i * N + j].coords = std::make_tuple(i, j);
                        grid[i * N + j].pher_amount = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
                        grid[i * N + j].ant_present = rand() % 2;

                        /* If the cell has an ant assigned to it, set the ant's coordinates 
                         * to that of the cell
                         */
                        if(grid[i * N + j].ant_present)
                        {
                                ant_t ant; // create a new ant
                                ant.position = std::make_tuple(i, j); // add its coordinates
                                grid[i * N + j].ant = ant; 
                                ants.emplace_back(ant); // add ant to ants vector

                                // std::cout << "Ant " << ant_count
                                //           << "\t Position : (" << std::get<0>(ants.at(ant_count).position)
                                //           << " ," << std::get<1>(ants.at(ant_count).position) << ")\n";

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

        /* Need to check if an ant has an upper, lower, left or right cell, because
         * the ants on the grid's boundaries don't have all kinds of neighbors.
         * 
         * Solve this using a dummy cell?
         */

        grid_cell_t dummy_cell;
        dummy_cell.ant_present = true;
        dummy_cell.pher_amount = 0.0f;
        dummy_cell.coords = std::make_tuple(0, 0);

        neigh_cells = {dummy_cell, dummy_cell, dummy_cell, dummy_cell};

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

                // need to check for existence of left, right, upper and lower cells

                if(ant_j - 1 >= 0) // if there exists a left cell
                        neigh_cells.at(0) = grid[ant_i * N + (ant_j - 1)];
                if(ant_j + 1 <= static_cast<int>(N)) // if there exists a right cell
                        neigh_cells.at(1) = grid[ant_i * N + (ant_j + 1)];
                if(ant_i - 1 >= 0) // if there exists an upper cell
                        neigh_cells.at(2) = grid[(ant_i - 1) * N + ant_j];     
                if(ant_i + 1 <= static_cast<int>(N)) // if there exists a below cell
                        neigh_cells.at(3) = grid[(ant_i + 1) * N + ant_j];     


                // neigh_cells.emplace_back(grid[ant_i * N + (ant_j - 1)]);     // add left cell
                // neigh_cells.emplace_back(grid[ant_i * N + (ant_j + 1)]);     // add right cell
                // neigh_cells.emplace_back(grid[(ant_i - 1) * N + ant_j]);     // add up cell
                // neigh_cells.emplace_back(grid[(ant_i + 1) * N + ant_j]);     // add down cell

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
                                        grid[ant_i * N + (ant_j + 1)].pher_amount += pher_incr; // right cell
                                        grid[(ant_i - 1) * N + ant_j].pher_amount += pher_incr; // up cell
                                        grid[(ant_i + 1) * N + ant_j].pher_amount += pher_incr; // down cell

                                        // vacate previous cell
                                        grid[ant_i * N + ant_j].ant_present = false;
                                        // then move the ant
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
                        float max_pher = (*max_pher_it).pher_amount; // probably not needed
                        int max_pher_cell_index = max_pher_it - neigh_cells.begin(); // get the index of the iterator

                        // check if the max_pher is empty
                        if(neigh_cells.at(max_pher_cell_index).ant_present == false)
                        {
                                /* Update the pherormone amount on the cell to be
                                 * vacated (i.e. the ant's current cell)
                                 */
                                float pher_loss = grid[ant_i * N + ant_j].pher_amount / 2;
                                float pher_incr = pher_loss / 4;

                                // distribute the pherormone loss, equally to all 4 neighboring cells
                                if(ant_j - 1 >=0)
                                        grid[ant_i * N + (ant_j - 1)].pher_amount += pher_incr; // left cell
                                if(ant_j + 1 <= static_cast<int>(N))
                                        grid[ant_i * N + (ant_j + 1)].pher_amount += pher_incr; // right cell
                                if(ant_i - 1 >= 0)
                                        grid[(ant_i - 1) * N + ant_j].pher_amount += pher_incr; // up cell
                                if(ant_i + 1 <= static_cast<int>(N))
                                        grid[(ant_i + 1) * N + ant_j].pher_amount += pher_incr; // down cell

                                // first vacate previous cell
                                grid[ant_i * N + ant_j].ant_present = false;
                                // then we can move there, so change the ant's coordinates
                                ants.at(curr_ant).position = neigh_cells.at(max_pher_cell_index).coords;

                                int cell_i = std::get<0>(neigh_cells.at(max_pher_cell_index).coords);
                                int cell_j = std::get<1>(neigh_cells.at(max_pher_cell_index).coords);

                                // std::cout << "For ant " << curr_ant << "max index = " << max_pher_cell_index
                                //         << " with pher = " << neigh_cells.at(max_pher_cell_index).pher_amount
                                //         << " and coords = (" << std::get<0>(neigh_cells.at(max_pher_cell_index).coords)
                                //         << ", " << std::get<1>(neigh_cells.at(max_pher_cell_index).coords)
                                //         << ") , so cell_i = " << cell_i << "\tcell_j = " << cell_j << std::endl;

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
                //then update the pherormone amounts of each cell (vacated or newly occupied)
                //update the cells occupied by ants and the empty cells
                update_pherormone();

                curr_time += dt;
        }
}

int main(int argc, char **argv)
{
        if(argc != 2)
        {
                std::cerr << "usage: " << argv[0] << " <log2(size)>" << std::endl;
                return 1;
        }

        const std::size_t N = 1 << std::atoi(argv[1]); // grid size is 2^argv[1]
        const float tmax = 0.01; // max sim time (value?)
        const int steps_between_measurements = 100;

        // create a N x N AntColonySystem
        AntColonySystem system(N);
        
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

        // const std::string ant_grid_filename = "Ant_grid_serial.dat";
        // system.write_grid_status(ant_grid_filename); // write logging data

        return 0;
}