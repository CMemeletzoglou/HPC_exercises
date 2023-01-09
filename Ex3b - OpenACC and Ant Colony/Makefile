CXXFLAGS?= -Wall -O3 -std=c++14 -g


ant_colony_serial : ant_colony_serial.cpp
	g++ $(CXXFLAGS) -o $@  $^

ants : ants.cpp
	g++ $(CXXFLAGS) -o $@  $^

clean :
	rm -f ant_colony_serial ants
