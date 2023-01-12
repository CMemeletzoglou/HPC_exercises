CXXFLAGS?= -Wall -O3 -std=c++14 -g

all: ant_colony_serial ants

ant_colony_serial : ant_colony_serial.cpp
	g++ $(CXXFLAGS) -o $@  $^

ants : ants.cpp
	g++ $(CXXFLAGS) -o $@  $^

oacc : oacc_ants.cpp
	nvc++ -fast -acc -Minfo=accel oacc_ants.cpp -o oacc_ants

clean :
	rm -f ant_colony_serial ants
