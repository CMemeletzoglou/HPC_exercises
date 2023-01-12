CXXFLAGS?= -Wall -O3 -std=c++14 -g

all: ant_colony_serial ants

ant_colony_serial : ant_colony_serial.cpp
	g++ $(CXXFLAGS) -o $@  $^

ants : ants.cpp
	g++ $(CXXFLAGS) -o $@  $^

ants_acc : ants_acc.cpp
	nvc++ -fast -acc -ta=tesla -Minfo=accel ants_acc.cpp -o ants_acc

clean :
	rm -f ant_colony_serial ants
