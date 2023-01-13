CXXFLAGS?= -Wall -O3 -std=c++14 -g

all: ant_colony_serial ant_colony_serial_2 ant_colony_acc

ant_colony_serial : ant_colony_serial.cpp
	g++ $(CXXFLAGS) -o $@  $^

ant_colony_serial_2 : ant_colony_serial_2.cpp
	g++ $(CXXFLAGS) -o $@  $^

ant_colony_acc : ant_colony_acc.cpp
	nvc++ -fast -acc -ta=tesla -Minfo=accel -o $@ $^

clean :
	rm -f ant_colony_serial ant_colony_serial_2 ant_colony_acc
