CXX = g++
CFLAGS = -O3 -Wall -std=c++11 -g
LDFLAGS = -fopenmp

all: vanilla fast

vanilla: particles.h timer.hpp utils.h
	$(CXX) $(CFLAGS) vanilla.cpp -o $@ $(LDFLAGS)

fast: particles.h timer.hpp utils.h
	$(CXX) $(CFLAGS) fast.cpp -march=native -mavx2 -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f vanilla fast
