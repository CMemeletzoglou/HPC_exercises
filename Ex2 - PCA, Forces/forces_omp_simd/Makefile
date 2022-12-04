CXX = g++
CFLAGS = -O2 -Wall -std=c++11 -g
LDFLAGS = -lm -lpthread -lz -ldl -fopenmp


all: vanilla fast

vanilla: particles.h timer.hpp utils.h
	$(CXX) $(CFLAGS) vanilla.cpp -o $@ $(LDFLAGS)

fast: particles.h timer.hpp utils.h
	$(CXX) $(CFLAGS) fast.cpp -mavx -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f vanilla fast
