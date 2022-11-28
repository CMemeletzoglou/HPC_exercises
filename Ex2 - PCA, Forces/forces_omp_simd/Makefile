CXX = g++-12

all: vanilla fast

vanilla: particles.h timer.hpp utils.h
	$(CXX) vanilla.cpp -std=c++11 -g -O2 -o $@

fast: particles.h timer.hpp utils.h
	@echo Fast: fill in here

.PHONY: clean
clean:
	rm -f vanilla fast
