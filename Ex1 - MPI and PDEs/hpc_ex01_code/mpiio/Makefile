UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	CC=gcc
endif
ifeq ($(UNAME_S),Darwin)
	CC=gcc-12
endif
MPICC?=mpicc

perf?=0
ifeq "$(perf)" "1"
	CPPFLAGS += -D_PERF_
endif

CFLAGS+=-Wall -O2
CFLAGS_THREADS=$(CFLAGS) -fopenmp

all: diffusion2d_mpi_nb_io

diffusion2d_mpi_nb_io: diffusion2d_mpi_nb_io.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f diffusion2d_mpi_nb_io *.dat
	rm -rf *.dSYM


# Execution examples
# mpirun -n 1 ./diffusion2d_mpi_nb 1 1 1024 1000 0.00000001
# mpirun -n 4 ./diffusion2d_mpi_nb 1 1 1024 1000 0.00000001
