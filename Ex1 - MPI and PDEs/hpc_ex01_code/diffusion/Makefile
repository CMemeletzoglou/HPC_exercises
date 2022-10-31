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

CFLAGS+=-Wall -O2 -lm
CFLAGS_THREADS=$(CFLAGS) -fopenmp

all: diffusion2d_serial diffusion2d_openmp diffusion2d_mpi_nb diffusion2d_mpi

diffusion2d_serial: diffusion2d_openmp.c
	$(CC) $(CFLAGS) -o $@ $<

diffusion2d_openmp: diffusion2d_openmp.c
	$(CC) $(CFLAGS_THREADS) -o $@ $<

diffusion2d_mpi_nb: diffusion2d_mpi_nb.c
	$(MPICC) $(CFLAGS) -o $@ $<

diffusion2d_mpi: diffusion2d_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $<

diffusion2d_mpi_square: diffusion2d_mpi_square.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f diffusion2d_serial diffusion2d_openmp diffusion2d_mpi_nb diffusion2d_mpi *.dat
	rm -rf *.dSYM


# Execution examples
# export OMP_NUM_THREADS=1; ./diffusion2d_openmp 1 1 1024 1000 0.00000001
# export OMP_NUM_THREADS=4; ./diffusion2d_openmp 1 1 1024 1000 0.00000001
# mpirun -n 1 ./diffusion2d_mpi_nb 1 1 1024 1000 0.00000001
# mpirun -n 4 ./diffusion2d_mpi_nb 1 1 1024 1000 0.00000001
