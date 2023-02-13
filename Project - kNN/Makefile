DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3 -march=native
CFLAGS += -DSURROGATES -Wall
LDFLAGS += -lm 

all: gendata myknn myknn_omp myknn_mpi

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c 
	gcc $(CFLAGS) -ggdb -c gendata.c

myknn: myknn.o
	gcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.c
	gcc $(CFLAGS) -ggdb -c myknn.c

myknn_simd: myknn_simd.o
	gcc -o myknn_simd myknn_simd.o $(LDFLAGS)

myknn_simd.o: myknn.c
	gcc -DSIMD $(CFLAGS) -ggdb -c -o myknn_simd.o myknn.c

myknn_omp: myknn_omp.o
	gcc -o myknn_omp myknn_omp.o $(LDFLAGS) -fopenmp

myknn_omp.o: myknn_omp.c
	gcc $(CFLAGS) -ggdb -fopenmp -c myknn_omp.c

myknn_mpi: myknn_mpi.o
	mpicc -o myknn_mpi myknn_mpi.o $(LDFLAGS)

myknn_mpi.o: myknn_mpi.c
	mpicc -DMPI -DSIMD $(CFLAGS) -c myknn_mpi.c

clean:
	rm -f myknn *.o gendata myknn_simd myknn_omp myknn_mpi
