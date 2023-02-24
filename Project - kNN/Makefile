DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -O3 -march=native
CFLAGS += -DSURROGATES -Wall

NVCCFLAGS = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -DSURROGATES -g -O3 --gpu-architecture=sm_35

NVCFLAGS = -acc -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -DSURROGATES -g -fast -Minfo=accel
NVCFLAGS += -ta=tesla:managed
LDFLAGS += -lm 

# Remember to set OMP_PROC_BIND to TRUE, using 'export OMP_PROC_BIND=TRUE', before executin then OpenMP version
# in order to activate thread pining/binding.

all: gendata myknn myknn_simd myknn_omp myknn_omp_simd myknn_mpi myknn_mpi_packed myknn_mpi_simd myknn_mpi_packed_simd myknn_cuda myknn_acc

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c 
	gcc $(CFLAGS) -ggdb -c gendata.c

######################################################
#-------------------- SERIAL -------------------------
myknn: myknn.o
	gcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.c
	gcc $(CFLAGS) -ggdb -c myknn.c

#-------------------- SERIAL + SIMD ------------------
myknn_simd: myknn_simd.o
	gcc -o myknn_simd myknn_simd.o $(LDFLAGS)

myknn_simd.o: myknn.c
	gcc -DSIMD -mavx $(CFLAGS) -ggdb -c -o myknn_simd.o myknn.c
######################################################

#-------------------- OpenMP -------------------------
myknn_omp: myknn_omp.o
	gcc -o myknn_omp myknn_omp.o $(LDFLAGS) -fopenmp

myknn_omp.o: myknn_omp.c
	gcc $(CFLAGS) -ggdb -fopenmp -c myknn_omp.c

#-------------------- OpenMP + SIMD ------------------
myknn_omp_simd: myknn_omp_simd.o
	gcc -o myknn_omp_simd myknn_omp_simd.o $(LDFLAGS) -fopenmp

myknn_omp_simd.o: myknn_omp.c
	gcc -DSIMD -mavx $(CFLAGS) -ggdb -fopenmp -o myknn_omp_simd.o -c myknn_omp.c
######################################################

#-------------------- MPI ----------------------------
myknn_mpi: myknn_mpi.o
	mpicc -o myknn_mpi myknn_mpi.o $(LDFLAGS)

myknn_mpi.o: myknn_mpi.c
	mpicc -DMPI $(CFLAGS) -c myknn_mpi.c

#-------------------- MPI + SIMD ---------------------
myknn_mpi_simd: myknn_mpi_simd.o
	mpicc -o myknn_mpi_simd myknn_mpi_simd.o $(LDFLAGS)

myknn_mpi_simd.o: myknn_mpi.c
	mpicc -DMPI -DSIMD -mavx $(CFLAGS) -o myknn_mpi_simd.o -c myknn_mpi.c

#-------------------- MPI Packed version -------------
myknn_mpi_packed: myknn_mpi_packed.o
	mpicc -o myknn_mpi_packed myknn_mpi_packed.o $(LDFLAGS)

myknn_mpi_packed.o: myknn_mpi_packed.c
	mpicc -DMPI $(CFLAGS) -c myknn_mpi_packed.c

#-------------------- MPI Packed + SIMD version ------
myknn_mpi_packed_simd: myknn_mpi_packed_simd.o
	mpicc -o myknn_mpi_packed_simd myknn_mpi_packed_simd.o $(LDFLAGS)

myknn_mpi_packed_simd.o: myknn_mpi_packed.c
	mpicc -DMPI -DSIMD -mavx $(CFLAGS) -o myknn_mpi_packed_simd.o -c myknn_mpi_packed.c
######################################################
	
#-------------------- CUDA ---------------------------
myknn_cuda: myknn_cuda.o
	nvcc -o myknn_cuda myknn_cuda.o

myknn_cuda.o: myknn_cuda.cu
	nvcc $(NVCCFLAGS) -o myknn_cuda.o -c myknn_cuda.cu

#-------------------- OpenACC ------------------------
myknn_acc: myknn_acc.o
	nvc -acc -o myknn_acc myknn_acc.o

myknn_acc.o: myknn_acc.c
	nvc $(NVCFLAGS) -o myknn_acc.o -c myknn_acc.c
######################################################

clean:
	rm -f myknn *.o gendata myknn myknn_simd myknn_omp myknn_omp_simd myknn_mpi myknn_mpi_packed myknn_mpi_simd myknn_mpi_packed_simd myknn_cuda myknn_acc
