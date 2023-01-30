DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2

CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -g -ggdb -O3
CFLAGS += -DSURROGATES -Wall
LDFLAGS += -lm 

all: gendata myknn myknn_omp

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c 
	gcc $(CFLAGS) -c gendata.c

myknn: myknn.o
	gcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.c
	gcc $(CFLAGS) -c myknn.c

myknn_simd: myknn_simd.o
	gcc -o myknn_simd myknn_simd.o $(LDFLAGS)

myknn_simd.o: myknn_simd.c
	gcc -DSIMD $(CFLAGS) -mavx2 -march=native -c myknn_simd.c

myknn_omp: myknn_omp.o
	gcc -o myknn_omp myknn_omp.o $(LDFLAGS) -fopenmp

myknn_omp.o: myknn_omp.c
	gcc $(CFLAGS) -fopenmp -c myknn_omp.c

clean:
	rm -f myknn *.o gendata myknn_omp myknn_simd
