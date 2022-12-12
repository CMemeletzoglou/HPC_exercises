# Make sure to load: gcc, mkl
CC = g++
CFLAGS = -O3 -Wall

MKL_SERIAL_LINK=
MKL_PARA_LINK=
LDFLAGS = -lm -lpthread -lz -ldl -fopenmp -llapack

all: pca_seq pca_omp

pca_seq: pca_seq.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(MKL_SERIAL_LINK) 

pca_seq.o: pca.cpp
	$(CC) $(CFLAGS) -o pca_seq.o -c pca.cpp

pca_omp: pca_omp.o
	$(CC) -o $@ $@.o $(OBJECTS) $(LDFLAGS) $(MKL_PARA_LINK) $(LDLFLAGS)

pca_omp.o: pca_omp.cpp
	$(CC) $(CFLAGS) -fopenmp -o pca_omp.o -c pca_omp.cpp

run_elvis:
	./pca_seq -m 469 -n 700 -npc 50 -if ../pca_data/elvis.bin.gz -of elvis.50.bin

run_cyclone:
	./pca_seq -m 4096 -n 4096 -npc 50 -if ../pca_data/cyclone.bin.gz -of cyclone.50.bin

run_earth:
	./pca_seq -m 9500 -n 9500 -npc 50 -if ../pca_data/earth.bin.gz -of earth.50.bin

run_omp_elvis:
	./pca_omp -m 496 -n 700 -npc 50 -if ../pca_data/elvis.bin.gz -of elvis.50.bin

run_omp_cyclone:
	./pca_omp -m 4096 -n 4096 -npc 50 -if ../pca_data/cyclone.bin.gz -of cyclone.50.bin

run_omp_earth:
	./pca_omp -m 9500 -n 9500 -npc 50 -if ../pca_data/earth.bin.gz -of earth.50.bin

clean:
	rm -f *.o *~ pca_seq pca_omp

