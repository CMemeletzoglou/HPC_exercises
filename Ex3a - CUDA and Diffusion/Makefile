# Makefile

SHELL := /bin/bash
CXX?=g++
CXXFLAGS?= -Wall -O3 -std=c++11 -lstdc++
NVCC?=nvcc
NVCCFLAGS?=-O3 --gpu-architecture=sm_35

.DEFAULT: all

.PHONY: all
all: diffusion_serial diffusion_cuda

diffusion_serial: diffusion2d_serial.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

diffusion_cuda: diffusion2d_cuda.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $^


.PHONY: clean
clean:
	-rm -f diffusion_serial
	-rm -f diffusion_cuda
