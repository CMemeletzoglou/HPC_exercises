#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --account=s659

./diffusion_serial 10
./diffusion_cuda_solution 10
