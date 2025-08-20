#!/bin/bash
#SBATCH --job-name=trial1

# How many nodes to use
#SBATCH -N 1

#SBATCH -p compute
#SBATCH -t 04:00:00

#SBATCH -A mia357

# Recommended to be # of CPU cores desired
#SBATCH --ntasks-per-node=128
#SBATCH --error=job-%j.err

#SBATCH --output=job-%j.out
#SBATCH --mail-user=cgruich@umich.edu
#SBATCH --mail-type=ALL

#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@120

#type 'man sbatch' for more information and options

#module purge

# Must match compilation settings
# CG: OLD CODE
#module load mkl/2020.4.304
#module load intel/20.4
#module load openmpi/4.0.2-intel20.4

# user setting for OpenMP compiled programs
#export OMP_PROC_BIND=true
#export OMP_PLACES=threads
# Number of threads, not cores
#export OMP_NUM_THREADS=256

#Load the modules:
module load cpu/0.15.4 intel/19.1.1.217 intel-mpi/2019.8.254
module load nwchem/7.0.0-openblas
which nwchem
# Now run the training script
mpirun -np $SLURM_NTASKS nwchem trial1.nw

wait

