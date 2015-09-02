#!/bin/tcsh
#PBS -q class
#PBS -l nodes=2:sixcore
#PBS -l walltime=00:10:00
#PBS -N Riley
# The below chnages the working directory to the location of
# your testmpi program
cd FourierTransform2D
# The below tells MPI to run your jobs on 16 processors
/usr/lib64/openmpi/bin/mpirun -np 16 --hostfile $PBS_NODEFILE ./fft2d


