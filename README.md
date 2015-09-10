# FourierTransform2D
# Project 1 in ECE 6122 - Advanced Programming Techniques.

Project leverages MPI to perform forward and inverse 2D DFT. Program uses blocking send and receive to copy chunks of the matrix between CPUs for computation. Program currently reads images from text files which are in a specific format. It also requires the number of rows and columns to be multiples of the number of CPUs.
