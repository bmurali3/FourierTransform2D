// Distributed two-dimensional Discrete FFT transform
// Bharath Murali
// ECE8893 Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

using namespace std;


void Transform1D(Complex* h, int w, Complex* H)
{
  // Implement a simple 1-d DFT using the double summation equation
  // given in the assignment handout.  h is the time-domain input
  // data, w is the width (N), and H is the output array.

  for(int n = 0; n < w; ++n)
  {
    Complex sum(0, 0);
    for(int k = 0; k < w; ++k)
    {
      double theta = 2*M_PI*n*k/w;
      double wreal = cos(theta);
      double wimag = sin(theta);
      Complex wnk(wreal, wimag);
      sum = sum + wnk * h[k];
    }
    H[n] = sum;
  }
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
  // 1) Use the InputImage object to read in the Tower.txt file and
  //    find the width/height of the input image.
  // 2) Use MPI to find how many CPUs in total, and which one
  //    this process is
  // 3) Allocate an array of Complex object of sufficient size to
  //    hold the 2d DFT results (size is width * height)
  // 4) Obtain a pointer to the Complex 1d array of input data
  // 5) Do the individual 1D transforms on the rows assigned to your CPU
  // 6) Send the resultant transformed values to the appropriate
  //    other processors for the next phase.
  // 6a) To send and receive columns, you might need a separate
  //     Complex array of the correct size.
  // 7) Receive messages from other processes to collect your columns
  // 8) When all columns received, do the 1D transforms on the columns
  // 9) Send final answers to CPU 0 (unless you are CPU 0)
  //   9a) If you are CPU 0, collect all values from other processors
  //       and print out with SaveImageData().
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Step (1) in the comments is the line above.
  // Your code here, steps 2-9

  // get width and height
  int width = image.GetWidth();
  int height = image.GetHeight();

  // get pointer to 2D matrix
  Complex* data = image.GetImageData();

  Complex* inter = new Complex[width * height];

  for(int i = 0; i < height; ++i)
  {
      Transform1D(data + (i * width), width, inter + (i * width));
  }

  image.SaveImageData("bh_after1d.txt", inter, width, height);

  // perform transpose
  for(int i = 0; i < width; ++i)
  {
    for(int j = 0; j < height; ++j)
    {
      if(i <= j) // consider lower triangle
        continue;
      else
      {
        Complex temp = inter[i + width * j];
        inter[i + width * j] = inter[j + width * i];
        inter[j + width * i] = temp;
      }
    }
  }

  Complex* output = new Complex[width * height];

  for(int i = 0; i < width; ++i)
  {
      Transform1D(inter + (i * height), height, output + (i * height));
  }
  
  // perform transpose
  for(int i = 0; i < width; ++i)
  {
    for(int j = 0; j < height; ++j)
    {
      if(i <= j) // consider lower triangle
        continue;
      else
      {
        Complex temp = output[i + width * j];
        output[i + width * j] = output[j + width * i];
        output[j + width * i] = temp;
      }
    }
  }

  image.SaveImageData("bh_after2d.txt", output, width, height);
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
}  
  

  
