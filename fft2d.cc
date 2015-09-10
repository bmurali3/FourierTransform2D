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

void InvTransform1D(Complex* h, int w, Complex* H)
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
      wreal = wreal / w;
      wimag = -wimag / w;
      Complex wnk(wreal, wimag);
      sum = sum + wnk * h[k];
    }
    H[n] = sum;
  }
}

void Transpose(Complex* array, int width, int height)
{
  for(int i = 0; i < width; ++i)
  {
    for(int j = 0; j < height; ++j)
    {
      if(i <= j) // consider lower triangle
        continue;
      else
      {
        Complex temp = array[i + width * j];
        array[i + width * j] = array[j + width * i];
        array[j + width * i] = temp;
      }
    }
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

  int nCPU, rank;
  MPI_Comm_size(MPI_COMM_WORLD,&nCPU);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  //cout << "numtasks: " << nCPU <<"rank: " << rank << endl;
  
  MPI_Status status;

  int rowsPerCPU = height / nCPU;
  int startingRow = rowsPerCPU * rank;

  Complex* output = new Complex[width * height];

  for(int i = 0; i < rowsPerCPU; ++i)
  {
    Transform1D(data + (startingRow * width) + (i * width), width, output + (startingRow * width) + (i * width));
  }

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Recv(data + rowsPerCPU * cpu * width,
                rowsPerCPU * width * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD,
                &status);
    }

    // Copy chunk of computed 1D data from H to h

    memcpy(data, output, rowsPerCPU * width * sizeof(Complex));
  }

  if(0 != rank)
  {
    MPI_Send(output + startingRow * width,
             rowsPerCPU * width * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD);
    cout << "cpu " << rank << ": Sending 1D DFT row chunks to cpu 0." << endl;
  }

  if(0 == rank)
  {
    cout << "cpu 0: Saving 2D DFT matrix into MyAfter2d.txt." << endl;
    image.SaveImageData("MyAfter1d.txt", data, width, height);
  }

  // perform transpose
  if(0 == rank)
  {
    cout << "cpu 0: Performing transpose." << endl;
    Transpose(data, width, height);
  }

  rowsPerCPU = width / nCPU;
  startingRow = rowsPerCPU * rank;

  if(0 != rank)
    MPI_Recv(data + startingRow * height,
             rowsPerCPU * height * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD,
             &status);

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Send(data + rowsPerCPU * cpu * height,
                rowsPerCPU * height * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD);
      cout << "cpu 0: Sending 1D DFT column chunks to cpu " << rank << "." << endl;
    }
  }
  

  for(int i = 0; i < rowsPerCPU; ++i)
  {
    Transform1D(data + (startingRow * height) + (i * height), height, output + (startingRow * height) + (i * height));
  }
  
  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Recv(data + rowsPerCPU * cpu * height,
                rowsPerCPU * height * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD,
                &status);
    }

    // Copy chunk of computed 1D data from H to h

    memcpy(data, output, rowsPerCPU * height * sizeof(Complex));
  }

  if(0 != rank)
  {
    MPI_Send(output + startingRow * width,
             rowsPerCPU * height * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD);
    cout << "cpu " << rank << ": Sending 2D DFT column chunks to cpu 0." << endl;
  }

  if(0 == rank)// perform transpose
  {
    cout << "cpu 0: Performing transpose." << endl;
    Transpose(data, height, width);
    cout << "cpu 0: Saving 2D DFT matrix into MyAfter2d.txt." << endl;
    image.SaveImageData("MyAfter2d.txt", data, width, height);
  }

  delete [] output;


  /*******************************************************************************************************/


  // INVERSE FOURIER TRANSFORM

  output = new Complex[width * height];

  // perform transpose
  if(0 == rank)
  {
    cout << "cpu 0: Performing transpose." << endl;
    Transpose(data, width, height);
  }

  rowsPerCPU = width / nCPU;
  startingRow = rowsPerCPU * rank;

  if(0 != rank)
    MPI_Recv(data + startingRow * height,
             rowsPerCPU * height * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD,
             &status);

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Send(data + rowsPerCPU * cpu * height,
                rowsPerCPU * height * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD);
      cout << "cpu 0: Sending post transpose column chunks to cpu " << rank << "." << endl;
    }
  }
  
  for(int i = 0; i < rowsPerCPU; ++i)
  {
    InvTransform1D(data + (startingRow * height) + (i * height), height, output + (startingRow * height) + (i * height));
  }

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Recv(data + rowsPerCPU * cpu * height,
                rowsPerCPU * height * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD,
                &status);
    }

    // Copy chunk of computed 1D data from H to h

    memcpy(data, output, rowsPerCPU * height * sizeof(Complex));
  }

  if(0 != rank)
  {
    MPI_Send(output + startingRow * width,
             rowsPerCPU * width * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD);
    cout << "cpu " << rank << ": Sending 1D inverse DFT column chunks to cpu 0." << endl;
  }


  // perform transpose
  if(0 == rank)
  {
    cout << "cpu 0: Performing transpose." << endl;
    Transpose(data, height, width);
  }

  rowsPerCPU = height / nCPU;
  startingRow = rowsPerCPU * rank;

  if(0 != rank)
    MPI_Recv(data + startingRow * width,
             rowsPerCPU * width * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD,
             &status);

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Send(data + rowsPerCPU * cpu * width,
                rowsPerCPU * width * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD);
      cout << "cpu 0: Sending post transpose column chunks to cpu " << rank << "." << endl;
    }
  }
  
  for(int i = 0; i < rowsPerCPU; ++i)
  {
    InvTransform1D(data + (startingRow * width) + (i * width), width, output + (startingRow * width) + (i * width));
  }

  if(0 == rank)
  {
    for(int cpu = 1; cpu < nCPU; ++cpu)
    {
      MPI_Recv(data + rowsPerCPU * cpu * width,
                rowsPerCPU * width * sizeof(Complex),
                MPI_CHAR,
                cpu,
                0,
                MPI_COMM_WORLD,
                &status);
    }

    // Copy chunk of computed 1D data from H to h

    memcpy(data, output, rowsPerCPU * width * sizeof(Complex));
  }

  if(0 != rank)
  {
    MPI_Send(output + startingRow * width,
             rowsPerCPU * width * sizeof(Complex),
             MPI_CHAR,
             0,
             0,
             MPI_COMM_WORLD);
    cout << "cpu " << rank << ": Sending 2D inverse DFT row chunks to cpu 0." << endl;
  }


  if(0 == rank)// perform transpose
  {
    cout << "cpu 0: Saving 2D inverse DFT matrix into MyAfter2dInverse.txt." << endl;
    image.SaveImageData("MyAfter2dInverse.txt", data, width, height);
  }

  delete [] output;
}

int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  int rc = MPI_Init(&argc,&argv);
  if (rc != MPI_SUCCESS) {
  printf ("Error starting MPI program. Terminating.\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  }

  Transform2D(fn.c_str()); // Perform the transform.

  // Finalize MPI here

  MPI_Finalize();
}  
