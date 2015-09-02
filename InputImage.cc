// Manage a text file with a black-white image
// George F. Riley, Georgia Tech, Fall 2009

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#include "InputImage.h"
#include "Complex.h"

using namespace std;

InputImage::InputImage(const char* fileName)
{
  ifstream ifs(fileName);
  if (!ifs)
    {
      cout << "Can't open image file " << fileName << endl;
      exit(1);
    }
  ifs >> w >> h;
  data = new Complex[w * h]; // Allocate the data array
  for (int r = 0; r < h; ++r)
    { // For each row
      for (int c = 0; c < w; ++c)
        {
          double real;
          ifs >> real;
          data[r * w + c] = Complex((double)real);
        }
    }
}

int InputImage::GetWidth() const
{
  return w;
}

int InputImage::GetHeight() const
{
  return h;
}

Complex* InputImage::GetImageData() const
{
  return data;
}

void InputImage::SaveImageData(const char* newFileName, Complex* d,
                               int w, int h)
{
  ofstream ofs(newFileName);
  if (!ofs)
    {
      cout << "Can't create output image " << newFileName << endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c].Mag() << " ";
        }
      ofs << endl;
    }
}

          
      
  
  
