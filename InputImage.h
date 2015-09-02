// Manage a text file with a black-white image
// George F. Riley, Georgia Tech, Fall 2009

#include <fstream>

class Complex;

class InputImage {
public:
  // Load the specified black/white text-only image file
  InputImage(const char* fileName);
  // Return the width of the image
  int       GetWidth() const;
  // Return the height of the image
  int       GetHeight() const;
  // Get a pointer to the image data.  Note that the return is a 1-D
  // array, which represents a 2-d image.  The data for row 1 is immediately
  // following the data for row 0 in the 1-d array.
  Complex*  GetImageData() const;
  // Save a new black/while image from the specified data array d
  // The value saved is the magnitude of each element of d. Theh size of
  // d should be w * h.
  void      SaveImageData(const char* newFileName, Complex* d, int w, int h);
private:
  int w;         // width of image
  int h;         // height of image
  Complex* data; // The data
};
