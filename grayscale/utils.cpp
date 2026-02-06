#include "utils.h"

using namespace std;
using namespace cv;

Mat read_image(string path) {
  Mat img = imread(path, IMREAD_UNCHANGED);
  if (img.empty()) {
    cerr << "Couldnot load image" << endl;
    exit(EXIT_FAILURE);
  }
  cout << "Loaded image, size: " << img.size() << endl;

  return img;
}