#include <cstdlib>

#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  if (argc < 2) {
    cerr << "Usage: ./grayscale <path-to-img>" << endl;
    exit(EXIT_FAILURE);
  }

  Mat image = read_image(argv[1]);
  return 0;
}
