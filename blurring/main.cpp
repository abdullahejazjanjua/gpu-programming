#include <cstdlib>

#include "utils.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: ./blur <path-to-img>" << endl;
        exit(EXIT_FAILURE);
    }
    
    Mat img_in = read_image(argv[1]);
    Mat img_out = Mat(img_in.rows, img_in.cols, CV_8UC3);
    
    blur(img_in.data, img_out.data, img_in.rows, img_in.cols, 3);
    
    bool check = imwrite("lena-blur.jpg", img_out);
    if (!check) {
        cerr << "Failed to save the image" << endl;
        exit(EXIT_FAILURE);
    }
    
    return 0;
}
