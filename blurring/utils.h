#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>

// utils.cpp
cv::Mat read_image(std::string path);

// grayscale.cu
void blur(unsigned char *data_inh, unsigned char *data_outh, int height, int width, int blurstep);

#endif