#include <iostream>
#include <vector>

void FillWithData(std::vector<float> &data);
std::vector<float> ConvCPU(const std::vector<float> &data_in,
                           const std::vector<float> &filter, int n, int m,
                           int filter_radius);
void CheckCorrectness(const std::vector<float> &dataout_cpu,
                      const std::vector<float> &dataout_gpu, int n, int m);

void conv1(float *datain, float *dataout, float *filter, int n, int m,
               int filter_radius);