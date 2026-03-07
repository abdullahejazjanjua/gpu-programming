#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

void FillWithData(std::vector<float> &data);
std::vector<float> ConvCPU(const std::vector<float> &data_in, const std::vector<float> &filter, int n, int m);
std::vector<float> ComputeTime(const std::vector<float> &datain, const std::vector<float>& filter, int n, int m, std::vector<float> (*func)(const std::vector<float>&, const std::vector<float> &, int, int));
void CheckCorrectness(const std::vector<float> &dataout_cpu, const std::vector<float> &dataout_gpu, int n, int m);

void ConvGpu(float *datain, float *dataout, float *filter, int n, int m, int conv_version);

#endif