#include <iostream>
#include <vector>
#include <chrono>

#include "macros.h"
#include "utils.h"

void FillWithData(std::vector<float> &data) {
  for (auto &value : data) {
    value = (float)(rand()) / (float)(RAND_MAX);
  }
}

std::vector<float> ConvCPU(const std::vector<float> &datain, const std::vector<float> &filter, int n, int m) {
  
    std::vector<float> dataout(n * m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
        float pvalue = 0.0f;
    
        for (int fi = 0; fi < FILTER_SIZE; ++fi) {
            for (int fj = 0; fj < FILTER_SIZE; ++fj) {
                int out_row = i + fi - FILTER_RADIUS;
                int out_col = j + fj - FILTER_RADIUS;
        
                if ((out_row >= 0 && out_row < n) && (out_col >= 0 && out_col < m))
                    pvalue += (datain[out_row * m + out_col] * filter[fi * FILTER_SIZE + fj]);
            }
        }
    
        dataout[i * m + j] = pvalue;
        }
  }

  return dataout;
}

void CheckCorrectness(const std::vector<float> &dataout_cpu, const std::vector<float> &dataout_gpu, int n, int m) {
    double error_naive = 0.0;
    float eps = 1e-8;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            float cpu_val = dataout_cpu[i * m + j];
            float naive_val = dataout_gpu[i * m + j];
        
            error_naive += std::abs(cpu_val - naive_val) / (std::max(std::max(cpu_val, naive_val), eps));
        }
    }
    
    double rel_error_naive = error_naive / (n * m);
    std::cout << "Relative error: " << rel_error_naive << std::endl;
}

std::vector<float> ComputeTime(const std::vector<float> &datain, const std::vector<float>& filter, int n, int m, std::vector<float> (*func)(const std::vector<float>&, const std::vector<float> &, int, int)) {
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<float> out = func(datain, filter, n, m);
    auto end = std::chrono::high_resolution_clock::now();
    
    std::cout
        << "CPU TIME: "
        << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)).count()
        << " microsecs" << std::endl;
        
    return out;
}