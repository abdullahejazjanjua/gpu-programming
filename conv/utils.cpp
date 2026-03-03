#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define FILTER_SIZE(r) ((2 * r) + 1)

void FillWithData(std::vector<float> &data) {
  for (auto &value : data) {
    value = (float)(rand()) / (float)(RAND_MAX);
  }
}

std::vector<float> ConvCPU(const std::vector<float> &datain,
                           const std::vector<float> &filter, int n, int m,
                           int filter_radius) {
  std::vector<float> dataout(n * m);
  int filter_size = FILTER_SIZE(filter_radius);

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      float pvalue = 0.0f;

      for (int fi = 0; fi < filter_size; ++fi) {
        for (int fj = 0; fj < filter_size; ++fj) {
          int out_row = i + fi - filter_radius;
          int out_col = j + fj - filter_radius;

          if ((out_row >= 0 && out_row < n) && (out_col >= 0 && out_col < m))
            pvalue +=
                (datain[out_row * m + out_col] * filter[fi * filter_size + fj]);
        }
      }

      dataout[i * m + j] = pvalue;
    }
  }

  return dataout;
}

void CheckCorrectness(const std::vector<float> &dataout_cpu,
                      const std::vector<float> &dataout_gpu, int n, int m) {
  double error_naive = 0.0;
  float eps = 1e-8;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float cpu_val = dataout_cpu[i * m + j];
      float naive_val = dataout_gpu[i * m + j];

      error_naive += std::abs(cpu_val - naive_val) /
                     (std::max(std::max(cpu_val, naive_val), eps));
    }
  }

  double rel_error_naive = error_naive / (n * m);
  std::cout << "Relative error: " << rel_error_naive << std::endl;
}