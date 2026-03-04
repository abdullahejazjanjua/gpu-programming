#include <chrono>
#include <iostream>
#include <vector>

#include "common.h"

void FillMatrix(std::vector<float> &mat);
void CheckError(const std::vector<float> &mat3_cpu,
                const std::vector<float> &mat3_gpu, int n, int m);
void MatMulCpu(const std::vector<float> &mat1, const std::vector<float> &mat2,
               std::vector<float> &mat3, int n, int m, int k);
void ComputeTime(const std::vector<float> &mat1, const std::vector<float> &mat2,
                 std::vector<float> &mat3, int n, int m, int k,
                 void (*func)(const std::vector<float> &,
                               const std::vector<float> &,
                               std::vector<float> &, int, int, int));
void transpose(std::vector<float>& src, std::vector<float>& dst, int m, int k);

    int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "USAGE: ./a.out <n> <m> <k>\n";
    return 1;
  }

  int n = std::stoi(argv[1]);
  int m = std::stoi(argv[2]);
  int k = std::stoi(argv[3]);

  std::vector<float> mat1(n * k);
  std::vector<float> mat2(m * k);
  std::vector<float> mat2_T(k * m);
  std::vector<float> mat3_cpu(n * m);
  std::vector<float> mat3_gpu_col_major(n * m);
  std::vector<float> mat3_gpu_row_major(n * m);

  FillMatrix(mat1);
  FillMatrix(mat2);
  
  transpose(mat2, mat2_T, m, k);
  
  ComputeTime(mat1, mat2, mat3_cpu, n, m, k, MatMulCpu);
  
  std::cout << "Row Major:= ";
  MatMulGpu(mat1.data(), mat2_T.data(), mat3_gpu_row_major.data(), n, m, k, true);
  
  std::cout << "Col Major:= ";
  MatMulGpu(mat1.data(), mat2.data(), mat3_gpu_col_major.data(), n, m, k, false);
  
  CheckError(mat3_cpu, mat3_gpu_col_major, n, m);
  CheckError(mat3_cpu, mat3_gpu_row_major, n, m);
}

void FillMatrix(std::vector<float> &mat) {
  for (auto &value : mat) {
    value = (float)(rand()) / (float)(RAND_MAX);
  }
}

void MatMulCpu(const std::vector<float> &mat1, const std::vector<float> &mat2,
               std::vector<float> &mat3, int n, int m, int k) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      float pvalue = 0.0f;
      for (int l = 0; l < k; l++) {
        pvalue += (mat1[i * k + l] * mat2[j * k + l]);
      }
      mat3[i * m + j] = pvalue;
    }
  }
}

void CheckError(const std::vector<float> &mat3_cpu,
                const std::vector<float> &mat3_gpu, int n, int m) {

  double error = 0.0;
  float eps = 1e-8;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {

      float cpu_val = mat3_cpu[i * m + j];
      float gpu_val = mat3_gpu[i * m + j];

      error += std::abs(cpu_val - gpu_val) /
               std::max(std::max(cpu_val, gpu_val), eps);
    }
  }

  double rel_error = error / (n * m);

  std::cout << "Relative error: " << rel_error << std::endl;
}

void ComputeTime(const std::vector<float> &mat1, const std::vector<float> &mat2,
                std::vector<float> &mat3, int n, int m, int k,
                 void (*func)(const std::vector<float> &,
                               const std::vector<float> &,
                               std::vector<float> &, int, int, int)) {
  auto begin = std::chrono::high_resolution_clock::now();
  func(mat1, mat2, mat3, n, m, k);
  auto end = std::chrono::high_resolution_clock::now();

  std::cout
      << "CPU TIME: "
      << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin)).count()
      << " microsecs" << std::endl;
}

void transpose(std::vector<float>& src, std::vector<float>& dst, int m, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            dst[j * m + i] = src[i * k + j];
        }
    }
}