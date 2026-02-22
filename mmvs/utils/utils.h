#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <iostream>

float* allocateMatrix(int num_rows, int num_cols);
float* mulMatrixCPU(float *mat1, float *mat2, int n, int m, int k);

void benchmark(int n, int m, int k);
void populate_matrix(float *mat, int rows, int cols);
void checkError(float *C_cpu, float *C_gpu_naive, float *C_gpu_tiling, int num_rows, int num_cols);
float* computeTime(float *mat1, float* mat2, int n, int m, int k, float* (*func)(float*, float*, int, int, int));



void mulMatrixGPUNaive(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K);
void mulMatrixGPUTiling(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K);


#endif