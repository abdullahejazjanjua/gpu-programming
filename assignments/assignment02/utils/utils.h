#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <iostream>

float* allocateMatrix(int num_rows, int num_cols);
void readMatrix(std::ifstream& f, float *mat, int num_rows, int num_cols);
void printMatrix(float *mat, int num_rows, int num_cols, std::ostream& out);
float* mulMatrixCPU(float *mat1, float *mat2, int n, int m, int k);


void mulMatrixGPUNaive(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K);
void mulMatrixGPUTiling(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K);


#endif