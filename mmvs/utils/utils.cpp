#include <cstdlib>
#include <iostream>
#include <cstring>

#include "utils.h"

using namespace std;

float* allocateMatrix(int num_rows, int num_cols) {
    float *mat = nullptr;
    try {
        mat = new float[(size_t) num_rows * num_cols]; 
    } 
    catch(const bad_alloc& e) {
        cerr << "Memory Allocation failed\n";
        exit(EXIT_FAILURE);
    }
    return mat;
}

float* mulMatrixCPU(float *mat1, float *mat2, int N, int M, int K) {
    float *mat3 = allocateMatrix(N, M);
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            float val = 0.0;
            for (int k = 0; k < K; k++) {
                val += (mat1[i * K + k] * mat2[k * M + j]);
             }
            mat3[i * M + j] = val;
        }
    }
    
    return mat3;
}

float* mmCPUStrassen(float *mat1, float *mat2, float *mat3, int size) {
    if (size <= 2 ) {
        
    }
    
    
    return nullptr;
}

void benchmark(int n, int m, int k) {
    float *A = allocateMatrix(n, k);
    float *B = allocateMatrix(k, m);
    float *C_gpu_naive = allocateMatrix(n, m);
    float *C_gpu_tiling = allocateMatrix(n, m);
    
    populate_matrix(A, n, k);
    populate_matrix(B, k, m);
 
    mulMatrixGPUNaive(A, B, C_gpu_naive, n, m, k);
    mulMatrixGPUTiling(A, B, C_gpu_tiling, n, m, k);
    float *C_cpu = computeTime(A, B, n, m, k, mulMatrixCPU);
    
    checkError(C_cpu, C_gpu_naive, C_gpu_tiling, n, m);
    
    delete[] A;
    delete[] B;
    delete[] C_gpu_naive;
    delete[] C_gpu_tiling;
    delete[] C_cpu;
}

float* computeTime(float *mat1, float* mat2, int n, int m, int k, float* (*func)(float*, float*, int, int, int)) {
    auto begin = chrono::high_resolution_clock::now();
    float *ptr = func(mat1, mat2, n, m, k);
    auto end = chrono::high_resolution_clock::now();    
    
    cout << "CPU TIME: " << (chrono::duration_cast<chrono::microseconds>(end - begin)).count() << " microsecs" << endl;
    
    return ptr;
}

void populate_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < (rows * cols); i++)
        mat[i] = (float)(rand()) / (float)(RAND_MAX);
}

void checkError(float *C_cpu, float *C_gpu_naive, float *C_gpu_tiling, int num_rows, int num_cols) {
    double error_naive = 0.0;
    double error_tiling = 0.0;
    float eps = 1e-8;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            int idx = i * num_cols + j;
            
            float cpu_val = C_cpu[idx];
            float naive_val = C_gpu_naive[idx];
            float tiling_val = C_gpu_tiling[idx];

            error_naive += abs(cpu_val - naive_val) / (std::max(std::max(cpu_val, naive_val), eps));
            error_tiling += abs(cpu_val - tiling_val) / std::max(std::max(cpu_val, tiling_val), eps);  

        }
    }

    double rel_error_naive = error_naive / (num_rows * num_cols);
    double rel_error_tiling = error_tiling / (num_rows * num_cols);
    
    if (rel_error_naive >= 1e-6)
        cout << "Relative error (naive): " << rel_error_naive << endl;
    if (rel_error_tiling >= 1e-6)
        cout << "Relative error (tiling): " << rel_error_tiling << endl;
}
