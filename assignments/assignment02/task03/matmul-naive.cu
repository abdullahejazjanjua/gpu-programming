#include "../utils/cuda_error.h"
#include <device_launch_parameters.h>
#include <driver_types.h>

__global__ void matmulKernel(float *mat1, float *mat2, float *mat3,  int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        float val = 0.0;
        for (int k = 0; k < K; k++) {
            val += (mat1[row * K + k] * mat2[k * M + col]);
        }
        mat3[row * M + col] = val;
    }
}

void mulMatrixGPUNaive(float *mat1_h, float *mat2_h, float *mat3_h,  int N, int M, int K) {
    float *mat1_d, *mat2_d, *mat3_d;
    gpuErrchk( cudaMalloc((void**)&mat1_d, ((size_t) N * K * sizeof(float))) );
    gpuErrchk( cudaMalloc((void**)&mat2_d, ((size_t) K * M * sizeof(float))) );
    gpuErrchk( cudaMalloc((void**)&mat3_d, ((size_t) N * M * sizeof(float))) );
    
    gpuErrchk( cudaMemcpy(mat1_d, mat1_h, ((size_t) N * K * sizeof(float)), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(mat2_d, mat2_h, ((size_t) K * M * sizeof(float)), cudaMemcpyHostToDevice) );
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid((M + 31) / 32, (N + 31) / 32, 1);
    matmulKernel<<<dimGrid, dimBlock>>>(mat1_d, mat2_d, mat3_d, N, M, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
    
    gpuErrchk( cudaMemcpy(mat3_h, mat3_d, ((size_t) N * M * sizeof(float)), cudaMemcpyDeviceToHost) );
    
    cudaFree(mat1_d);
    cudaFree(mat2_d);
    cudaFree(mat3_d);
    
}
